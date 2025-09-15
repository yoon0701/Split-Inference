# -*- coding: utf-8 -*-
# jetson.py
import socket
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import struct
import time
import os
import glob
import statistics
from model_utils import get_split_mobilenet_v2

# --- 설정 ---
SERVER_HOST = '203.255.177.152'   # 연구실 서버 IP
SERVER_PORT = 3277               # 서버와 동일한 포트 번호
SPLIT_LAYER = 'features.16.conv.2'
IMAGE_FOLDER_PATH = "/nas-ssd/datasets/imagenette/inference"
NUM_TEST_IMAGES = 200  # 테스트할 이미지 개수

# --- 모델 준비 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"클라이언트: '{device}'를 사용하여 모델을 실행합니다.")

model_part1, _ = get_split_mobilenet_v2(SPLIT_LAYER)
model_part1.to(device)
model_part1.eval()

# --- 이미지 전처리 ---
preprocess = transforms.Compose([
    transforms.Resize(112),
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 이미지 리스트 수집 ---
image_files = glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*/*.jpg"))
if len(image_files) == 0:
    print(f"경로 '{IMAGE_FOLDER_PATH}'에서 이미지를 찾을 수 없습니다.")
    exit()

test_images = image_files[:NUM_TEST_IMAGES]
print(f"총 {len(test_images)}개의 이미지를 테스트합니다.")

# --- 서버 연결 ---
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    print(f"서버 {SERVER_HOST}:{SERVER_PORT}에 연결되었습니다.")

    # --- Warm-up (GPU 캐싱 / 네트워크 준비) ---
    print("Warm-up iteration 실행 중...")
    dummy_input = torch.randn(1, 3, 96, 96).to(device)
    with torch.no_grad():
        for _ in range(5):  # 5번 정도 돌리면 안정화됨
            _ = model_part1(dummy_input)
    print("Warm-up 완료!\n")

    # 성능 기록용
    rtt_list, compute_list, transfer_list, recv_list = [], [], [], []
    total_start = time.time()

    # --- 본격적인 테스트 루프 ---
    for idx, img_path in enumerate(test_images, 1):
        try:
            input_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[{idx}] 이미지 불러오기 실패: {img_path}, 오류: {e}")
            continue

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # --- Split Inference 시작 ---
        start_time = time.time()

        # 1. Compute time (Jetson)
        compute_start = time.time()
        with torch.no_grad():
            intermediate_tensor = model_part1(input_batch)
        compute_end = time.time()

        # 2. Transfer time (send to server)
        payload = pickle.dumps(intermediate_tensor.cpu())
        transfer_start = time.time()
        client_socket.sendall(struct.pack('>Q', len(payload)))
        client_socket.sendall(payload)
        transfer_end = time.time()
        print(f"[{idx}] 중간 텐서 전송 완료: {len(payload)} 바이트, 파일: {os.path.basename(img_path)}")

        # 3. Receive time (server → client)
        recv_start = time.time()
        result_size_bytes = client_socket.recv(8)
        result_size = struct.unpack('>Q', result_size_bytes)[0]

        result_payload = b""
        while len(result_payload) < result_size:
            packet = client_socket.recv(4096)
            if not packet:
                break
            result_payload += packet
        recv_end = time.time()

        if len(result_payload) != result_size:
            raise ValueError(f"데이터 크기 불일치: 기대={result_size}, 수신={len(result_payload)}")

        final_output = pickle.loads(result_payload)

        # 4. RTT
        end_time = time.time()
        compute_time = compute_end - compute_start
        transfer_time = transfer_end - transfer_start
        recv_time = recv_end - recv_start
        rtt = end_time - start_time

        compute_list.append(compute_time)
        transfer_list.append(transfer_time)
        recv_list.append(recv_time)
        rtt_list.append(rtt)

        print(f"[{idx}] 최종 결과 수신 완료")
        print(f"   Compute: {compute_time:.4f} 초 | Transfer: {transfer_time:.4f} 초 | Receive: {recv_time:.4f} 초 | RTT: {rtt:.4f} 초")

        # --- 결과 해석 ---
        probabilities = torch.nn.functional.softmax(final_output[0], dim=0)
        try:
            with open("imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            print(f"   Top-5 예측 결과:")
            for i in range(top5_prob.size(0)):
                print(f"      {i+1}. {categories[top5_catid[i]]}: {top5_prob[i].item():.4f}")
        except FileNotFoundError:
            print(f"   'imagenet_classes.txt' 없음 → Top-1 인덱스만 출력")
            print(f"   예측 인덱스: {torch.argmax(probabilities).item()}")

    # --- 전체 통계 ---
    total_end = time.time()
    total_time = total_end - total_start
    num_images = len(rtt_list)
    if num_images > 0:
        avg_compute = statistics.mean(compute_list)
        avg_transfer = statistics.mean(transfer_list)
        avg_recv = statistics.mean(recv_list)
        avg_rtt = statistics.mean(rtt_list)
        fps = num_images / total_time

        print("\n===== 최종 성능 통계 =====")
        print(f"분할 지점: {SPLIT_LAYER}")
        print(f"총 이미지 수: {num_images}")
        print(f"총 소요 시간: {total_time:.4f} 초")
        print(f"평균 Compute Time: {avg_compute:.4f} 초")
        print(f"평균 Transfer Time: {avg_transfer:.4f} 초")
        print(f"평균 Receive Time: {avg_recv:.4f} 초")
        print(f"평균 RTT: {avg_rtt:.4f} 초")
        print(f"FPS: {fps:.2f}")

except Exception as e:
    print(f"오류 발생: {e}")

finally:
    client_socket.close()
    print("연결을 종료합니다.")