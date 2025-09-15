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

from model_utils import get_split_mobilenet_v2

# --- 설정 ---
SERVER_HOST = '203.255.177.152'   # 연구실 서버 IP
SERVER_PORT = 3277               # 서버와 동일한 포트 번호
SPLIT_LAYER = 'features.16.conv.2'
IMAGE_FOLDER_PATH = "/nas-ssd/datasets/imagenette/inference"
NUM_TEST_IMAGES = 10   # 테스트할 이미지 개수

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

# 앞에서 N개만 사용 (또는 random.sample 로 무작위 선택 가능)
test_images = image_files[:NUM_TEST_IMAGES]
print(f"총 {len(test_images)}개의 이미지를 테스트합니다.")

# --- 서버 연결 ---
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    print(f"서버 {SERVER_HOST}:{SERVER_PORT}에 연결되었습니다.")

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

        # 1. Part 1 추론 (Jetson)
        with torch.no_grad():
            intermediate_tensor = model_part1(input_batch)

        # 2. 직렬화 및 전송
        payload = pickle.dumps(intermediate_tensor.cpu())
        client_socket.sendall(struct.pack('>Q', len(payload)))
        client_socket.sendall(payload)
        print(f"[{idx}] 중간 텐서 전송 완료: {len(payload)} 바이트, 파일: {os.path.basename(img_path)}")

        # 3. 서버 결과 수신
        result_size_bytes = client_socket.recv(8)
        result_size = struct.unpack('>Q', result_size_bytes)[0]

        result_payload = b""
        while len(result_payload) < result_size:
            packet = client_socket.recv(4096)
            if not packet:
                break
            result_payload += packet

        if len(result_payload) != result_size:
            raise ValueError(f"데이터 크기 불일치: 기대={result_size}, 수신={len(result_payload)}")

        final_output = pickle.loads(result_payload)

        end_time = time.time()
        rtt = end_time - start_time
        print(f"[{idx}] 최종 결과 수신 완료, RTT: {rtt:.4f} 초")

        # --- 결과 해석 ---
        probabilities = torch.nn.functional.softmax(final_output[0], dim=0)

        try:
            with open("imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]

            top5_prob, top5_catid = torch.topk(probabilities, 5)
            print(f"[{idx}] Top-5 예측 결과:")
            for i in range(top5_prob.size(0)):
                print(f"   {i+1}. {categories[top5_catid[i]]}: {top5_prob[i].item():.4f}")
        except FileNotFoundError:
            print(f"[{idx}] 'imagenet_classes.txt' 파일 없음 → Top-1 인덱스만 출력")
            print(f"   예측 인덱스: {torch.argmax(probabilities).item()}")

except Exception as e:
    print(f"오류 발생: {e}")

finally:
    client_socket.close()
    print("연결을 종료합니다.")