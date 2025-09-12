# -*- coding: utf-8 -*-
# jetson.py
import socket
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import struct
import time
import numpy as np

from model_utils import get_split_mobilenet_v2

# --- 설정 ---
SERVER_HOST = '203.255.177.152'  # 여기에 연구실 서버의 IP 주소를 입력하세요.
SERVER_PORT = 3277             # 서버와 동일한 포트 번호
# 논문에서 가장 좋은 트레이드오프를 보인 'block_16_project_BN'을 분할 지점으로 설정
SPLIT_LAYER = 'features.16.conv.2'
IMAGE_FOLDER_PATH = "/nas-ssd/datasets/imagenet2012/imagenet/val"
NUM_TEST_IMAGES = 50 # 테스트할 이미지 개수 (전체 다 하려면 이 부분을 수정)

# --- 모델 준비 ---
# CPU 또는 GPU 설정 (젯슨 보드에 GPU가 있다면 'cuda')
device = torch.device("cpu")
print(f"클라이언트: '{device}'를 사용하여 모델을 실행합니다.")

# 모델을 분할하여 클라이언트가 실행할 Part 1만 로드
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

try:
    input_image = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device) # 모델은 배치 입력을 기대
except FileNotFoundError:
    print(f"오류: 이미지 파일 '{IMAGE_PATH}'을 찾을 수 없습니다.")
    exit()


# --- 서버 연결 및 추론 요청 ---
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    print(f"서버 {SERVER_HOST}:{SERVER_PORT}에 연결되었습니다.")

    # --- Split Inference 시작 ---
    start_time = time.time()

    # 1. Part 1 추론 (젯슨 보드)
    with torch.no_grad():
        intermediate_tensor = model_part1(input_batch)

    # 2. 중간 텐서 직렬화 및 전송
    # 서버로 보내기 전에 텐서를 CPU로 이동시키는 것이 안전
    payload = pickle.dumps(intermediate_tensor.cpu())
    
    # 데이터 크기 전송 (8바이트, unsigned long long)
    client_socket.sendall(struct.pack('>Q', len(payload)))
    # 실제 데이터 전송
    client_socket.sendall(payload)
    print(f"중간 텐서 전송 완료: {len(payload)} 바이트")

    # 3. 서버로부터 최종 결과 수신
    # 결과 크기 수신
    result_size_bytes = client_socket.recv(8)
    result_size = struct.unpack('>Q', result_size_bytes)[0]
    
    # 실제 결과 데이터 수신
    result_payload = b""
    while len(result_payload) < result_size:
        packet = client_socket.recv(4096)
        if not packet:
            break
        result_payload += packet

    final_output = pickle.loads(result_payload)

    end_time = time.time()
    rtt = end_time - start_time
    print(f"\n--- 최종 결과 수신 완료 ---")
    print(f"Round-Trip Time (RTT): {rtt:.4f} 초")

    # --- 결과 해석 (ImageNet 클래스) ---
    probabilities = torch.nn.functional.softmax(final_output[0], dim=0)
    
    # ImageNet 클래스 레이블 로드 (labels.txt 파일이 필요)
    try:
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        print("\nTop 5 예측:")
        for i in range(top5_prob.size(0)):
            print(f"{i+1}. {categories[top5_catid[i]]}: {top5_prob[i].item():.4f}")
    except FileNotFoundError:
        print("\n'imagenet_classes.txt' 파일을 찾을 수 없어 클래스 이름을 표시할 수 없습니다.")
        print(f"가장 높은 확률을 가진 인덱스: {torch.argmax(probabilities).item()}")


except Exception as e:
    print(f"오류 발생: {e}")

finally:
    client_socket.close()
    print("연결을 종료합니다.")