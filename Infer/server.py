# -*- coding: utf-8 -*-
# server.py
import socket
import torch
import pickle
import struct

from model_utils import get_split_mobilenet_v2

# --- 설정 ---
SERVER_HOST = '0.0.0.0'  # 모든 IP에서 접속 허용
SERVER_PORT = 3277      # 사용할 포트 번호
# 논문에서 가장 좋은 트레이드오프를 보인 'block_16_project_BN'을 분할 지점으로 설정
SPLIT_LAYER = 'features.16.conv.2'

# --- 모델 준비 ---
# CPU 또는 GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"서버: '{device}'를 사용하여 모델을 실행합니다.")

# 모델을 분할하여 서버가 실행할 Part 2만 로드
_, model_part2 = get_split_mobilenet_v2(SPLIT_LAYER)
model_part2.to(device)
model_part2.eval()


# --- 서버 소켓 설정 ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(1)
print(f"서버가 {SERVER_HOST}:{SERVER_PORT}에서 연결을 기다리는 중입니다...")

client_socket, addr = server_socket.accept()
print(f"{addr}에서 연결되었습니다.")

# --- 데이터 수신 및 추론 루프 ---
try:
    while True:
        # 1. 데이터 크기 수신 (4바이트, unsigned long long)
        data_size_bytes = client_socket.recv(8)
        if not data_size_bytes:
            print("클라이언트 연결이 끊겼습니다.")
            break
        
        data_size = struct.unpack('>Q', data_size_bytes)[0]

        # 2. 중간 텐서 데이터 수신
        payload = b""
        while len(payload) < data_size:
            packet = client_socket.recv(4096)
            if not packet:
                break
            payload += packet
        
        print(f"중간 텐서 수신 완료: {len(payload)} 바이트")

        # 3. 텐서 역직렬화 및 추론
        intermediate_tensor = pickle.loads(payload)
        intermediate_tensor = intermediate_tensor.to(device)

        with torch.no_grad():
            output = model_part2(intermediate_tensor)
        
        # 4. 결과 직렬화 및 전송
        output_data = pickle.dumps(output.cpu()) # 클라이언트로 보내기 전에 CPU로 이동
        
        # 결과 데이터 크기 전송
        client_socket.sendall(struct.pack('>Q', len(output_data)))
        # 실제 결과 데이터 전송
        client_socket.sendall(output_data)
        
        print("추론 완료 및 결과 전송 완료.")

except Exception as e:
    print(f"오류 발생: {e}")

finally:
    client_socket.close()
    server_socket.close()
    print("서버를 종료합니다.")