# server.py
import socket, torch, pickle, struct
from model_utils import get_split_mobilenet_v2

SERVER_HOST = '0.0.0.0'
SERVER_PORT = 3277
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"서버: {device} 사용")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(1)
print(f"서버가 {SERVER_HOST}:{SERVER_PORT} 대기 중...")

client_socket, addr = server_socket.accept()
print(f"{addr} 연결됨")

try:
    while True:
        # 0. 먼저 split layer 이름 길이 수신
        layer_len_bytes = client_socket.recv(4)
        if not layer_len_bytes:
            break
        layer_len = struct.unpack('>I', layer_len_bytes)[0]
        split_layer = client_socket.recv(layer_len).decode()
        print(f"[서버] Split Layer 요청: {split_layer}")

        # 해당 split layer로 모델 로드
        _, model_part2 = get_split_mobilenet_v2(split_layer)
        model_part2.to(device).eval()

        # 1. 데이터 크기 수신
        data_size_bytes = client_socket.recv(8)
        if not data_size_bytes:
            break
        data_size = struct.unpack('>Q', data_size_bytes)[0]

        # 2. 텐서 수신
        payload = b""
        while len(payload) < data_size:
            packet = client_socket.recv(4096)
            if not packet:
                break
            payload += packet

        intermediate_tensor = pickle.loads(payload).to(device)

        # 3. 추론
        with torch.no_grad():
            output = model_part2(intermediate_tensor)

        # 4. 결과 전송
        output_data = pickle.dumps(output.cpu())
        client_socket.sendall(struct.pack('>Q', len(output_data)))
        client_socket.sendall(output_data)

except Exception as e:
    print("오류:", e)

finally:
    client_socket.close()
    server_socket.close()
