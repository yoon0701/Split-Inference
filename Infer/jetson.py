# -*- coding: utf-8 -*-
# jetson.py
import socket, torch, torchvision.transforms as transforms
from PIL import Image
import pickle, struct, time, os, glob, csv
from model_utils import get_split_mobilenet_v2

SERVER_HOST = '203.255.177.152'
SERVER_PORT = 3277
IMAGE_FOLDER_PATH = "/nas-ssd/datasets/imagenette/inference"
NUM_TEST_IMAGES = 20   # 속도 테스트용 적당히 줄임

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 준비
preprocess = transforms.Compose([
    transforms.Resize(112),
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_files = glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*/*.jpg"))
test_images = image_files[:NUM_TEST_IMAGES]

# Split layer 후보 (measure_all_layers.py에서 뽑을 수 있음)
with open("layer_names.txt", "r") as f:
    SPLIT_LAYERS = [line.strip() for line in f.readlines() if line.strip()]

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_HOST, SERVER_PORT))
print("서버 연결됨")

# CSV 저장
with open("split_latency_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split_layer", "avg_compute", "avg_transfer", "avg_recv", "avg_rtt"])

    for split_layer in SPLIT_LAYERS:
        print(f"\n===== Split Layer: {split_layer} =====")
        model_part1, _ = get_split_mobilenet_v2(split_layer)
        model_part1.to(device).eval()

        rtt_list, compute_list, transfer_list, recv_list = [], [], [], []

        for img_path in test_images:
            input_image = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(input_image).unsqueeze(0).to(device)

            # Part1 실행
            compute_start = time.time()
            with torch.no_grad():
                intermediate_tensor = model_part1(input_tensor)
            compute_end = time.time()

            # 서버에 split layer 이름 먼저 전송
            layer_bytes = split_layer.encode()
            client_socket.sendall(struct.pack('>I', len(layer_bytes)))
            client_socket.sendall(layer_bytes)

            # 중간 텐서 전송
            payload = pickle.dumps(intermediate_tensor.cpu())
            transfer_start = time.time()
            client_socket.sendall(struct.pack('>Q', len(payload)))
            client_socket.sendall(payload)
            transfer_end = time.time()

            # 결과 수신
            recv_start = time.time()
            result_size_bytes = client_socket.recv(8)
            result_size = struct.unpack('>Q', result_size_bytes)[0]
            result_payload = b""
            while len(result_payload) < result_size:
                result_payload += client_socket.recv(4096)
            recv_end = time.time()

            final_output = pickle.loads(result_payload)

            # 시간 기록
            compute_list.append(compute_end - compute_start)
            transfer_list.append(transfer_end - transfer_start)
            recv_list.append(recv_end - recv_start)
            rtt_list.append(recv_end - compute_start)

        # 평균 기록
        avg_compute = sum(compute_list)/len(compute_list)
        avg_transfer = sum(transfer_list)/len(transfer_list)
        avg_recv = sum(recv_list)/len(recv_list)
        avg_rtt = sum(rtt_list)/len(rtt_list)
        writer.writerow([split_layer, avg_compute, avg_transfer, avg_recv, avg_rtt])
        print(f"[{split_layer}] RTT 평균: {avg_rtt:.4f}")
