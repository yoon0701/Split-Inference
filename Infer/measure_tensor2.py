# measure_tensor.py
# -*- coding: utf-8 -*-

import torch
import torchvision.models as models
import pickle
from model_utils import get_split_mobilenet_v2

# --- 설정 ---
# 논문에서 언급된 3가지 분할 지점 중 하나를 선택하여 테스트할 수 있습니다.
# 1. 'features.2.conv.0.0'  (초기 레이어 -> 텐서가 매우 클 것으로 예상)
# 2. 'features.15.conv.1' (중간 레이어)
# 3. 'features.16.conv.2' (후기 레이어)
SPLIT_LAYER = 'features.16.conv.2'
INPUT_SIZE = (1, 3, 224, 224) # (배치, 채널, 높이, 너비)

# --- 모델 준비 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"'{device}'를 사용하여 모델을 실행합니다.")
print(f"분할 지점: '{SPLIT_LAYER}'")

# 모델을 분할하여 Part 1만 가져옴
model_part1, _ = get_split_mobilenet_v2(SPLIT_LAYER)
model_part1.to(device)
model_part1.eval()

# --- 추론 및 측정 ---
try:
    # 모델에 입력할 임의의 더미 데이터 생성
    dummy_input = torch.randn(INPUT_SIZE).to(device)

    # Part 1 추론 실행
    with torch.no_grad():
        intermediate_tensor = model_part1(dummy_input)

    print("\n--- 중간 텐서 측정 결과 ---")
    
    # 1. 메모리상 크기 측정
    num_elements = intermediate_tensor.numel()
    element_size = intermediate_tensor.element_size()
    memory_size_bytes = num_elements * element_size
    memory_size_mb = memory_size_bytes / (1024 * 1024)

    print(f"텐서 모양 (Shape): {list(intermediate_tensor.shape)}")
    print(f"메모리상 크기: {memory_size_mb:.4f} MB")

    # 2. 직렬화 크기 측정
    payload = pickle.dumps(intermediate_tensor.cpu())
    payload_size_mb = len(payload) / (1024 * 1024)
    print(f"직렬화 크기 (Pickle): {payload_size_mb:.4f} MB")

except Exception as e:
    print(f"측정 중 오류 발생: {e}")