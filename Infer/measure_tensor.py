# -*- coding: utf-8 -*-
# measure_all_layers.py
import torch
import torchvision.models as models
import pickle

# --- 설정 ---
INPUT_SIZE = (1, 3, 224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 모델 준비 ---
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.to(device)
model.eval()

dummy_input = torch.randn(INPUT_SIZE).to(device)

# --- Hook 함수 정의 ---
def get_size_hook(name):
    def hook(module, input, output):
        num_elements = output.numel()
        element_size = output.element_size()
        memory_size_bytes = num_elements * element_size
        memory_size_mb = memory_size_bytes / (1024 * 1024)

        payload = pickle.dumps(output.cpu())
        payload_size_mb = len(payload) / (1024 * 1024)

        print(f"[{name}]")
        print(f"   Shape: {list(output.shape)}")
        print(f"   Memory (tensor): {memory_size_mb:.4f} MB")
        print(f"   Memory (pickle): {payload_size_mb:.4f} MB\n")
    return hook

# --- 모든 submodule에 hook 등록 ---
for name, module in model.named_modules():
    # 연산이 실제로 있는 레이어만 (Sequential 같은 container 제외)
    if len(list(module.children())) == 0:
        module.register_forward_hook(get_size_hook(name))

# --- Forward 실행 ---
with torch.no_grad():
    _ = model(dummy_input)
