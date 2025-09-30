# -*- coding: utf-8 -*-
# list_layers.py
import torchvision.models as models

# pretrained=True (Jetson에서는 weights 파라미터 없음)
model = models.mobilenet_v2(pretrained=True)

for name, module in model.features.named_modules():
    # 연산이 실제 있는 레이어만 출력 (Sequential 같은 container 제외)
    if len(list(module.children())) == 0:
        print(name)
