# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
from torch import nn

def get_split_mobilenet_v2(split_layer_name: str):
    """
    MobileNetV2를 주어진 레이어에서 분할.
    split_layer_name은 Jetson의 named_modules() 출력과 일치해야 함.
    예: '0.0', '2.conv.0.0', '16.conv.2', '18.1'
    """
    print(f"MobileNetV2 모델을 로드하고 '{split_layer_name}' 레이어에서 분할합니다.")

    # pretrained=True (Jetson 구버전 torchvision 호환)
    full_model = models.mobilenet_v2(pretrained=True)
    full_model.eval()

    # split_layer_name에서 블록 인덱스 추출 (예: '16.conv.2' → 16)
    block_index = int(split_layer_name.split('.')[0])

    # Part 1: 해당 블록까지 포함
    model_part1_features = full_model.features[:block_index + 1]

    # Part 2: 나머지 features + avgpool + classifier
    model_part2_features = full_model.features[block_index + 1:]
    model_part2 = nn.Sequential(
        model_part2_features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        full_model.classifier
    )

    print("모델 분할 완료!")
    return model_part1_features, model_part2
