# -*- coding: utf-8 -*-
# model_utils.py
import torch
import torchvision.models as models
from torch import nn

def get_split_mobilenet_v2(split_layer_name: str):
    """
    사전 훈련된 MobileNetV2 모델을 지정된 레이어 이름으로 분할합니다.
    논문에서 언급된 'block_16_project_BN'을 기본 예시로 사용합니다.

    Args:
        split_layer_name (str): 분할 기준이 될 레이어의 이름.
                                e.g., 'features.2.conv.0.0', 'features.15.conv.2', 'features.16.conv.2'

    Returns:
        (nn.Sequential, nn.Sequential): 분할된 모델의 Part 1, Part 2
    """
    print(f"MobileNetV2 모델을 로드하고 '{split_layer_name}' 레이어에서 분할합니다.") #여기까지 로그찍힌 것을 확인!
    
    # 1. 사전 훈련된 MobileNetV2 모델 로드
    full_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    full_model.eval() # 추론 모드로 설정

    # 모델의 모든 레이어 이름과 인덱스를 확인 (디버깅용)
    # for i, (name, layer) in enumerate(full_model.features.named_children()):
    #     print(i, name, layer)

    # 2. 분할 지점 찾기
    split_index = -1
    for i, (name, layer) in enumerate(full_model.features.named_children()):
        # MobileNetV2의 블록 구조에 맞게 이름 확인
        # 예: 논문의 'Block_16_project_BN'은 PyTorch 모델에서 'features.16.conv.2' (BatchNormalization)에 해당
        if name == split_layer_name.split('.')[1]: # e.g., '16'
             # 블록 내의 세부 레이어까지 찾아야 함
             sub_split_index = -1
             for sub_i, (sub_name, sub_layer) in enumerate(layer.conv.named_children()):
                 full_layer_name = f"features.{name}.conv.{sub_name}"
                 if full_layer_name == split_layer_name:
                     sub_split_index = sub_i
                     break
             
             if sub_split_index != -1:
                 split_index = i
                 break

    if split_index == -1:
        raise ValueError(f"분할 지점 '{split_layer_name}'을 찾을 수 없습니다. 레이어 이름을 확인하세요.")

    # 3. 모델 분할
    # features의 마지막 레이어가 분할 지점이라면, classifier도 part2에 포함해야 함
    
    # Part 1: 입력부터 분할 지점까지
    model_part1_features = full_model.features[:split_index + 1]

    # Part 2: 분할 지점 다음부터 끝까지
    model_part2_features = full_model.features[split_index + 1:]
    
    # Part 2는 features의 나머지 부분과 avgpool, classifier를 포함해야 함
    model_part2 = nn.Sequential(
        model_part2_features,
        nn.AdaptiveAvgPool2d((1, 1)), # (O) 없는 속성 대신 풀링 레이어를 직접 추가
        nn.Flatten(1),
        full_model.classifier
    )

    print("모델 분할 완료!")
    return model_part1_features, model_part2

# --- 논문에서 언급된 분할 지점에 해당하는 PyTorch 레이어 이름 ---
# 1. Block_2_expand_layer -> 'features.2.conv.0.0' (Conv2d)
# 2. Block_15_project_layer -> 'features.15.conv.1' (Conv2d)
# 3. Block_16_project_BN_layer -> 'features.16.conv.2' (BatchNorm2d)