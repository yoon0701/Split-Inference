# Split Inference with MobileNetV2

이 프로젝트는 **Jetson (엣지 디바이스)** 과 **서버 (GPU 서버)** 간의 Split Inference를 구현한 예제입니다.  
MobileNetV2 모델을 특정 Split Layer 지점에서 분할하여,  

- **Part 1**은 Jetson에서 실행  
- **Part 2**는 서버에서 실행  

하고, 네트워크를 통해 중간 텐서를 주고받으며 **latency 분석**을 수행합니다.  

---

## 📂 프로젝트 구조
```plaintext
Split_Inference/
├── jetson.py          # Jetson 클라이언트 코드
├── server.py          # 서버 코드
├── model_utils.py     # MobileNetV2 분할 함수
├── list_layers.py     # 모델 레이어 이름 추출
├── measure_tensor.py  # 중간 텐서 크기/메모리 분석
└── layer_names.txt    # (생성됨) 레이어 이름 목록

## 🚀 실행 방법

### 1. Jetson에서 레이어 이름 추출
```bash
python3 list_layers.py > layer_names.txt
