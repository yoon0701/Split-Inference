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
Split_Inference/
├── jetson.py          # Jetson 클라이언트 코드
├── server.py          # 서버 코드
├── model_utils.py     # MobileNetV2 분할 함수
├── list_layers.py     # 모델 레이어 이름 추출
├── measure_tensor.py  # 중간 텐서 크기/메모리 분석
└── layer_names.txt    # (생성됨) 레이어 이름 목록

# Split Inference with MobileNetV2

이 프로젝트는 **Jetson (엣지 디바이스)** 과 **서버 (GPU 서버)** 간의 Split Inference를 구현한 예제입니다.  
MobileNetV2 모델을 특정 Split Layer 지점에서 분할하여,  

- **Part 1**은 Jetson에서 실행  
- **Part 2**는 서버에서 실행  

하고, 네트워크를 통해 중간 텐서를 주고받으며 **latency 분석**을 수행합니다.  

---


yaml
코드 복사

---

## ⚙️ 환경 설정
- Python 3.6+
- PyTorch
- Torchvision
- Pillow (이미지 처리)

추가 패키지 설치:
```bash
pip install torch torchvision pillow
🚀 실행 방법
1. Jetson에서 레이어 이름 추출
bash
코드 복사
python3 list_layers.py > layer_names.txt
MobileNetV2의 모든 레이어 이름을 파일로 저장합니다.

layer_names.txt는 jetson.py에서 split point 후보로 사용됩니다.

2. 서버 실행
bash
코드 복사
python3 server.py
서버는 지정된 포트(기본 3277)에서 Jetson 연결을 기다립니다.

Jetson에서 split layer 이름 + 중간 텐서가 전송되면, 해당 layer 이후 모델을 실행하고 결과를 반환합니다.

3. Jetson 실행
bash
코드 복사
python3 jetson.py
layer_names.txt에 있는 모든 split point 후보에 대해:

Jetson에서 Part 1 실행

서버로 중간 텐서 전송

서버에서 Part 2 실행 후 결과 수신

compute / transfer / receive / RTT 시간 기록

최종 결과는 split_latency_results.csv에 저장됩니다.

4. 결과 확인
bash
코드 복사
head -n 5 split_latency_results.csv
예시 출력:

python-repl
코드 복사
split_layer,avg_compute,avg_transfer,avg_recv,avg_rtt
0.0,0.0123,0.0056,0.0032,0.0211
1.conv.0.0,0.0135,0.0060,0.0030,0.0225
...
