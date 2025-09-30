Split_Inference/
├── jetson.py # Jetson 클라이언트 코드
├── server.py # 서버 코드
├── model_utils.py # MobileNetV2 분할 함수
├── list_layers.py # 모델 레이어 이름 추출
├── measure_tensor.py # 중간 텐서 크기/메모리 분석
└── layer_names.txt # (생성됨) 레이어 이름 목록



1. Jetson에서 레이어 이름 추출(젯슨 접속해서 실행)
python3 list_layers.py > layer_names.txt
MobileNetV2의 모든 레이어 이름을 파일로 저장합니다.

layer_names.txt는 jetson.py에서 split point 후보로 사용됩니다.

2. 서버 실행
python3 server.py


서버는 지정된 포트(기본 3277)에서 Jetson 연결을 기다립니다.

Jetson에서 split layer 이름과 중간 텐서가 전송되면, 해당 layer 이후 모델을 실행하고 결과를 반환합니다.

3. Jetson 실행
python3 jetson.py


layer_names.txt에 있는 모든 split point 후보에 대해 다음을 반복합니다:

Jetson에서 Part 1 실행

서버로 중간 텐서 전송

서버에서 Part 2 실행 후 결과 수신

compute / transfer / receive / RTT 시간 기록

최종 결과는 split_latency_results.csv에 저장됩니다.
