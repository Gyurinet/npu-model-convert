# npu-model-convert

학습한 모델(PyTorch·ONNX)을 NPU에서 돌도록 바꾸고, 바꾸기 전후의 결과를 비교하는 스크립트 모음입니다.
**같은 ONNX 모델 하나**에서 두 종류의 NPU용 모델을 만들고, 같은 평가 사진으로 정확도와 속도를 비교합니다.

| NPU | 모델 형식 | 변환 도구 | 폴더 |
|---|---|---|---|
| DEEPX DX-M1(M.2 모듈) | `.dxnn` | DEEPX DX-COM | `dx-m1/` |
| Qualcomm Hexagon NPU(IQ-9075 등에 내장) | `.tflite` | TensorFlow Lite 변환기 + onnx2tf, 실행은 LiteRT + QNN 델리게이트 | `hexagon/` |

변환은 x86_64 리눅스 PC에서 하고, 결과 확인만 NPU가 있는 장비에서 합니다. 이 README의 명령과 수치는 2026년 10월 실제 PC와 장비(Qualcomm IQ-9075 EVK에 DX-M1을 꽂은 구성)에서 확인한 것입니다.
NPU 예제를 바로 돌려 보려면 [iq-9075-evk-labs](https://github.com/Gyurinet/iq-9075-evk-labs)를 함께 보세요.

## 목차

- [결과 요약](#결과-요약)
- [흐름](#흐름)
- [필요한 것](#필요한-것)
- [저장소 구성](#저장소-구성)
- [1. 공통 준비](#1-공통-준비)
- [2. DX-M1용 변환](#2-dx-m1용-변환)
- [3. 내장 NPU(Hexagon)용 변환](#3-내장-npuhexagon용-변환)
- [내 모델에 적용할 때](#내-모델에-적용할-때)
- [알아 둘 점](#알아-둘-점)
- [확인한 환경](#확인한-환경)
- [라이선스](#라이선스)

## 결과 요약

torchvision 사전 학습 모델을 ONNX로 내보낸 뒤 두 NPU용으로 바꿔, Imagenette 검증 사진 500장(ImageNet 1,000종 중 10종)의 top-1 정확도를 쟀습니다.

| 모델 | ONNX(PC, 32비트 실수) | DX-M1 8비트 | 내장 NPU 실수(fp16) | 내장 NPU 8비트 |
|---|---:|---:|---:|---:|
| MobileNetV3-Large | 79.0% | 77.2% | 78.8% | 76.8%(보정 100장으로는 74.8%) |
| ResNet50 | 84.0% | 85.0% | 84.0% | 84.2% |

| 모델 | DX-M1 NPU 계산 | 내장 NPU 실수(fp16) | 내장 NPU 8비트 | ARM CPU 8코어, 실수 |
|---|---:|---:|---:|---:|
| MobileNetV3-Large | 0.71 ms | 1.30 ms | 0.43 ms | 3.59 ms |
| ResNet50 | 2.00 ms | 3.35 ms | 0.80 ms | 37.99 ms |

- DX-M1 시간은 NPU 계산 시간(`get_npu_inference_time`), 내장 NPU와 CPU 시간은 `invoke()` 한 번의 시간입니다. 둘 다 사진 한 장씩 보낼 때의 중앙값입니다.
- 보정 사진: DX-M1은 DEEPX 보정용 사진 100장, 내장 NPU 8비트는 Imagenette 학습용 사진 500장(종류마다 50장)입니다. 평가에 쓴 검증용 사진과는 겹치지 않습니다.
- 검출 모델(YOLO11n, coco128 128장)은 DX-M1에서 mAP50-95가 ONNX 0.500 → 0.484였습니다([2-4](#2-4-검출-모델yolo11n)).

## 흐름

```text
PyTorch 모델 ──export_torchvision.py──▶ ONNX ──check_onnx.py(모양·연산 확인)
                                         │
              ┌──────────────────────────┴──────────────────────────┐
     dx-m1/: DX-COM(dxcom)                                 hexagon/to_tflite.py
              │  보정 사진으로 8비트 양자화                          │  onnx2tf → TensorFlow 변환기
              ▼                                                      ▼  (실수 모델, 8비트 모델)
           .dxnn                                                  .tflite
              │                                                      │
     DX-M1 장비: dx-m1/eval/classify_npu.py              IQ-9075: hexagon/classify_npu.py
              └────────── eval/classify_prepare.py가 만든 같은 평가 세트(cls_eval.npz)로 비교 ──┘
```

## 필요한 것

| 항목 | 내용 |
|---|---|
| 변환용 PC | x86_64, Ubuntu 24.04, Python 3.12, 메모리 16 GB 이상 권장. GPU는 없어도 됩니다(검증은 CPU만 사용) |
| DX-M1 장비(2단계) | DX-M1을 꽂은 리눅스 장비, DX-RT 3.4.2와 Python 패키지 `dx_engine`([iq-9075-evk-labs](https://github.com/Gyurinet/iq-9075-evk-labs) 시작하기 B) |
| 내장 NPU 장비(3단계) | IQ-9075 EVK 등 Hexagon NPU 장비, Qualcomm 패키지와 `ai-edge-litert`([iq-9075-evk-labs](https://github.com/Gyurinet/iq-9075-evk-labs) 시작하기 A) |
| 저장 공간(PC) | 가상환경 2개 약 6 GB, 평가·보정 데이터 약 400 MB, 변환 결과 수백 MB |

## 저장소 구성

```text
npu-model-convert/
├── export_torchvision.py     torchvision 분류 모델 → ONNX(공통)
├── check_onnx.py             ONNX 입력·출력 모양, opset, 연산 종류 확인(공통)
├── eval/classify_prepare.py  분류 평가 세트(cls_eval.npz): 전처리한 사진, 정답, 모델별 ONNX 출력(공통)
├── requirements.txt          공통 Python 패키지(변환용 PC)
├── dx-m1/                    DEEPX DX-M1
│   ├── mobilenetv3.json      224×224 분류 모델 변환 설정(보정 사진, 전처리). ResNet50에도 그대로 쓴다
│   ├── yolo11n.json          YOLO11n 검출 모델 변환 설정
│   ├── preopt_yolo.py        YOLO 후처리를 TopK 우선 구조로 바꿔 변환(dx_com.pre_optimize)
│   ├── make_reference.py, compare_npu.py   사진 폴더로 변환 전후 출력 비교(빠른 확인)
│   ├── eval/                 정확도 측정: classify_npu.py(분류), detect_prepare·detect_npu·detect_map.py(검출 mAP)
│   └── requirements.txt      DX-COM, Ultralytics
└── hexagon/                  Qualcomm Hexagon NPU
    ├── to_tflite.py          ONNX → .tflite(실수, 8비트). 정규화를 모델 안에 넣어 입력은 사진 화소값 그대로
    ├── inspect_tflite.py     입력·출력과 양자화 값, 연산 종류, NPU에 올라가는지
    ├── classify_npu.py       평가 세트로 NPU·CPU 정확도와 속도(IQ-9075에서 실행)
    └── requirements.txt      TensorFlow, onnx2tf 등
```

모델(`.pt`, `.onnx`, `.dxnn`, `.tflite`), 데이터, 변환 결과는 저장소에 넣지 않습니다(`.gitignore`).
아래 명령은 모두 저장소 폴더(`~/npu-model-convert`)를 작업 폴더로 씁니다.

## 1. 공통 준비

**1-1. 저장소와 시스템 패키지(변환용 PC)**

```bash
sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 make python3-venv
cd ~
git clone https://github.com/Gyurinet/npu-model-convert.git
```

**1-2. 데이터 받기**

보정용 사진(DEEPX 배포 100장), 평가용 Imagenette(약 340 MB), 검출 평가용 coco128을 받습니다.

```bash
cd ~/npu-model-convert
curl -LO https://sdk.deepx.ai/dataset/calibration_dataset.tar.gz
tar xzf calibration_dataset.tar.gz
mv calibration_dataset calib
curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar xzf imagenette2-320.tgz
curl -LO https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
unzip -q coco128.zip
```

내장 NPU 8비트 변환에 쓸 보정 사진 500장(Imagenette 학습용에서 종류마다 50장)도 만들어 둡니다.

```bash
cd ~/npu-model-convert
mkdir -p calib500
for d in imagenette2-320/train/*; do ls $d/*.JPEG | head -50 | xargs cp -t calib500; done
ls calib500 | wc -l
```

**1-3. 가상환경**

DX-COM과 TensorFlow는 필요한 패키지 버전이 달라 가상환경을 따로 만듭니다. 둘 다 공통 패키지(`requirements.txt`)와 PyTorch CPU판을 먼저 넣습니다.
DX-M1만 또는 내장 NPU만 쓴다면 해당 가상환경 하나만 만들면 됩니다.

```bash
cd ~/npu-model-convert
T="torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu"
python3 -m venv ~/dxcom-venv
~/dxcom-venv/bin/pip install $T
~/dxcom-venv/bin/pip install -r requirements.txt -r dx-m1/requirements.txt
python3 -m venv ~/tflite-venv
~/tflite-venv/bin/pip install $T
~/tflite-venv/bin/pip install -r requirements.txt -r hexagon/requirements.txt
```

**1-4. ONNX로 내보내기와 평가 세트**

torchvision 사전 학습 모델을 ONNX로 내보내고(입력 1×3×224×224, opset 17), 모양과 연산을 확인합니다.
평가 세트 `cls_eval.npz`에는 전처리한 사진(224×224 RGB, uint8), 정답, 모델별 ONNX 출력이 들어가며, 두 NPU의 비교에 함께 씁니다.

```bash
cd ~/npu-model-convert
source ~/tflite-venv/bin/activate
python export_torchvision.py mobilenet_v3_large mobilenetv3.onnx
python export_torchvision.py resnet50 resnet50.onnx
python check_onnx.py mobilenetv3.onnx
python eval/classify_prepare.py imagenette2-320/val mobilenetv3.onnx resnet50.onnx -o cls_eval.npz
```

```text
opset : [17]
input : input [1, 3, 224, 224]
output: logits [1, 1000]
ops   : Add, Conv, Flatten, Gemm, GlobalAveragePool, HardSigmoid, HardSwish, Mul, Relu
mobilenetv3          ONNX top-1 79.0%
resnet50             ONNX top-1 84.0%
500 images -> cls_eval.npz
```

PyTorch가 `legacy TorchScript-based ONNX export` 경고를 내지만, DX-COM 변환을 검증한 방식이라 그대로 씁니다.

## 2. DX-M1용 변환

### 2-1. 분류 모델 변환

`dx-m1/mobilenetv3.json`에는 입력 모양과 보정 설정, 보정 사진의 전처리(짧은 변 232 → 가운데 224 자르기 → RGB → 0~1 → 정규화 → NCHW)가 들어 있습니다.
DX-COM은 이 전처리를 모델 앞에 넣으므로, 변환한 `.dxnn`의 입력은 224×224 RGB 사진(uint8, HWC) 그대로입니다. ResNet50도 전처리가 같아 같은 설정을 씁니다.

```bash
cd ~/npu-model-convert
source ~/dxcom-venv/bin/activate
dxcom -m mobilenetv3.onnx -c dx-m1/mobilenetv3.json -o out_dx_mobilenetv3
dxcom -m resnet50.onnx -c dx-m1/mobilenetv3.json -o out_dx_resnet50
ls out_dx_*/*.dxnn
```

변환은 PC에서 분류 모델당 10~20초, YOLO11n TopK 변환은 1분 반쯤 걸립니다. `--gen_log --export_html`을 붙이면 층별 기록과 HTML 보고서가 함께 생깁니다.

### 2-2. 장비에서 정확도 비교

`.dxnn`과 평가 세트, 평가 스크립트를 DX-M1 장비로 옮겨 실행합니다.

```bash
# 작업용 PC
cd ~/npu-model-convert
ssh ubuntu@<장비 IP> mkdir -p convert
scp cls_eval.npz out_dx_*/*.dxnn dx-m1/eval/classify_npu.py ubuntu@<장비 IP>:convert/
```

```bash
# DX-M1 장비
cd ~/convert
source ~/npu-venv/bin/activate
python3 classify_npu.py cls_eval.npz mobilenetv3.dxnn resnet50.dxnn
```

```text
mobilenetv3          ONNX 79.0%  NPU 77.2%  1순위 일치 93.2%  NPU 0.71 ms
resnet50             ONNX 84.0%  NPU 85.0%  1순위 일치 97.6%  NPU 2.00 ms
```

"1순위 일치"는 NPU와 ONNX가 같은 종류를 1순위로 고른 비율입니다. 정확도가 비슷해도 이 값이 낮으면 양자화로 출력이 많이 달라졌다는 뜻입니다.

### 2-3. 사진 폴더로 빠르게 확인

평가 데이터 대신 현장 사진 몇 장으로 변환 전후 출력을 비교할 수도 있습니다.

```bash
# 작업용 PC
cd ~/npu-model-convert
source ~/dxcom-venv/bin/activate
python dx-m1/make_reference.py mobilenetv3.onnx <사진 폴더> ref.npz
# DX-M1 장비(ref.npz, mobilenetv3.dxnn, dx-m1/compare_npu.py를 옮긴 뒤)
python3 compare_npu.py mobilenetv3.dxnn ref.npz
```

### 2-4. 검출 모델(YOLO11n)

Ultralytics로 ONNX를 내보내고, 그대로 변환한 것과 후처리를 TopK 우선 구조로 바꿔 변환한 것(`preopt_yolo.py`)을 비교합니다.

```bash
cd ~/npu-model-convert
source ~/dxcom-venv/bin/activate
yolo export model=yolo11n.pt format=onnx opset=17 imgsz=640 simplify=True
dxcom -m yolo11n.onnx -c dx-m1/yolo11n.json -o out_dx_yolo11n
python dx-m1/preopt_yolo.py yolo11n.onnx dx-m1/yolo11n.json out_dx_topk
python dx-m1/eval/detect_prepare.py coco128 yolo11n.onnx yolo11n_topk.onnx
```

```bash
# 작업용 PC: 장비로 옮기기
cd ~/npu-model-convert
F="det_input.npz out_dx_yolo11n/yolo11n.dxnn out_dx_topk/yolo11n_topk.dxnn"
scp $F dx-m1/eval/detect_npu.py dx-m1/eval/yolo_common.py ubuntu@<장비 IP>:convert/
```

```bash
# DX-M1 장비
cd ~/convert
source ~/npu-venv/bin/activate
python3 detect_npu.py yolo11n.dxnn det_input.npz npu_yolo11n.npy
python3 detect_npu.py yolo11n_topk.dxnn det_input.npz npu_yolo11n_topk.npy
```

```bash
# 작업용 PC: 결과를 받아 mAP 계산
cd ~/npu-model-convert
scp "ubuntu@<장비 IP>:convert/npu_*.npy" .
source ~/dxcom-venv/bin/activate
python dx-m1/eval/detect_map.py det_ref.npz NPU=npu_yolo11n.npy "NPU TopK"=npu_yolo11n_topk.npy
```

```text
ONNX yolo11n             mAP50 0.668  mAP50-95 0.500
ONNX yolo11n_topk        mAP50 0.660  mAP50-95 0.496
NPU                      mAP50 0.653  mAP50-95 0.484
NPU TopK                 mAP50 0.653  mAP50-95 0.484
```

TopK 구조는 정확도가 같으면서 장비의 후처리 시간이 3.04 ms에서 0.41 ms로 줄었습니다(`detect_npu.py` 출력, 128장 중앙값).

## 3. 내장 NPU(Hexagon)용 변환

### 3-1. 변환

`hexagon/to_tflite.py`는 다음 순서로 바꿉니다.

1. ONNX를 단순화(onnxsim)하고, ReLU 뒤 MaxPool의 테두리 채우기를 0으로 고칩니다(필요한 모델만, [알아 둘 점](#알아-둘-점)). 결과는 `<이름>_prepared.onnx`입니다.
2. onnx2tf로 TensorFlow SavedModel을 만듭니다. 입력 배치가 NCHW에서 NHWC로 바뀝니다.
3. 모델 앞에 정규화(평균·표준편차)를 붙여, 입력이 사진 화소값(0~255) 그대로가 되게 합니다.
4. TensorFlow Lite 변환기로 실수 모델(`_float.tflite`)과 8비트 모델(`_int8.tflite`)을 만듭니다. 8비트는 보정 사진으로 층마다 값의 범위를 재서 정하고, 입력·출력도 8비트(uint8)입니다.

```bash
cd ~/npu-model-convert
source ~/tflite-venv/bin/activate
python hexagon/to_tflite.py mobilenetv3.onnx calib500 -o out_hx_mobilenetv3 --num 500
python hexagon/to_tflite.py resnet50.onnx calib500 -o out_hx_resnet50 --num 500
```

```text
SavedModel 입력 input [1, 224, 224, 3], 출력 output_0
저장: out_hx_mobilenetv3/mobilenetv3_float.tflite (21.9 MB)
보정: 사진 500장
저장: out_hx_mobilenetv3/mobilenetv3_int8.tflite (6.1 MB)
MaxPool 1개: 테두리를 0으로 채우게 고침
SavedModel 입력 input [1, 224, 224, 3], 출력 output_0
저장: out_hx_resnet50/resnet50_float.tflite (102.2 MB)
보정: 사진 500장
저장: out_hx_resnet50/resnet50_int8.tflite (26.3 MB)
```

MobileNetV3는 1분, ResNet50은 2분 반쯤 걸립니다. 변환 중 TensorFlow가 CUDA·oneDNN 관련 안내를 많이 찍지만 GPU가 없어도 정상입니다.

### 3-2. 장비에서 확인

`.tflite`와 평가 세트를 IQ-9075로 옮기고, 장비에도 이 저장소를 받아 스크립트를 씁니다. 장비의 가상환경에는 `ai-edge-litert`와 `numpy`가 있으면 됩니다(iq-9075-evk-labs의 `~/qai-venv` 그대로).

```bash
# IQ-9075
cd ~
git clone https://github.com/Gyurinet/npu-model-convert.git
mkdir -p ~/convert
```

```bash
# 작업용 PC
cd ~/npu-model-convert
scp cls_eval.npz out_hx_*/*.tflite ubuntu@<장비 IP>:convert/
```

먼저 모델이 NPU에 올라가는지 봅니다. NPU가 맡지 못하는 연산이 있으면 "올라가지 않음"이 나옵니다.

```bash
# IQ-9075
cd ~/convert
source ~/qai-venv/bin/activate
python3 ~/npu-model-convert/hexagon/inspect_tflite.py resnet50_int8.tflite
```

```text
입력 serving_default_image:0: [1, 224, 224, 3] uint8, 배율 1, 영점 0
출력 PartitionedCall:0: [1, 1000] uint8, 배율 0.0479108, 영점 55
연산 81개: CONV_2D 53, ADD 16, PAD 5, QUANTIZE 2, SUB 1, MUL 1, MAX_POOL_2D 1, MEAN 1, FULLY_CONNECTED 1
파일 26.3 MB
NPU: 올라감
```

평가 세트로 NPU와 ARM CPU의 정확도·속도를 비교합니다(`--cpu`를 빼면 NPU만).

```bash
# IQ-9075
cd ~/convert
source ~/qai-venv/bin/activate
H=~/npu-model-convert/hexagon
python3 $H/classify_npu.py cls_eval.npz mobilenetv3_float.tflite mobilenetv3_int8.tflite --cpu
python3 $H/classify_npu.py cls_eval.npz resnet50_float.tflite resnet50_int8.tflite --cpu
```

```text
평가 사진 500장
mobilenetv3_float.tflite  (ONNX 79.0%)
  NPU 78.8%  1순위 일치 99.6%  1.30 ms/장
  CPU 79.0%  1순위 일치 100.0%  3.59 ms/장
mobilenetv3_int8.tflite  (ONNX 79.0%)
  NPU 76.8%  1순위 일치 85.0%  0.43 ms/장
  CPU 63.2%  1순위 일치 69.4%  4.19 ms/장
평가 사진 500장
resnet50_float.tflite  (ONNX 84.0%)
  NPU 84.0%  1순위 일치 100.0%  3.35 ms/장
  CPU 84.0%  1순위 일치 100.0%  37.99 ms/장
resnet50_int8.tflite  (ONNX 84.0%)
  NPU 84.2%  1순위 일치 95.2%  0.80 ms/장
  CPU 84.6%  1순위 일치 96.2%  11.65 ms/장
```

- 실수 모델은 NPU에서 fp16으로 계산하며, ONNX와 거의 같은 결과를 냅니다. 8비트 모델은 실수 모델보다 3~4배 빠릅니다.
- ResNet50은 8비트로도 정확도가 그대로였고, MobileNetV3는 2%p 낮아졌습니다. MobileNetV3의 hard-swish·SE 블록은 양자화에 민감한 것으로 알려져 있습니다.
- 처음 실행할 때는 NPU용 그래프를 만드느라 모델마다 몇 초가 걸리고, 두 번째부터는 캐시(`~/.cache/qnn-delegate`)를 씁니다.

### 3-3. 보정 사진 수의 영향

MobileNetV3 8비트 모델을 보정 사진만 바꿔 만들면 결과가 달라집니다.

| 보정 사진 | 내장 NPU 8비트 정확도 | 1순위 일치 |
|---|---:|---:|
| DEEPX 보정용 사진 100장(`calib`) | 74.8% | 85.8% |
| Imagenette 학습용 500장(`calib500`) | 76.8% | 85.0% |

```bash
cd ~/npu-model-convert
source ~/tflite-venv/bin/activate
python hexagon/to_tflite.py mobilenetv3.onnx calib -o out_hx_mobilenetv3_c100
```

보정 사진은 실제로 들어올 사진과 비슷하고 종류가 고르게 섞일수록 좋습니다. 현장 모델이라면 현장에서 찍은 사진을 수백 장 모아 씁니다.

## 내 모델에 적용할 때

- **분류 모델**은 `export_torchvision.py` 대신 자기 모델을 `torch.onnx.export`로 내보낸 뒤 그대로 따라 하면 됩니다. 전처리가 다르면 `to_tflite.py`의 `--mean`, `--std`, `--resize`와 `dx-m1/*.json`의 `preprocessings`를 맞춥니다.
- **평가 세트**는 `eval/classify_prepare.py`의 `WNID`(폴더 이름 → 클래스 번호)를 바꿔 자기 검증 사진으로 만듭니다.
- **먼저 NPU에 올라가는지** 봅니다. 내장 NPU는 `inspect_tflite.py`, DX-M1은 `dxcom`의 변환 기록(`--gen_log`)으로 확인합니다.
- **실수 모델과 8비트 모델을 꼭 비교**합니다. 정확도뿐 아니라 "1순위 일치"와 틀린 사진을 함께 봅니다. 크게 틀어지면 보정 사진을 늘리거나 바꾸고, 그래도 안 되면 내장 NPU에서는 실수 모델(fp16)을 씁니다.
- `hexagon/to_tflite.py`는 지금 입력이 사진 한 장(1×3×H×W)인 분류 모델을 기준으로 만들었습니다. 검출 모델의 내장 NPU 변환은 아직 이 저장소에 없습니다.

## 알아 둘 점

- **내장 NPU의 그래프 캐시는 모델 구조로만 구분됩니다.** QNN 델리게이트에 `cache_dir`만 주면, 구조가 같고 가중치만 다른 모델(보정·학습을 다시 한 모델)이 먼저 만든 캐시를 그대로 써서 결과가 틀립니다. 검증 중 보정 100장 모델이 500장 모델의 캐시를 써서 두 결과가 똑같이 나왔습니다. `hexagon/classify_npu.py`는 파일 내용의 해시를 `model_token`으로 주어 모델 파일마다 캐시를 따로 둡니다.
- **MaxPool 테두리 채우기**: onnx2tf는 MaxPool의 테두리를 가장 작은 실수(약 −3.4×10³⁸)로 채웁니다. 이 값이 8비트 양자화 범위에 들어가 앞 층의 배율이 망가져, ResNet50 8비트 모델의 정확도가 0%가 되었습니다. `to_tflite.py`는 ReLU 바로 뒤의 MaxPool이면 0으로 채우게 고칩니다(ReLU 출력은 0 이상이라 최댓값이 같습니다). onnx2tf가 변환 전에 하는 단순화(onnxsim)는 이 0 채우기를 MaxPool 안으로 다시 합칠 때가 있어서, 단순화는 `to_tflite.py`가 먼저 하고 onnx2tf에서는 끕니다(`not_use_onnxsim`). 변환 뒤 `inspect_tflite.py`의 연산 목록에 `PADV2`가 없으면 고쳐진 것입니다.
- **16비트 활성값(a16w8)**: TensorFlow 2.21 변환기는 MobileNetV3의 `HARD_SWISH`를 16비트로 양자화하지 못해(`Quantization to 16x8-bit not yet supported`) 이 저장소에서는 쓰지 않습니다.
- **ARM CPU의 8비트 MobileNetV3**는 같은 파일인데도 NPU보다 정확도가 크게 낮았습니다(63.2% 대 76.8%). CPU와 NPU는 8비트 연산의 반올림과 hard-swish 계산 방식이 다를 수 있으므로, 8비트 모델은 실제로 쓸 장치에서 평가합니다.
- **DX-COM 2.4.1**에서는 JSON의 `calibration_method`, `calibration_num`이 반영되지 않았습니다. 바꾸려면 `dx_com.compile()`의 인수로 넘깁니다.
- **Ultralytics YOLO**는 AGPL-3.0입니다. 제품에 넣기 전에 라이선스를 확인합니다.

## 확인한 환경

- 변환용 PC: AMD Ryzen 7 9700X, 메모리 30 GB, Ubuntu 24.04, Python 3.12.3, GPU 미사용
- DX-M1 변환: `dx-com` 2.4.1, `ultralytics` 8.4.184, `torch` 2.14.1+cpu, `torchvision` 0.29.1+cpu, `onnx` 1.20.1, `onnxruntime` 1.26.0
- 내장 NPU 변환: `tensorflow` 2.21.0, `tf_keras` 2.21.0, `onnx2tf` 2.6.9, 위와 같은 `torch`·`onnx`
- 장비: Qualcomm IQ-9075 EVK(Ubuntu 24.04, 커널 6.8.0-1084-qcom)
  - DX-M1: DX-RT 3.4.2, 펌웨어 2.7.4, `dx_engine` 3.4.2
  - 내장 NPU: Qualcomm AI Runtime 2.49(`qairt-libs`), `tensorflow-lite-qcom-apps` 2.16.1, `ai-edge-litert` 2.3.0

## 라이선스

이 저장소의 코드 라이선스는 [`LICENSE`](LICENSE)를 따릅니다. 교육기관의 강의·실습과 기업·개인의 사내 평가·개발·시험에 쓰고 고칠 수 있으며,
판매하거나 상용 제품·서비스에 넣어 배포하려면 규리넷 주식회사의 서면 허가가 필요합니다.
모델, 데이터(Imagenette, coco128, DEEPX 보정용 사진)와 외부 소프트웨어는 각 배포처의 조건을 따릅니다.
