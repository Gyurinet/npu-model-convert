# 분류 평가 세트로 ONNX(FP32)와 DX-M1(INT8) 정확도를 비교한다(DX-M1을 꽂은 장비)
#   python classify_npu.py cls_eval.npz mobilenetv3.dxnn resnet50.dxnn
# 모델 파일 이름(확장자 제외)이 classify_prepare.py에 넘긴 ONNX 이름과 같아야 한다. 다르면 경로=이름으로 지정한다.
import os, sys
import numpy as np
from dx_engine import InferenceEngine

d = np.load(sys.argv[1])
y = d["label"]
for arg in sys.argv[2:]:
    path, _, key = arg.partition("=")
    key = key or os.path.basename(path)[:-5]
    ie = InferenceEngine(path)
    out, npu = [], []
    for x in d["u8"]:
        out.append(ie.run([x])[0].reshape(-1))
        npu.append(ie.get_npu_inference_time() / 1000)
    out, ref = np.stack(out), d[key]
    print(f"{key:20s} ONNX {np.mean(ref.argmax(1) == y):.1%}  NPU {np.mean(out.argmax(1) == y):.1%}  "
          f"1순위 일치 {np.mean(out.argmax(1) == ref.argmax(1)):.1%}  NPU {np.median(npu):.2f} ms")
