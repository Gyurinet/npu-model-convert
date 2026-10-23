# .dxnn 출력을 ONNX 기준 출력과 비교한다(DX-M1을 꽂은 장비에서 실행)
#   python compare_npu.py mobilenetv3.dxnn ref.npz
import sys
import numpy as np
from dx_engine import InferenceEngine

ie = InferenceEngine(sys.argv[1])
d = np.load(sys.argv[2])
out = np.stack([ie.run([x])[0].reshape(-1) for x in d["u8"]])
ref = d["ref"]
cos = (out * ref).sum(1) / np.linalg.norm(out, axis=1) / np.linalg.norm(ref, axis=1)
top1 = (out.argmax(1) == ref.argmax(1)).mean()
top5 = np.mean([len(set(np.argsort(o)[-5:]) & set(np.argsort(r)[-5:])) / 5 for o, r in zip(out, ref)])
print(f"images {len(out)}  top-1 일치 {top1:.1%}  top-5 겹침 {top5:.1%}  코사인 유사도 평균 {cos.mean():.4f} (최소 {cos.min():.4f})")
for n, o, r in zip(d["names"], out, ref):
    if o.argmax() != r.argmax():
        print(f"  다름: {n}  NPU {o.argmax()}  ONNX {r.argmax()}")
