# .tflite 모델 들여다보기: 입력·출력과 양자화 값, 연산 종류별 개수,
# 내장 NPU(QNN 델리게이트)에 올라가는지
#   python inspect_tflite.py mobilenetv3_int8.tflite   (IQ-9075에서 실행)
#   python inspect_tflite.py model.tflite --no-npu     (연산만 본다. 변환용 PC에서도 된다)
import argparse
import os
from collections import Counter

from ai_edge_litert.interpreter import Interpreter, OpResolverType

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--no-npu", action="store_true", help="NPU에 올려 보는 단계를 건너뛴다")
a = ap.parse_args()

# CPU 가속(XNNPACK)을 끄고 읽어야 모델에 들어 있는 연산 그대로 보인다
raw = Interpreter(
    a.model, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
)
for kind, details in (("입력", raw.get_input_details()), ("출력", raw.get_output_details())):
    for d in details:
        shape = [int(v) for v in d["shape"]]
        s, z = d["quantization"]
        q = f", 배율 {s:.6g}, 영점 {z}" if s else ""
        print(f"{kind} {d['name']}: {shape} {d['dtype'].__name__}{q}")
ops = Counter(o["op_name"] for o in raw._get_ops_details())
print(f"연산 {sum(ops.values())}개:", ", ".join(f"{k} {v}" for k, v in ops.most_common()))
print(f"파일 {os.path.getsize(a.model) / 1e6:.1f} MB")

if not a.no_npu:
    # 같은 폴더의 NPU 올리기 함수(IQ-9075의 QNN 델리게이트가 필요)
    from classify_npu import load, quiet

    it = load(a.model, npu=True)
    print("NPU: 올라감" if it else "NPU: 올라가지 않음")
    with quiet():
        del it  # 모델을 놓을 때 나오는 기록을 숨긴다
