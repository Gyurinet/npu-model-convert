# ONNX 분류 모델을 IQ-9075 내장 NPU(Hexagon)용 .tflite로 바꾼다(변환용 PC)
#   python to_tflite.py mobilenetv3.onnx calib -o out_mnv3 --num 500
# 결과: <이름>_float.tflite(실수, NPU에서 fp16으로 계산), <이름>_int8.tflite(8비트 양자화)
# 두 모델 모두 입력은 RGB 사진 화소값(0~255, NHWC)이고,
# 정규화(평균·표준편차)는 모델 안에서 한다.
import argparse
import glob
import os

import numpy as np
import onnx
import onnx2tf
import onnxsim
import tensorflow as tf
from PIL import Image
from torchvision import transforms

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("calib", help="보정용 사진 폴더")
ap.add_argument("-o", "--out", default="out_tflite")
ap.add_argument("--num", type=int, default=100, help="보정에 쓸 사진 수")
ap.add_argument("--resize", type=int, help="자르기 전 짧은 변 크기(기본: 입력 × 232/224)")
ap.add_argument("--mean", default="0.485,0.456,0.406", help="정규화 평균(0~1 기준)")
ap.add_argument("--std", default="0.229,0.224,0.225", help="정규화 표준편차(0~1 기준)")
a = ap.parse_args()
name = os.path.basename(a.onnx)[:-5]
mean = np.array(a.mean.split(","), np.float32) * 255
std = np.array(a.std.split(","), np.float32) * 255


def zero_pad_maxpool(m):
    """ReLU 뒤 MaxPool의 테두리 채우기를 0으로 바꾼다.

    onnx2tf는 MaxPool 테두리를 가장 작은 실수(약 -3.4e38)로 채운다(PADV2). 이 값이 8비트 양자화
    범위에 들어가면 앞 층 출력의 배율이 망가져 결과가 모두 0이 된다(ResNet50에서 정확도 0%).
    ReLU 출력은 0 이상이므로 0으로 채워도 최댓값은 같다.
    """
    producer = {o: n for n in m.graph.node for o in n.output}
    fixed = 0
    for n in list(m.graph.node):
        pads = next((at for at in n.attribute if at.name == "pads"), None)
        src = producer.get(n.input[0])
        if n.op_type != "MaxPool" or pads is None or not any(pads.ints) or src is None:
            continue
        if src.op_type not in ("Relu", "Clip"):
            continue
        k = len(pads.ints) // 2
        pad_name = n.name + "_zero_pad"
        pads_init = onnx.numpy_helper.from_array(
            np.array([0, 0, *pads.ints[:k], 0, 0, *pads.ints[k:]], np.int64), pad_name + "_pads"
        )
        m.graph.initializer.append(pads_init)
        pad = onnx.helper.make_node(
            "Pad", [n.input[0], pads_init.name], [pad_name + "_out"], name=pad_name
        )
        m.graph.node.insert(list(m.graph.node).index(n), pad)
        n.input[0] = pad.output[0]
        pads.ints[:] = [0] * len(pads.ints)
        fixed += 1
    return fixed


# 1. ONNX 정리: 단순화(onnxsim) → MaxPool 테두리 고치기
#    onnx2tf도 변환 전에 단순화를 하는데, 그때 0 채우기(Pad)를 MaxPool 안으로 다시 합칠 수 있어
#    단순화는 여기서 먼저 하고 onnx2tf에서는 끈다
os.makedirs(a.out, exist_ok=True)
m, ok = onnxsim.simplify(onnx.load(a.onnx))
if not ok:
    raise SystemExit("onnxsim으로 단순화한 모델의 출력이 원래 모델과 다릅니다")
fixed = zero_pad_maxpool(m)
if fixed:
    print(f"MaxPool {fixed}개: 테두리를 0으로 채우게 고침")
onnx_path = os.path.join(a.out, name + "_prepared.onnx")
onnx.save(m, onnx_path)

# 2. ONNX → TensorFlow SavedModel(NCHW → NHWC로 바뀐다)
sm_dir = os.path.join(a.out, "saved_model")
onnx2tf.convert(
    input_onnx_file_path=onnx_path,
    output_folder_path=sm_dir,
    tflite_backend="tf_converter",
    not_use_onnxsim=True,
    output_signaturedefs=True,
    non_verbose=True,
)
sm = tf.saved_model.load(sm_dir)
f = sm.signatures["serving_default"]
((in_name, in_spec),) = f.structured_input_signature[1].items()
out_key = list(f.structured_outputs)[0]
_, h, w, _ = in_spec.shape
print(f"SavedModel 입력 {in_name} {list(in_spec.shape)}, 출력 {out_key}")


# 3. 앞에 정규화를 붙인다: 입력은 사진 화소값(0~255) 그대로
@tf.function(input_signature=[tf.TensorSpec([1, h, w, 3], tf.float32, name="image")])
def model(image):
    return {"logits": f(**{in_name: (image - mean) / std})[out_key]}


# 4. 보정용 사진: 평가할 때와 같은 전처리(짧은 변 맞추기 → 가운데 자르기)
crop = transforms.Compose(
    [transforms.Resize(a.resize or int(h * 232 / 224)), transforms.CenterCrop((h, w))]
)
exts = (".jpg", ".jpeg", ".png")
files = sorted(p for p in glob.glob(a.calib + "/*") if p.lower().endswith(exts))[: a.num]


def representative():
    for p in files:
        yield [np.asarray(crop(Image.open(p).convert("RGB")), np.float32)[None]]


def convert(int8):
    c = tf.lite.TFLiteConverter.from_concrete_functions([model.get_concrete_function()], sm)
    if int8:
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = representative  # 이 사진들로 층마다 값의 범위를 잰다
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.uint8  # 화소값 0~255가 그대로 들어간다(배율 1)
        c.inference_output_type = tf.uint8
    path = os.path.join(a.out, f"{name}_{'int8' if int8 else 'float'}.tflite")
    with open(path, "wb") as fp:
        fp.write(c.convert())
    print(f"저장: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


convert(int8=False)
print(f"보정: 사진 {len(files)}장")
convert(int8=True)
