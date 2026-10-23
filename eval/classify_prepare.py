# Imagenette 검증 이미지로 분류 정확도 평가 세트를 만든다(변환용 PC)
#   python classify_prepare.py imagenette2-320/val mobilenetv3.onnx resnet50.onnx -o cls_eval.npz
# 결과: uint8 입력(224x224 RGB), 정답(ImageNet 클래스 번호), 모델별 ONNX 출력
import argparse, glob, os
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

# Imagenette 폴더 이름(WordNet ID) → ImageNet 1,000개 클래스 번호
WNID = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497,
        "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}

ap = argparse.ArgumentParser()
ap.add_argument("val_dir")
ap.add_argument("onnx", nargs="+")
ap.add_argument("--per-class", type=int, default=50)
ap.add_argument("-o", "--out", default="cls_eval.npz")
a = ap.parse_args()

files = [f for w in WNID for f in sorted(glob.glob(os.path.join(a.val_dir, w, "*")))[:a.per_class]]
label = np.array([WNID[os.path.basename(os.path.dirname(f))] for f in files])
crop = transforms.Compose([transforms.Resize(232), transforms.CenterCrop(224)])     # mobilenetv3.json과 같은 전처리
u8 = np.stack([np.asarray(crop(Image.open(f).convert("RGB")), np.uint8) for f in files])
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
x = ((u8 / 255.0 - mean) / std).transpose(0, 3, 1, 2).astype(np.float32)
out = {"u8": u8, "label": label}
for path in a.onnx:
    key = os.path.basename(path)[:-5]
    sess = ort.InferenceSession(path)
    name = sess.get_inputs()[0].name
    out[key] = np.stack([sess.run(None, {name: xi[None]})[0][0] for xi in x])
    print(f"{key:20s} ONNX top-1 {np.mean(out[key].argmax(1) == label):.1%}")
np.savez_compressed(a.out, **out)
print(f"{len(files)} images -> {a.out}")
