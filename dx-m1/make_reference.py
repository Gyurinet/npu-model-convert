# 비교용 입력(uint8 HWC)과 ONNX Runtime 기준 출력을 만든다(분류 모델, 변환용 PC에서 실행)
#   python make_reference.py mobilenetv3.onnx <이미지 폴더> ref.npz
import glob, sys
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

onnx_path, image_dir, out = sys.argv[1:4]
files = sorted(f for f in glob.glob(image_dir + "/*") if f.lower().endswith((".jpg", ".jpeg", ".png")))
crop = transforms.Compose([transforms.Resize(232), transforms.CenterCrop(224)])   # JSON의 resize·centercrop과 같게
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
sess = ort.InferenceSession(onnx_path)
name = sess.get_inputs()[0].name
u8, ref = [], []
for f in files:
    a = np.asarray(crop(Image.open(f).convert("RGB")), np.uint8)              # NPU 입력: 224x224x3 RGB
    x = ((a / 255.0 - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)  # ONNX 입력: 정규화한 NCHW
    u8.append(a)
    ref.append(sess.run(None, {name: x})[0][0])
np.savez_compressed(out, names=np.array([f.split("/")[-1] for f in files]), u8=np.stack(u8), ref=np.stack(ref))
print(f"{len(files)} images -> {out}")
