# coco128로 검출 평가 세트를 만든다(변환용 PC)
#   python detect_prepare.py coco128 yolo11n.onnx yolo11n_topk.onnx
# 결과: det_input.npz(장비로 옮길 레터박스 입력), det_ref.npz(모델별 ONNX 출력, 정답 계산용 정보)
import argparse, glob, os
import cv2
import numpy as np
import onnxruntime as ort
from yolo_common import letterbox

ap = argparse.ArgumentParser()
ap.add_argument("coco_dir", help="coco128 폴더(images/train2017, labels/train2017)")
ap.add_argument("onnx", nargs="+")
ap.add_argument("--limit", type=int, default=0, help="앞에서부터 N장만(0은 전부)")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.coco_dir, "images", "train2017", "*.jpg")))
files = files[:a.limit] if a.limit else files
sess = {os.path.basename(p)[:-5]: ort.InferenceSession(p) for p in a.onnx}
u8, meta, refs = [], [], {k: [] for k in sess}
for f in files:
    img = cv2.imread(f)
    lb, r, off = letterbox(img)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    u8.append(rgb)
    meta.append([r, off[0], off[1], img.shape[1], img.shape[0]])        # 비율, 여백 x·y, 원본 너비·높이
    x = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None]
    for k, s in sess.items():
        refs[k].append(s.run(None, {s.get_inputs()[0].name: x})[0][0])
names = np.array([os.path.basename(f)[:-4] for f in files])
np.savez_compressed("det_input.npz", names=names, u8=np.stack(u8), meta=np.array(meta, np.float32))
np.savez_compressed("det_ref.npz", names=names, meta=np.array(meta, np.float32),
                    labels_dir=os.path.join(a.coco_dir, "labels", "train2017"), **{k: np.stack(v) for k, v in refs.items()})
print(f"{len(files)} images -> det_input.npz, det_ref.npz ({', '.join(sess)})")
