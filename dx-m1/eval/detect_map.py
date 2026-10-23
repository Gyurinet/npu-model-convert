# ONNX 출력과 NPU 출력으로 같은 방식의 mAP를 계산한다(변환용 PC)
#   python detect_map.py det_ref.npz NPU=npu_yolo11n.npy "NPU TopK"=npu_yolo11n_topk.npy
# det_ref.npz 안의 ONNX 출력도 함께 계산한다. 신뢰도 0.001, NMS IoU 0.7, 상자 최대 300개
import sys
import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou
from yolo_common import postprocess

d = np.load(sys.argv[1])
labels_dir = str(d["labels_dir"])
iouv = np.linspace(0.5, 0.95, 10)

def labels(name, w, h):
    try:
        a = np.loadtxt(f"{labels_dir}/{name}.txt", ndmin=2)
    except OSError:
        a = np.zeros((0, 5))
    xy, wh = a[:, 1:3] * [w, h], a[:, 3:5] * [w, h]
    return a[:, 0].astype(int), np.column_stack([xy - wh / 2, xy + wh / 2])

def evaluate(outs):
    stats = []
    for name, m, y in zip(d["names"], d["meta"], outs):
        det = postprocess(y, m[0], (m[1], m[2]), conf=0.001, iou=0.7)[:300]
        tcls, tbox = labels(name, m[3], m[4])
        correct = np.zeros((len(det), 10), bool)
        if len(det) and len(tcls):
            iou = box_iou(torch.tensor(tbox), torch.tensor(det[:, :4])).numpy()
            same = tcls[:, None] == det[:, 5][None].astype(int)
            for k, t in enumerate(iouv):
                x = np.argwhere((iou >= t) & same)
                if len(x):
                    x = x[np.argsort(-iou[x[:, 0], x[:, 1]])]
                    x = x[np.unique(x[:, 1], return_index=True)[1]]
                    x = x[np.unique(x[:, 0], return_index=True)[1]]
                    correct[x[:, 1], k] = True
        stats.append((correct, det[:, 4], det[:, 5], tcls))
    tp, conf, pcls, tcls = [np.concatenate(s) for s in zip(*stats)]
    ap = ap_per_class(tp, conf, pcls, tcls)[5]                       # [클래스, IoU 10단계]
    return ap[:, 0].mean(), ap.mean()

runs = [(f"ONNX {k}", d[k]) for k in d.files if k not in ("names", "meta", "labels_dir")]
runs += [(arg.partition("=")[0], np.load(arg.partition("=")[2])) for arg in sys.argv[2:]]
for label, outs in runs:
    m50, m = evaluate(outs)
    print(f"{label:24s} mAP50 {m50:.3f}  mAP50-95 {m:.3f}")
