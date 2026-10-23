# 검출 예제 공용: 레터박스 전처리와 후처리(NMS)
import cv2
import numpy as np

def letterbox(img, size=640, pad=114):
    """비율을 지키며 size×size로 맞추고 남는 곳을 회색으로 채운다."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), pad, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)

def postprocess(pred, r, offset, conf=0.25, iou=0.45):
    """pred: [84, N] (cx, cy, w, h, 80개 클래스 점수) → [(x1, y1, x2, y2, score, cls)]"""
    pred = pred.T
    cls = pred[:, 4:].argmax(1)
    score = pred[np.arange(len(pred)), 4 + cls]
    keep = score >= conf
    p, cls, score = pred[keep], cls[keep], score[keep]
    xywh = p[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2                       # 왼쪽 위 기준
    idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), score.tolist(), cls.tolist(), conf, iou)
    idx = np.array(idx, int).reshape(-1)
    boxes = xywh[idx]
    boxes[:, 2:] += boxes[:, :2]                          # x1y1x2y2
    boxes[:, [0, 2]] -= offset[0]; boxes[:, [1, 3]] -= offset[1]
    boxes /= r                                            # 원본 이미지 좌표로
    return np.column_stack([boxes, score[idx], cls[idx]])
