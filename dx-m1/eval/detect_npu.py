# 검출 모델을 장비에서 실행해 원시 출력과 단계별 시간을 저장한다(DX-M1을 꽂은 장비)
#   python detect_npu.py yolo11n.dxnn det_input.npz npu_yolo11n.npy
import sys, time
import numpy as np
from dx_engine import InferenceEngine
from yolo_common import postprocess

ie = InferenceEngine(sys.argv[1])
d = np.load(sys.argv[2])
outs, t_run, t_post, npu = [], [], [], []
for x, m in zip(d["u8"], d["meta"]):
    t0 = time.perf_counter(); y = ie.run([x])[0][0]; t1 = time.perf_counter()
    postprocess(y, m[0], (m[1], m[2])); t2 = time.perf_counter()
    outs.append(y); t_run.append(t1 - t0); t_post.append(t2 - t1)
    npu.append(ie.get_npu_inference_time() / 1000)
ms = lambda v: f"{np.median(v) * 1000:.2f}"
print(f"run() {ms(t_run)} ms, NPU {np.median(npu):.2f} ms, 후처리 {ms(t_post)} ms ({len(outs)}장 중앙값)")
np.save(sys.argv[3], np.stack(outs))
