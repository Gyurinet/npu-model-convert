# 분류 평가 세트로 ONNX(FP32)와 IQ-9075 내장 NPU의 정확도를 비교한다(IQ-9075에서 실행)
#   python classify_npu.py cls_eval.npz mobilenetv3_float.tflite mobilenetv3_int8.tflite --cpu
# 모델 파일 이름의 첫 '_' 앞(예: mobilenetv3)이 classify_prepare.py에 넘긴
# ONNX 이름과 같아야 한다. 다르면 경로=이름으로 지정한다.
# --cpu를 붙이면 같은 모델을 ARM CPU(8코어)로도 실행한다.
import argparse
import contextlib
import ctypes
import hashlib
import os
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

DELEGATE = "/usr/lib/libQnnTFLiteDelegate.so"
# fp16 계산(실수 모델), 최고 성능 모드. 그래프 캐시를 남겨 두 번째부터 빨리 올린다
OPTIONS = {
    "backend_type": "htp",
    "htp_precision": "1",
    "htp_performance_mode": "2",
    "cache_dir": os.path.expanduser("~/.cache/qnn-delegate"),
}


@contextlib.contextmanager
def quiet():
    """델리게이트가 모델을 준비하며 찍는 기록을 숨긴다(QNN_VERBOSE=1이면 그대로 보인다)"""
    if os.environ.get("QNN_VERBOSE"):
        yield
        return
    libc = ctypes.CDLL(None)
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        libc.fflush(None)
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        libc.fflush(None)
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in [*saved, null]:
            os.close(fd)


def load(path, npu):
    """모델을 NPU(QNN 델리게이트) 또는 ARM CPU(8코어)에 올린다. NPU에 못 올리면 None"""
    if not npu:
        with quiet():  # "Created TensorFlow Lite XNNPACK delegate" 안내를 숨긴다
            it = Interpreter(path, num_threads=8)
            it.allocate_tensors()
        return it
    os.makedirs(OPTIONS["cache_dir"], exist_ok=True)
    # 캐시는 모델 구조로만 구분되어, 구조가 같고 가중치만 다른 모델(다시 보정·학습한 모델)이
    # 먼저 만든 캐시를 그대로 쓴다. 파일 내용의 해시를 model_token으로 주어 따로 저장한다
    with open(path, "rb") as fp:
        token = hashlib.sha1(fp.read()).hexdigest()[:16]
    # 델리게이트 라이브러리가 없거나 열리지 않으면(설치 문제) 그대로 오류로 멈춘다
    with quiet():
        delegate = load_delegate(DELEGATE, {**OPTIONS, "model_token": token})
    try:
        with quiet():
            it = Interpreter(path, experimental_delegates=[delegate])
            it.allocate_tensors()
        return it
    except RuntimeError:
        # NPU가 맡지 못하는 연산이 있으면 델리게이트 적용이 RuntimeError로 실패한다.
        # 이 스크립트는 NPU에 올라가는지 알려 주는 것이 목적이라, 알리고 다음 모델로 넘어간다
        print("  NPU에 올리지 못함(QNN_VERBOSE=1로 다시 실행하면 이유가 보인다)")
        return None


def run(it, images):
    """사진마다 실행해 출력(양자화를 풀어 실수로)과 1장 처리 시간 중앙값을 돌려준다"""
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = o["quantization"]
    outs, ms = [], []
    for x in images:
        it.set_tensor(i["index"], x[None].astype(i["dtype"]))
        t = time.perf_counter()
        it.invoke()
        ms.append((time.perf_counter() - t) * 1000)
        v = it.get_tensor(o["index"])[0].astype(np.float32)
        outs.append((v - z) * s if s else v)
    return np.stack(outs), np.median(ms[5:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval", help="classify_prepare.py가 만든 평가 세트(.npz)")
    ap.add_argument("models", nargs="+", help=".tflite 파일(경로=이름 형식도 된다)")
    ap.add_argument("--cpu", action="store_true", help="ARM CPU로도 실행해 비교")
    a = ap.parse_args()

    d = np.load(a.eval)
    y = d["label"]
    print(f"평가 사진 {len(y)}장")
    for arg in a.models:
        path, _, key = arg.partition("=")
        key = key or os.path.basename(path).split("_")[0]
        ref = d[key]
        print(f"{os.path.basename(path)}  (ONNX {np.mean(ref.argmax(1) == y):.1%})")
        for npu in [True, False] if a.cpu else [True]:
            it = load(path, npu)
            if it is None:
                continue
            with quiet():
                out, ms = run(it, d["u8"])
                del it  # 모델을 놓을 때 나오는 기록도 숨긴다
            print(
                f"  {'NPU' if npu else 'CPU'} {np.mean(out.argmax(1) == y):.1%}  "
                f"1순위 일치 {np.mean(out.argmax(1) == ref.argmax(1)):.1%}  {ms:.2f} ms/장"
            )


if __name__ == "__main__":
    main()
