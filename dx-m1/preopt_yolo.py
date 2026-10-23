# YOLO(v8·11) 후처리를 TopK 우선 구조로 바꾼 뒤 변환한다
#   python preopt_yolo.py yolo11n.onnx yolo11n.json out_topk
import sys
import onnx
import dx_com

src, config, out_dir = sys.argv[1:4]
model = onnx.load(src)
# 검출 헤드의 bbox(cv2)·class(cv3) 마지막 Conv 출력. YOLO11은 model.23, YOLOv8은 model.22
head = "/model.23/{b}.{i}/{b}.{i}.2/Conv_output_0"
layers = [{"bbox": head.format(b="cv2", i=i), "cls_conf": head.format(b="cv3", i=i)} for i in range(3)]
opt = dx_com.pre_optimize(model, passes={"yolo_postprocess": {
    "layers": layers, "num_classes": 80, "topk": 300, "input_height": 640, "input_width": 640}})
print("output:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in opt.graph.output])
onnx.save(opt, src.replace(".onnx", "_topk.onnx"))
dx_com.compile(model=src.replace(".onnx", "_topk.onnx"), output_dir=out_dir, config=config, gen_log=True)
