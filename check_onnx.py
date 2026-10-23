# ONNX 파일의 입력·출력 모양과 연산자 목록을 확인한다
#   python check_onnx.py model.onnx
import sys
import onnx

m = onnx.load(sys.argv[1])
onnx.checker.check_model(m)
shape = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
print("opset :", [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")])
for i in m.graph.input:
    print("input :", i.name, shape(i))
for o in m.graph.output:
    print("output:", o.name, shape(o))
print("ops   :", ", ".join(sorted({n.op_type for n in m.graph.node})))
