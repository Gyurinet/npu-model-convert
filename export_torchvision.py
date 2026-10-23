# torchvision 사전 학습 분류 모델을 ONNX로 내보낸다
#   python export_torchvision.py mobilenet_v3_large mobilenetv3.onnx
import sys
import torch
import torchvision

name, out = sys.argv[1], sys.argv[2]
model = torchvision.models.get_model(name, weights="DEFAULT")
model.eval()                                     # 추론 모드(Dropout·BatchNorm 고정)
dummy = torch.randn(1, 3, 224, 224)              # 배치 1, 고정 크기
torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"],
                  opset_version=17, dynamo=False)  # 가중치까지 파일 하나로 저장
print("saved:", out)
