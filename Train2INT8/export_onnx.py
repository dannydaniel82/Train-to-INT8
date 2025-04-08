# export_onnx.py
import torch
from model import SimpleClassifier

def export_to_onnx():
    model = SimpleClassifier(pretrained=False)
    model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "model.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11 # choose version
    )
    print("ONNX Export 완료")

if __name__ == "__main__":
    export_to_onnx()
