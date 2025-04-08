# inference.py

import torch
import time
import numpy as np
import onnxruntime as ort
from model import SimpleClassifier

def infer_pytorch():
    model = SimpleClassifier(pretrained=False)
    model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224)

    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end = time.time()

    print(f"✅ PyTorch Output: {output.numpy().squeeze():.4f}")
    print(f"🕒 PyTorch Inference Time: {end - start:.6f} sec")


def infer_onnx(model_path="model.onnx", name="ONNX"):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    start = time.time()
    output = sess.run(None, {input_name: dummy_input})
    end = time.time()

    print(f"✅ {name} Output: {output[0].squeeze():.4f}")
    print(f"🕒 {name} Inference Time: {end - start:.6f} sec")


if __name__ == "__main__":
    print("\n📦 PyTorch Inference")
    infer_pytorch()

    print("\n📦 ONNX Inference")
    infer_onnx("model.onnx", "ONNX")

    print("\n📦 ONNX INT8 Inference")
    infer_onnx("model_int8.onnx", "ONNX INT8")
