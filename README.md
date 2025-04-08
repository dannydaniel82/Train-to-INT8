# Train-to-INT8
## PyTorch → ONNX → INT8 Inference Pipeline

This project aims to train PyTorch models, export them in ONNX format and apply INT8 quantization
to compare **speed and lightweight performance**


## Files

| 파일명              | 설명                                              |
|-------------------|-------------------------------------------------|
| `model.py`         | PyTorch model definition (ONNX-friendly structure)               |
| `train.py`         | Train your model(custom your model and code)               |
| `export_onnx.py`   | Export trained model in ONNX format                    |
| `quantize_onnx.py` | Apply INT8 quantization(based by ONNX Runtime)     |
| `inference.py`     | Performance Test (PyTorch, ONNX, ONNX INT8) |


