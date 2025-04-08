# quantize_onnx.py
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize():
    quantize_dynamic(
        model_input="model.onnx",
        model_output="model_int8.onnx",
        weight_type=QuantType.QInt8
    )
    print("✅ INT8 양자화 완료")

if __name__ == "__main__":
    quantize()
