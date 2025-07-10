import torch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
import os
from resnet_custom import custom_resnet18  # your custom ResNet18 with dropout

# === STEP 1: Load PyTorch .pth model and export to ONNX ===
def export_to_onnx(pth_path, onnx_path):
    model = custom_resnet18(num_classes=3)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 3, 512, 512)  # input size must match your training
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=11)
    print(f"[✅] Exported to ONNX: {onnx_path}")

# === STEP 2: Convert ONNX to TensorFlow SavedModel ===
def onnx_to_tensorflow(onnx_path, tf_model_dir):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_dir)
    print(f"[✅] Converted to TensorFlow SavedModel: {tf_model_dir}")

# === STEP 3: Convert TensorFlow to .tflite ===
def convert_to_tflite(tf_model_dir, tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: quantization
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"[✅] Saved TFLite model: {tflite_path}")

# === Main ===
if __name__ == "__main__":
    pth_path = "best_fish_model.pth"
    onnx_path = "fish_model.onnx"
    tf_model_dir = "fish_model_tf"
    tflite_path = "fish_model.tflite"

    export_to_onnx(pth_path, onnx_path)
    onnx_to_tensorflow(onnx_path, tf_model_dir)
    convert_to_tflite(tf_model_dir, tflite_path)
