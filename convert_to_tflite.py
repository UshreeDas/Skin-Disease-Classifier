"""One-off conversion: model/skin_model.keras -> model/skin_model.tflite

TFLite's interpreter skips Keras/tf.function graph-tracing overhead and uses
pre-allocated fixed-size buffers, which is both faster and lighter on CPU-only,
memory-constrained hosts than calling model.predict() directly. Dynamic-range
quantization (the default weight optimization below) also shrinks the file
size roughly 4x, which reduces disk and load-time memory further.

Run locally (not part of the deployed app): python convert_to_tflite.py
"""
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KERAS_MODEL_PATH = BASE_DIR / "model" / "skin_model.keras"
TFLITE_MODEL_PATH = BASE_DIR / "model" / "skin_model.tflite"

print(f"Loading Keras model from {KERAS_MODEL_PATH} ...")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantization
tflite_model = converter.convert()

TFLITE_MODEL_PATH.write_bytes(tflite_model)

keras_size_mb = KERAS_MODEL_PATH.stat().st_size / (1024 * 1024)
tflite_size_mb = TFLITE_MODEL_PATH.stat().st_size / (1024 * 1024)
print(f"Keras model:  {keras_size_mb:.1f} MB")
print(f"TFLite model: {tflite_size_mb:.1f} MB")
print(f"Saved to {TFLITE_MODEL_PATH}")
