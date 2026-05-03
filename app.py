import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

from runtime_config import configure_tensorflow_runtime

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "skin_model.keras"
CLASS_PATH = BASE_DIR / "model" / "class_names.json"

runtime_info = configure_tensorflow_runtime(tf)
print(f"Runtime: {runtime_info['accelerator']} ({', '.join(runtime_info['gpu_names']) or 'CPU'})")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/runtime")
def runtime():
    return jsonify(runtime_info)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img_array = preprocess_image(file)
        predictions = model.predict(img_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        if predicted_index >= len(class_names):
            return jsonify({"error": "Model output does not match class labels"}), 500

        return jsonify({
            "disease": class_names[predicted_index],
            "confidence": round(confidence, 2),
            "runtime": runtime_info["accelerator"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
