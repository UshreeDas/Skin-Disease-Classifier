import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "model/skin_model.keras"
CLASS_PATH = "model/class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
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

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img_array = preprocess_image(file)
        predictions = model.predict(img_array)

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        return jsonify({
            "disease": class_names[predicted_index],
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)