import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PdfImage, Table, TableStyle
import tensorflow as tf

from runtime_config import configure_tensorflow_runtime

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "skin_model.tflite"
CLASS_PATH = BASE_DIR / "model" / "class_names.json"

runtime_info = configure_tensorflow_runtime(tf)
print(f"Runtime: {runtime_info['accelerator']} ({', '.join(runtime_info['gpu_names']) or 'CPU'})")

# Serve with the TFLite interpreter rather than the full Keras model. TFLite
# uses pre-allocated fixed-size buffers and skips Keras/tf.function graph
# tracing, which is both much faster and much lighter on CPU-only,
# memory-constrained hosts (see convert_to_tflite.py for the conversion step).
# num_threads=1 matches hosts with very small CPU allocations (e.g. free-tier
# containers), where a larger thread pool would just add scheduling overhead.
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=1)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)


def run_inference(img_array):
    interpreter.set_tensor(input_details["index"], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details["index"])


# Warm up the interpreter with one dummy prediction at boot, so any one-time
# allocation cost happens at boot (where a slow start is tolerated) instead
# of on a user's first real request.
try:
    run_inference(np.zeros((1, 224, 224, 3), dtype=np.float32))
    print("Model warm-up prediction complete.")
except Exception as warm_up_error:
    print(f"Model warm-up failed: {warm_up_error}")

EDUCATIONAL_INFORMATION = {
    "Acne": {
        "reasons": "Acne can be associated with blocked pores, excess skin oil, bacteria, hormonal changes, and some cosmetics or medicines.",
        "prevention": "Use gentle non-comedogenic skin products, avoid picking lesions, cleanse gently, and keep hair and frequently touched items clean.",
        "care": "A dermatologist can help assess persistent, painful, or scarring acne and recommend suitable care.",
    },
    "Eczema": {
        "reasons": "Eczema may be linked to skin-barrier sensitivity, irritation, allergies, dry skin, environmental triggers, or family history.",
        "prevention": "Use fragrance-free moisturizers, identify possible irritants, avoid very hot water, and protect skin from harsh soaps.",
        "care": "Regular moisturising and medical assessment can be helpful, especially for severe itching, infection signs, or persistent flare-ups.",
    },
    "Melanoma": {
        "reasons": "Melanoma is a serious skin cancer associated with abnormal pigment cells. Risk factors can include UV exposure, sunburns, many moles, and family history.",
        "prevention": "Protect skin from UV exposure with shade, protective clothing, and sunscreen. Monitor changing or unusual pigmented spots.",
        "care": "Seek prompt in-person assessment from a qualified dermatologist for any suspicious, changing, bleeding, or new skin lesion.",
    },
    "Psoriasis": {
        "reasons": "Psoriasis is an immune-mediated condition that can be influenced by genetics, infections, stress, skin injury, smoking, or certain medicines.",
        "prevention": "Avoid known triggers where possible, moisturize regularly, avoid skin trauma, and maintain general wellbeing.",
        "care": "A dermatologist can confirm the condition and discuss appropriate management for plaques, discomfort, or extensive disease.",
    },
    "Vitiligo": {
        "reasons": "Vitiligo involves loss of skin pigment. It may be associated with autoimmune processes, family history, or skin trauma in some people.",
        "prevention": "Use sun protection to reduce contrast and protect depigmented skin. Avoid unproven treatments that may irritate the skin.",
        "care": "A dermatologist can confirm the cause of pigment changes and discuss safe support and treatment options where appropriate.",
    },
}

DEFAULT_INFORMATION = {
    "reasons": "Skin changes can have many possible causes, including irritation, infection, inflammation, environmental exposure, or other medical conditions.",
    "prevention": "Use gentle skin care, avoid known irritants, and protect the affected area from excessive sun exposure or friction.",
    "care": "For a reliable diagnosis and personalised care, consult a qualified dermatologist or healthcare professional.",
}

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index():
    return render_template("index.html", active_page="home")

@app.get("/predict")
def predict_page():
    return render_template("predict.html", active_page="predict")

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
        predictions = run_inference(img_array)

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        if predicted_index >= len(class_names):
            return jsonify({"error": "Model output does not match class labels"}), 500

        return jsonify({
            "disease": class_names[predicted_index],
            "confidence": round(confidence, 2),
            "education": EDUCATIONAL_INFORMATION.get(class_names[predicted_index], DEFAULT_INFORMATION),
            "runtime": runtime_info["accelerator"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/generate-report")
def generate_report():
    patient_name = request.form.get("patient_name", "").strip()
    patient_age = request.form.get("patient_age", "").strip()
    disease = request.form.get("disease", "").strip()
    confidence = request.form.get("confidence", "").strip()
    image_file = request.files.get("image")

    if not patient_name or len(patient_name) > 120:
        return jsonify({"error": "Enter a valid patient name."}), 400
    try:
        age = int(patient_age)
        if not 0 <= age <= 130:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Enter a valid age between 0 and 130."}), 400
    if not disease or not confidence or not image_file:
        return jsonify({"error": "Prediction details and the analyzed image are required."}), 400

    try:
        image = Image.open(image_file).convert("RGB")
        image.thumbnail((900, 700))
        image_buffer = BytesIO()
        image.save(image_buffer, format="JPEG", quality=90)
        image_buffer.seek(0)

        report_buffer = BytesIO()
        doc = SimpleDocTemplate(report_buffer, pagesize=A4, rightMargin=44, leftMargin=44, topMargin=44, bottomMargin=44)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#123b5d"), fontSize=21, leading=26, spaceAfter=8))
        styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#0d766e"), fontSize=13, leading=18, spaceBefore=14, spaceAfter=6))
        styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=9.5, leading=14, textColor=colors.HexColor("#334155")))
        story = [
            Paragraph("DermaVision - Skin Disease Classification Report", styles["ReportTitle"]),
            Paragraph("AI-powered academic and research report", styles["Body"]), Spacer(1, 12),
        ]
        generated_at = datetime.now().strftime("%d %B %Y, %I:%M %p")
        patient_table = Table([["Patient Name", patient_name], ["Patient Age", str(age)], ["Report Generated", generated_at]], colWidths=[1.55 * inch, 4.55 * inch])
        patient_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e7f3f2")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#123b5d")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]))
        image_width = 4.5 * inch
        image_height = min(3.4 * inch, image_width * image.height / image.width)
        image_width = min(image_width, image_height * image.width / image.height)
        story += [Paragraph("Patient Information", styles["Section"]), patient_table, Paragraph("Analyzed Image", styles["Section"]), PdfImage(image_buffer, width=image_width, height=image_height)]
        result_table = Table([["Predicted Disease", disease], ["Prediction Confidence", f"{confidence}%"]], colWidths=[1.8 * inch, 4.3 * inch])
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e7f3f2")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#123b5d")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]))
        info = EDUCATIONAL_INFORMATION.get(disease, DEFAULT_INFORMATION)
        story += [Paragraph("AI Analysis Result", styles["Section"]), result_table]
        story += [Paragraph("Educational Information", styles["Section"])]
        for heading, content in [("Probable Medical Reasons", info["reasons"]), ("Prevention", info["prevention"]), ("General Care / Possible Solutions", info["care"])]:
            story += [Paragraph(f"<b>{heading}</b>", styles["Body"]), Paragraph(content, styles["Body"]), Spacer(1, 5)]
        story += [Paragraph("Project Information", styles["Section"]), Paragraph("Developed by Team DermaVision<br/>Team Members: Ushree Das, Nilajana Pal<br/>Mentor: Prof. Dr. Sudip Dogra", styles["Body"])]
        story += [Paragraph("Medical Disclaimer", styles["Section"]), Paragraph("This report is generated by an AI-based academic/research system and is intended for educational purposes only. The prediction does not constitute a confirmed medical diagnosis or prescription. Please consult a qualified dermatologist or healthcare professional for diagnosis and treatment.", styles["Body"])]
        doc.build(story)
        report_buffer.seek(0)
        return send_file(report_buffer, as_attachment=True, download_name="dermavision-report.pdf", mimetype="application/pdf")
    except Exception as e:
        return jsonify({"error": f"Unable to generate report: {str(e)}"}), 500

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
