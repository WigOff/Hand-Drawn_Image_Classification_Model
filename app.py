"""
app.py — Hand-Drawing Classification Server
============================================
Loads the trained Quick Draw CNN and exposes:
  GET  /           → drawing canvas UI
  POST /predict    → returns top-3 class predictions as JSON
"""

import os, json, base64, io
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import tensorflow as tf

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("model", "quickdraw_model.keras")
CLASSES_PATH= os.path.join("model", "class_names.json")
IMG_SIZE    = 28

app = Flask(__name__)

# ── load model once at startup ─────────────────────────────────────────────
print("Loading model …", flush=True)
model       = tf.keras.models.load_model(MODEL_PATH)
class_names : list[str] = json.load(open(CLASSES_PATH))
print(f"Model ready.  Classes: {class_names}", flush=True)


# ── helpers ────────────────────────────────────────────────────────────────

def preprocess_canvas(data_url: str) -> np.ndarray:
    """
    Convert a base-64 PNG data-URL from the canvas into a
    (1, 28, 28, 1) float32 array ready for the model.
    Quick Draw images are WHITE strokes on BLACK background.
    """
    # strip the data-url header
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # white canvas → make background black, strokes white (invert)
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    combined   = Image.alpha_composite(background, img).convert("L")
    inverted   = ImageOps.invert(combined)

    # resize to 28×28 using high-quality downsampling
    resized = inverted.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    arr = np.array(resized, dtype="float32") / 255.0
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# ── routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", classes=class_names)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "No image data received"}), 400

    try:
        arr  = preprocess_canvas(payload["image"])
        preds = model.predict(arr, verbose=0)[0]           # (num_classes,)

        # top-3
        top3_idx   = np.argsort(preds)[::-1][:3]
        results    = [
            {
                "label":      class_names[i],
                "confidence": round(float(preds[i]) * 100, 1)
            }
            for i in top3_idx
        ]
        return jsonify({"predictions": results})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/classes")
def get_classes():
    return jsonify({"classes": class_names})


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
