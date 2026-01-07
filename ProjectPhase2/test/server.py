"""
Flask API to serve predictions and collect samples. MERN frontend can call this API.
Run: python3 server.py
"""
from flask import Flask, request, jsonify
import os
import csv
from trainer_and_model import predict, train_knn, load_model

SAMPLES_CSV = "samples_labeled.csv"
MODEL_PATH = "model.joblib"

app = Flask(__name__)

# ensure CSV exists with header
if not os.path.exists(SAMPLES_CSV):
    with open(SAMPLES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","F1","F2","F3","F4","F5","AX","AY","AZ","GX","GY","GZ","label"])

@app.route("/predict", methods=["POST"])
def api_predict():
    # Accept JSON payload containing the sensor fields.
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400
    try:
        res = predict(data)
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/add_sample", methods=["POST"])
def add_sample():
    """
    Accepts JSON:
    { F1:..., F2:..., ... , GZ:..., label: "HELLO" }
    """
    data = request.get_json()
    if not data or "label" not in data:
        return jsonify({"error": "Missing data or label"}), 400
    # assemble row
    import datetime
    feat_cols = ["F1","F2","F3","F4","F5","AX","AY","AZ","GX","GY","GZ"]
    try:
        row = [datetime.datetime.utcnow().isoformat()]
        for c in feat_cols:
            row.append(data.get(c, ""))
        row.append(str(data["label"]))
        with open(SAMPLES_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return jsonify({"ok": True, "message": "sample added"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train_route():
    try:
        model = train_knn(SAMPLES_CSV, k=3)
        return jsonify({"ok": True, "message": "trained", "model_type": model.get("type")})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    m = load_model()
    return jsonify({"has_model": m is not None, "model_type": m.get("type") if m else None})

if __name__ == "__main__":
    # Listen on all interfaces so your MERN app on same network can call it
    app.run(host="0.0.0.0", port=5000, debug=True)
