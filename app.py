"""
app.py — Flask Backend
Road Accident Severity Prediction | IML AAT Miniproject | BMSCE
Run: python app.py  (after running train_model.py once)
"""

from flask import Flask, render_template, request, jsonify
import joblib, json, os, numpy as np

BASE    = os.path.dirname(os.path.abspath(__file__))
MDL_DIR = os.path.join(BASE, "model")

app = Flask(__name__)

# ── Load saved artefacts ───────────────────────────────────────────────────────
rf     = joblib.load(os.path.join(MDL_DIR, "rf_model.pkl"))
svm    = joblib.load(os.path.join(MDL_DIR, "svm_model.pkl"))
nn     = joblib.load(os.path.join(MDL_DIR, "nn_model.pkl"))
scaler = joblib.load(os.path.join(MDL_DIR, "scaler.pkl"))

with open(os.path.join(MDL_DIR, "metrics.json")) as f:
    metrics = json.load(f)

with open(os.path.join(MDL_DIR, "classification_reports.json")) as f:
    clf_reports = json.load(f)

# ── Constants ──────────────────────────────────────────────────────────────────
SEVERITY_LABEL = {0: "Slight", 1: "Serious", 2: "Fatal"}
SEVERITY_COLOR = {0: "#16a34a", 1: "#d97706", 2: "#dc2626"}

FEATURE_ORDER = [
    "Hour_of_Day", "Speed_Limit_kmh", "Weather_Condition", "Road_Type",
    "Light_Condition", "Number_of_Vehicles", "Alcohol_Involved",
    "Seatbelt_Worn", "Road_Surface"
]

ALGO_INFO = {
    "rf": {
        "name": "Random Forest",
        "type": "Ensemble Bagging (200 CART trees)",
        "key":  "Gini impurity split | max_depth=12 | class_weight=balanced",
        "note": "Does NOT require feature scaling"
    },
    "svm": {
        "name": "SVM (RBF Kernel)",
        "type": "Kernel-based SVM | One-vs-One multi-class",
        "key":  "C=10 | gamma=scale (auto) | Platt scaling for probabilities",
        "note": "Requires StandardScaler (applied internally)"
    },
    "nn": {
        "name": "Neural Network (MLP)",
        "type": "Feedforward MLP: Input→128→64→32→Output",
        "key":  "ReLU activations | Adam optimiser | L2 α=0.001 | Early stopping",
        "note": "Requires StandardScaler (applied internally)"
    }
}


def parse_input(data):
    return np.array([[
        int(data["hour"]),
        int(data["speed"]),
        int(data["weather"]),
        int(data["road_type"]),
        int(data["light"]),
        int(data["vehicles"]),
        int(data["alcohol"]),
        int(data["seatbelt"]),
        int(data["road_surface"]),
    ]])


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics, algo_info=ALGO_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data    = request.get_json()
        X_raw   = parse_input(data)
        X_sc    = scaler.transform(X_raw)
        model_k = data.get("model", "rf")

        if model_k == "rf":
            pred  = int(rf.predict(X_raw)[0])
            proba = rf.predict_proba(X_raw)[0].tolist()
        elif model_k == "svm":
            pred  = int(svm.predict(X_sc)[0])
            proba = svm.predict_proba(X_sc)[0].tolist()
        else:
            pred  = int(nn.predict(X_sc)[0])
            proba = nn.predict_proba(X_sc)[0].tolist()

        return jsonify({
            "severity":       SEVERITY_LABEL[pred],
            "severity_index": pred,
            "color":          SEVERITY_COLOR[pred],
            "algo":           ALGO_INFO[model_k],
            "probabilities": {
                "Slight":   round(proba[0] * 100, 1),
                "Serious":  round(proba[1] * 100, 1),
                "Fatal":    round(proba[2] * 100, 1),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/metrics")
def get_metrics():
    return jsonify({"metrics": metrics, "reports": clf_reports})


if __name__ == "__main__":
    print("🚗  Road Accident Severity Predictor")
    print("     http://127.0.0.1:5000")
    app.run(debug=True)
