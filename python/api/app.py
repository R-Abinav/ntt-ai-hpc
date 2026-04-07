import math
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request

# ── model architecture ────────────────────────────────────────────────────────
class CostModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# ── load both models and scaler once at startup ───────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

mlp_model = CostModel()
mlp_model.load_state_dict(
    torch.load(os.path.join(_HERE, "cost_model.pt"), map_location="cpu", weights_only=True)
)
mlp_model.eval()

with open(os.path.join(_HERE, "cost_model_xgb.pkl"), "rb") as _f:
    xgb_model = pickle.load(_f)

with open(os.path.join(_HERE, "scaler.pkl"), "rb") as _f:
    scaler = pickle.load(_f)

ACTIVE_MODEL = "xgboost" #or it can "mlp"

CANDIDATE_THREADS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
ALL_N = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
         32768, 65536, 131072, 262144, 524288, 1048576]

app = Flask(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────
def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def _score_all_threads(n: int, model_name: str) -> list[dict]:
    log2_n = math.log2(n)
    rows = np.array([[log2_n, t] for t in CANDIDATE_THREADS], dtype=np.float32)
    rows_scaled = scaler.transform(rows)

    if model_name == "mlp":
        with torch.no_grad():
            raw = mlp_model(torch.tensor(rows_scaled)).squeeze().numpy()
    elif model_name == "xgboost":
        raw = xgb_model.predict(rows_scaled)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return [
        {"threads": t, "predicted_time_us": round(float(np.expm1(p)), 2)}
        for t, p in zip(CANDIDATE_THREADS, raw)
    ]

# ── endpoints ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "ntt-cost-model", "active_model": ACTIVE_MODEL})

@app.route("/active_model", methods=["POST"])
def set_active_model():
    global ACTIVE_MODEL
    data = request.get_json(silent=True)
    if not data or "model" not in data:
        return jsonify({"error": "request body must contain 'model'"}), 400
    if data["model"] not in ("mlp", "xgboost"):
        return jsonify({"error": "model must be 'mlp' or 'xgboost'"}), 400
    ACTIVE_MODEL = data["model"]
    return jsonify({"active_model": ACTIVE_MODEL})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "n" not in data:
        return jsonify({"error": "request body must contain 'n'"}), 400
    n = data["n"]
    if not isinstance(n, int) or isinstance(n, bool):
        return jsonify({"error": "n must be an integer"}), 400
    if n <= 0:
        return jsonify({"error": "n must be a positive integer"}), 400
    if not _is_power_of_2(n):
        return jsonify({"error": "n must be a power of 2"}), 400
    if n < 64 or n > 1048576:
        return jsonify({"error": "n must be between 64 and 1048576"}), 400

    model_name = request.args.get("model", ACTIVE_MODEL)
    if model_name not in ("mlp", "xgboost"):
        return jsonify({"error": "model must be 'mlp' or 'xgboost'"}), 400

    all_preds = _score_all_threads(n, model_name)
    best = min(all_preds, key=lambda x: x["predicted_time_us"])
    return jsonify({
        "n": n,
        "model_used": model_name,
        "recommended_threads": best["threads"],
        "predicted_time_us": best["predicted_time_us"],
        "all_predictions": all_preds,
    })

@app.route("/optimal_map", methods=["GET"])
def optimal_map():
    model_name = request.args.get("model", ACTIVE_MODEL)
    if model_name not in ("mlp", "xgboost"):
        return jsonify({"error": "model must be 'mlp' or 'xgboost'"}), 400
    result = []
    for n in ALL_N:
        all_preds = _score_all_threads(n, model_name)
        best = min(all_preds, key=lambda x: x["predicted_time_us"])
        result.append({"n": n, "recommended_threads": best["threads"]})
    return jsonify(result)

@app.route("/compare", methods=["GET"])
def compare():
    n = request.args.get("n", type=int)
    if n is None:
        return jsonify({"error": "query param 'n' is required"}), 400
    if not _is_power_of_2(n) or n < 64 or n > 1048576:
        return jsonify({"error": "n must be a power of 2 between 64 and 1048576"}), 400

    mlp_preds  = _score_all_threads(n, "mlp")
    xgb_preds  = _score_all_threads(n, "xgboost")
    mlp_best   = min(mlp_preds,  key=lambda x: x["predicted_time_us"])
    xgb_best   = min(xgb_preds,  key=lambda x: x["predicted_time_us"])

    return jsonify({
        "n": n,
        "mlp":     {"recommended_threads": mlp_best["threads"], "predicted_time_us": mlp_best["predicted_time_us"]},
        "xgboost": {"recommended_threads": xgb_best["threads"], "predicted_time_us": xgb_best["predicted_time_us"]},
        "agreement": mlp_best["threads"] == xgb_best["threads"],
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)