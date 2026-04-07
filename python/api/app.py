import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request

# ── model architecture (must match training exactly) ──────────────────────────


class CostModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# ── load model and scaler once at startup ─────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

model = CostModel()
model.load_state_dict(
    torch.load(
        os.path.join(_HERE, "cost_model.pt"), map_location="cpu", weights_only=True
    )
)
model.eval()

with open(os.path.join(_HERE, "scaler.pkl"), "rb") as _f:
    scaler = pickle.load(_f)

CANDIDATE_THREADS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]

ALL_N = [
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]

app = Flask(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _score_all_threads(n: int) -> list[dict]:
    """Score every candidate thread count for a given NTT size n.

    Returns a list of dicts sorted by CANDIDATE_THREADS order, each with
    'threads' and 'predicted_time_us' (inverse log1p of raw model output).
    """
    log2_n = math.log2(n)
    rows = np.array([[log2_n, t] for t in CANDIDATE_THREADS], dtype=np.float32)
    rows_scaled = scaler.transform(rows)

    with torch.no_grad():
        raw = model(torch.tensor(rows_scaled)).squeeze().numpy()

    return [
        {"threads": t, "predicted_time_us": round(float(np.expm1(p)), 2)}
        for t, p in zip(CANDIDATE_THREADS, raw)
    ]


# ── endpoints ─────────────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "ntt-cost-model"})


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

    all_preds = _score_all_threads(n)
    best = min(all_preds, key=lambda x: x["predicted_time_us"])

    return jsonify(
        {
            "n": n,
            "recommended_threads": best["threads"],
            "predicted_time_us": best["predicted_time_us"],
            "all_predictions": all_preds,
        }
    )


@app.route("/optimal_map", methods=["GET"])
def optimal_map():
    result = []
    for n in ALL_N:
        all_preds = _score_all_threads(n)
        best = min(all_preds, key=lambda x: x["predicted_time_us"])
        result.append({"n": n, "recommended_threads": best["threads"]})
    return jsonify(result)


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5001)
