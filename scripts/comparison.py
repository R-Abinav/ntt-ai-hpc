#!/usr/bin/env python3
# comparison.py
#
# three-way timing comparison: Serial vs Pure-HPC vs AI-HPC
#
# prerequisites:
#   1. build/bench_comparison binary exists  (add bench_comparison to CMakeLists.txt)
#   2. python/api/cost_model_xgb.pkl  and  python/api/scaler.pkl  exist
#   3.  pip install xgboost scikit-learn numpy matplotlib pandas
#
# run from the project root:
#   python scripts/comparison.py
#
# outputs:
#   results/comparison.csv
#   docs/plot7_three_way_comparison.png
#   docs/plot8_speedup_vs_serial.png

import math
import os
import pickle
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_BIN   = os.path.join(ROOT, "build", "bench_comparison")
XGB_PATH    = os.path.join(ROOT, "python", "api", "cost_model_xgb.pkl")
SCALER_PATH = os.path.join(ROOT, "python", "api", "scaler.pkl")
CSV_OUT     = os.path.join(ROOT, "results", "comparison.csv")
PLOT7_OUT   = os.path.join(ROOT, "docs", "plot7_three_way_comparison.png")
PLOT8_OUT   = os.path.join(ROOT, "docs", "plot8_speedup_vs_serial.png")

# ntt sizes to benchmark
NTT_SIZES = [64, 128, 256, 512, 1024, 2048, 4096,
             8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

CANDIDATE_THREADS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]

# max threads for pure-hpc mode — use logical core count, capped at 32
import multiprocessing
MAX_THREADS = min(multiprocessing.cpu_count(), 32)

# ── helpers ───────────────────────────────────────────────────────────────────

def check_prerequisites():
    missing = []
    if not os.path.isfile(BENCH_BIN):
        missing.append(f"binary not found: {BENCH_BIN}\n  → add bench_comparison to CMakeLists.txt and rebuild")
    if not os.path.isfile(XGB_PATH):
        missing.append(f"model not found: {XGB_PATH}\n  → run the jupyter notebook first")
    if not os.path.isfile(SCALER_PATH):
        missing.append(f"scaler not found: {SCALER_PATH}\n  → run the jupyter notebook first")
    if missing:
        print("ERROR: prerequisites not met:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

def load_models():
    with open(XGB_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return xgb_model, scaler

def ai_recommend(n: int, xgb_model, scaler) -> int:
    """query xgboost for the optimal thread count for size n."""
    log2_n = math.log2(n)
    rows = np.array([[log2_n, t] for t in CANDIDATE_THREADS], dtype=np.float32)
    rows_scaled = scaler.transform(rows)
    preds = xgb_model.predict(rows_scaled)
    best_idx = int(np.argmin(preds))
    return CANDIDATE_THREADS[best_idx]

def run_bench(n: int, threads: int) -> float:
    """run bench_comparison and return median time in microseconds."""
    result = subprocess.run(
        [BENCH_BIN, str(n), str(threads)],
        capture_output=True, text=True, check=True
    )
    # output format: n,threads,time_us
    parts = result.stdout.strip().split(",")
    return float(parts[2])

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    check_prerequisites()

    print(f"loading models from {os.path.dirname(XGB_PATH)}...")
    xgb_model, scaler = load_models()
    print(f"max threads for pure-hpc: {MAX_THREADS}")
    print()

    rows = []

    for n in NTT_SIZES:
        ai_threads = ai_recommend(n, xgb_model, scaler)

        print(f"n = {n:>8}  |  ai recommends {ai_threads:>2} threads  |  ", end="", flush=True)

        # serial: always 1 thread
        t_serial = run_bench(n, 1)

        # pure-hpc: max available threads
        t_hpc = run_bench(n, MAX_THREADS)

        # ai-hpc: xgboost-recommended threads
        # if ai already picked 1 or MAX_THREADS we still re-run to keep timing fair
        t_ai = run_bench(n, ai_threads)

        speedup_hpc = t_serial / t_hpc
        speedup_ai  = t_serial / t_ai

        print(f"serial={t_serial:>10.1f} us  |  hpc={t_hpc:>10.1f} us  |  ai-hpc={t_ai:>10.1f} us  |  "
              f"speedup(hpc)={speedup_hpc:.2f}x  speedup(ai)={speedup_ai:.2f}x")

        rows.append({
            "n":              n,
            "serial_us":      round(t_serial, 2),
            "hpc_threads":    MAX_THREADS,
            "hpc_us":         round(t_hpc, 2),
            "ai_threads":     ai_threads,
            "ai_hpc_us":      round(t_ai, 2),
            "speedup_hpc":    round(speedup_hpc, 4),
            "speedup_ai_hpc": round(speedup_ai, 4),
        })

    df = pd.DataFrame(rows)

    # ── save csv ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"\ncsv saved → {CSV_OUT}")

    # ── plot 7: grouped bar chart — absolute times ────────────────────────────
    labels = [f"$2^{{{int(math.log2(n))}}}$" for n in NTT_SIZES]
    x = np.arange(len(NTT_SIZES))
    width = 0.28

    fig, ax = plt.subplots(figsize=(16, 6))
    b1 = ax.bar(x - width, df["serial_us"],  width, label="Serial (1 thread)",          color="#4C72B0")
    b2 = ax.bar(x,         df["hpc_us"],     width, label=f"Pure HPC ({MAX_THREADS}T)",  color="#DD8452")
    b3 = ax.bar(x + width, df["ai_hpc_us"],  width, label="AI-HPC (XGBoost-selected T)", color="#55A868")

    ax.set_yscale("log")
    ax.set_xlabel("NTT Size n", fontsize=12)
    ax.set_ylabel("Time (µs, log scale)", fontsize=12)
    ax.set_title("Serial vs Pure-HPC vs AI-HPC — Wall-Clock Time per poly_mul_ntt", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # annotate ai thread count above each ai bar
    for i, row in df.iterrows():
        ax.text(x[i] + width, row["ai_hpc_us"] * 1.08,
                f"{int(row['ai_threads'])}T", ha="center", va="bottom", fontsize=7, color="#1a6e35")

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT7_OUT), exist_ok=True)
    plt.savefig(PLOT7_OUT, dpi=150)
    plt.close()
    print(f"plot7 saved → {PLOT7_OUT}")

    # ── plot 8: speedup vs serial ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(NTT_SIZES, df["speedup_hpc"],    "o--", color="#DD8452",
            label=f"Pure HPC ({MAX_THREADS}T) speedup", linewidth=1.8, markersize=6)
    ax.plot(NTT_SIZES, df["speedup_ai_hpc"], "s-",  color="#55A868",
            label="AI-HPC speedup", linewidth=2.2, markersize=7)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.2, label="Serial baseline (1×)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(NTT_SIZES)
    ax.set_xticklabels([f"$2^{{{int(math.log2(n))}}}$" for n in NTT_SIZES], fontsize=8)
    ax.set_xlabel("NTT Size n (log₂ scale)", fontsize=12)
    ax.set_ylabel("Speedup vs Serial", fontsize=12)
    ax.set_title("Speedup: Pure-HPC vs AI-HPC (relative to Serial baseline)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # shade region where pure-hpc is slower than serial
    ax.fill_between(NTT_SIZES, 1.0, df["speedup_hpc"],
                    where=(df["speedup_hpc"] < 1.0),
                    alpha=0.15, color="red", label="Pure-HPC overhead zone")

    plt.tight_layout()
    plt.savefig(PLOT8_OUT, dpi=150)
    plt.close()
    print(f"plot8 saved → {PLOT8_OUT}")

    # ── print summary table ───────────────────────────────────────────────────
    print()
    print(f"{'n':>10}  {'serial_us':>12}  {'hpc_us':>12}  {'ai_hpc_us':>12}  {'ai_T':>6}  {'spdup_hpc':>10}  {'spdup_ai':>10}")
    print("-" * 82)
    for _, row in df.iterrows():
        print(f"{int(row['n']):>10}  {row['serial_us']:>12.1f}  {row['hpc_us']:>12.1f}  "
              f"{row['ai_hpc_us']:>12.1f}  {int(row['ai_threads']):>6}  "
              f"{row['speedup_hpc']:>10.3f}x  {row['speedup_ai_hpc']:>10.3f}x")

if __name__ == "__main__":
    main()