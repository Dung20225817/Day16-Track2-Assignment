#!/bin/bash
# =============================================================================
# startup.sh — GCP VM startup script (CPU mode: Python + LightGBM)
# Runs automatically when the VM first boots via terraform metadata_startup_script
# =============================================================================
set -e
LOG_FILE="/var/log/startup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo " LAB 16 — CPU Mode Startup: $(date)"
echo "=============================================="

# ─── 1. System update & Python ────────────────────────────────────────────────
echo "[1/4] Installing system packages..."
apt-get update -q
apt-get install -y -q python3 python3-pip python3-venv curl wget unzip

# ─── 2. Install ML libraries ──────────────────────────────────────────────────
echo "[2/4] Installing Python ML packages (lightgbm, scikit-learn, etc.)..."
pip3 install --quiet --upgrade pip
pip3 install --quiet \
  lightgbm \
  scikit-learn \
  pandas \
  numpy \
  kaggle \
  flask

# ─── 3. Copy benchmark script ─────────────────────────────────────────────────
echo "[3/4] Setting up benchmark script..."
mkdir -p /opt/ml-benchmark

cat > /opt/ml-benchmark/benchmark.py << 'PYEOF'
#!/usr/bin/env python3
"""
LightGBM Benchmark — Credit Card Fraud Detection
LAB 16 fallback: CPU instance thay GPU
"""
import time, json, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

RESULT_FILE = "/opt/ml-benchmark/benchmark_result.json"
DATA_FILE   = "/opt/ml-benchmark/creditcard.csv"

def download_sample_data():
    """Tạo dataset giả nếu chưa có Kaggle credentials."""
    print("  Generating synthetic credit card fraud dataset (50,000 rows)...")
    np.random.seed(42)
    n = 50_000
    X = np.random.randn(n, 28)
    # ~0.17% fraud rate (giống dataset thật)
    y = (np.random.rand(n) < 0.0017).astype(int)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = np.abs(np.random.randn(n)) * 100
    df["Class"]  = y
    df.to_csv(DATA_FILE, index=False)
    print(f"  Generated {n} rows, {y.sum()} fraud cases ({y.mean()*100:.2f}%)")

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n[Step 1] Loading dataset...")
t0 = time.time()

if not os.path.exists(DATA_FILE):
    # Thử tải từ Kaggle trước, nếu không có credentials thì dùng synthetic data
    try:
        import subprocess
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud",
             "--unzip", "-p", "/opt/ml-benchmark/"],
            check=True, timeout=120
        )
    except Exception:
        download_sample_data()

df = pd.read_csv(DATA_FILE)
load_time = time.time() - t0
print(f"  Loaded {len(df):,} rows in {load_time:.2f}s")
print(f"  Fraud rate: {df['Class'].mean()*100:.4f}%")

X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── Training ───────────────────────────────────────────────────────────────────
print("\n[Step 2] Training LightGBM model...")
t1 = time.time()

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_test,  label=y_test, reference=dtrain)

params = {
    "objective":      "binary",
    "metric":         "auc",
    "boosting_type":  "gbdt",
    "num_leaves":     63,
    "learning_rate":  0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq":   5,
    "verbose":        -1,
    "n_jobs":         -1,    # dùng hết CPU cores
}

callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)]
model = lgb.train(
    params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dvalid],
    callbacks=callbacks,
)

train_time = time.time() - t1
print(f"  Training time: {train_time:.2f}s | Best iteration: {model.best_iteration}")

# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\n[Step 3] Evaluating model...")
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

auc       = roc_auc_score(y_test, y_prob)
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred)

# ── Inference benchmark ────────────────────────────────────────────────────────
print("\n[Step 4] Benchmarking inference latency...")

# Single row
t2 = time.time()
for _ in range(100):
    model.predict(X_test.iloc[[0]])
lat_1row = (time.time() - t2) / 100 * 1000  # ms

# 1000 rows
t3 = time.time()
model.predict(X_test.iloc[:1000])
lat_1000rows = (time.time() - t3) * 1000  # ms

# ── Print results ──────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  BENCHMARK RESULTS")
print("="*50)
print(f"  Data load time       : {load_time:.3f}s")
print(f"  Training time        : {train_time:.3f}s")
print(f"  Best iteration       : {model.best_iteration}")
print(f"  AUC-ROC              : {auc:.6f}")
print(f"  Accuracy             : {accuracy:.6f}")
print(f"  F1-Score             : {f1:.6f}")
print(f"  Precision            : {precision:.6f}")
print(f"  Recall               : {recall:.6f}")
print(f"  Inference (1 row)    : {lat_1row:.3f} ms")
print(f"  Inference (1000 rows): {lat_1000rows:.3f} ms")
print("="*50)

# ── Save JSON result ───────────────────────────────────────────────────────────
result = {
    "instance_type"       : "n2-standard-8 (CPU)",
    "dataset_rows"        : len(df),
    "data_load_time_sec"  : round(load_time, 3),
    "training_time_sec"   : round(train_time, 3),
    "best_iteration"      : model.best_iteration,
    "auc_roc"             : round(auc, 6),
    "accuracy"            : round(accuracy, 6),
    "f1_score"            : round(f1, 6),
    "precision"           : round(precision, 6),
    "recall"              : round(recall, 6),
    "inference_1row_ms"   : round(lat_1row, 3),
    "inference_1000rows_ms": round(lat_1000rows, 3),
    "timestamp"           : time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(RESULT_FILE, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n  Results saved to: {RESULT_FILE}")
print("  DONE!")
PYEOF

chmod +x /opt/ml-benchmark/benchmark.py

# ─── 4. Start simple HTTP server on port 8000 (Load Balancer health check) ───
echo "[4/4] Starting HTTP health server on port 8000..."
cat > /opt/ml-benchmark/health_server.py << 'PYEOF'
from flask import Flask, jsonify
import subprocess, os, json

app = Flask(__name__)
RESULT_FILE = "/opt/ml-benchmark/benchmark_result.json"

@app.route("/health")
def health():
    return jsonify({"status": "ok", "mode": "cpu-lightgbm"}), 200

@app.route("/v1/benchmark/result")
def result():
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE) as f:
            return jsonify(json.load(f)), 200
    return jsonify({"status": "not_run_yet", "hint": "SSH in and run: python3 /opt/ml-benchmark/benchmark.py"}), 200

@app.route("/v1/benchmark/run")
def run():
    subprocess.Popen(["python3", "/opt/ml-benchmark/benchmark.py"])
    return jsonify({"status": "started", "message": "Benchmark running in background. Check /v1/benchmark/result in ~60s"}), 202

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
PYEOF

# Chạy server dưới background
nohup python3 /opt/ml-benchmark/health_server.py >> /var/log/health_server.log 2>&1 &

echo "=============================================="
echo " Startup complete: $(date)"
echo " Health server running on port 8000"
echo " SSH in and run: python3 /opt/ml-benchmark/benchmark.py"
echo "=============================================="
