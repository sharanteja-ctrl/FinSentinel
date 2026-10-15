#!/bin/bash
# FinSentinel - One-Click Launcher for SLAM Review
# Checks dependencies, verifies models, and launches Streamlit.

set -e

echo "========================================================"
echo "  🛡️  LAUNCHING FINSENTINEL AI FRAUD DETECTION SYSTEM   "
echo "========================================================"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH."
    exit 1
fi

echo "[1/3] Verifying environment & trained models..."
if [ ! -f "models/best_model.pkl" ] || [ ! -f "models/scaler.pkl" ]; then
    echo "[INFO] Models not detected. Running training pipeline first..."
    python3 src/train.py
else
    echo "  ✓ Pre-trained models and scaler verified."
fi

echo "[2/3] Checking SQLite transaction database..."
python3 -c "
import sys; sys.path.insert(0, '.');
from database.db import init_db, seed_demo_transactions_if_empty;
init_db();
seed_demo_transactions_if_empty();
"
echo "  ✓ Database initialized and seeded."

echo "[3/3] Launching FinSentinel Streamlit Web Platform on http://localhost:8501 ..."
streamlit run app.py --server.port 8501
