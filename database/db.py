"""
SQLite Database Manager for FinSentinel.
Provides persistent storage, historical transaction logging, analytics querying, and seeding.
"""

import os
import sqlite3
from datetime import datetime, timedelta
import random

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "transactions.db")


def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the transactions table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            amount REAL NOT NULL,
            prediction TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            risk_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            top_features TEXT,
            status TEXT NOT NULL,
            is_demo INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def insert_transaction(txn_data: dict) -> str:
    """Inserts a new analyzed transaction record into SQLite."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    if "txn_id" not in txn_data:
        txn_data["txn_id"] = f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(100, 999)}"

    if "timestamp" not in txn_data:
        txn_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO transactions (
            txn_id, timestamp, amount, prediction, fraud_probability,
            risk_score, risk_level, top_features, status, is_demo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        txn_data["txn_id"],
        txn_data["timestamp"],
        float(txn_data.get("amount", 0.0)),
        txn_data.get("prediction", "GENUINE"),
        float(txn_data.get("fraud_probability", 0.0)),
        int(txn_data.get("risk_score", 0)),
        txn_data.get("risk_level", "LOW RISK"),
        txn_data.get("top_features", ""),
        txn_data.get("status", "APPROVED" if txn_data.get("risk_score", 0) <= 30 else "BLOCKED"),
        int(txn_data.get("is_demo", 0))
    ))
    conn.commit()
    conn.close()
    return txn_data["txn_id"]


def get_all_transactions(filter_risk: str = None, filter_pred: str = None, limit: int = 200):
    """Retrieves transaction history with optional filters."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM transactions WHERE 1=1"
    params = []

    if filter_risk and filter_risk != "All":
        query += " AND risk_level = ?"
        params.append(filter_risk)

    if filter_pred and filter_pred != "All":
        query += " AND prediction = ?"
        params.append(filter_pred)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_transaction_stats():
    """Aggregates executive KPIs for the Streamlit dashboard."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM transactions")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as fraud FROM transactions WHERE prediction = 'FRAUD'")
    fraud = cursor.fetchone()["fraud"]

    cursor.execute("SELECT COUNT(*) as genuine FROM transactions WHERE prediction = 'GENUINE'")
    genuine = cursor.fetchone()["genuine"]

    cursor.execute("SELECT COUNT(*) as high_risk FROM transactions WHERE risk_level = 'HIGH RISK'")
    high_risk = cursor.fetchone()["high_risk"]

    cursor.execute("SELECT COUNT(*) as med_risk FROM transactions WHERE risk_level = 'MEDIUM RISK'")
    med_risk = cursor.fetchone()["med_risk"]

    cursor.execute("SELECT COUNT(*) as low_risk FROM transactions WHERE risk_level = 'LOW RISK'")
    low_risk = cursor.fetchone()["low_risk"]

    cursor.execute("SELECT AVG(amount) as avg_amt, MAX(amount) as max_amt FROM transactions")
    amt_row = cursor.fetchone()
    avg_amt = amt_row["avg_amt"] or 0.0
    max_amt = amt_row["max_amt"] or 0.0

    conn.close()

    fraud_rate = (fraud / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "fraud": fraud,
        "genuine": genuine,
        "high_risk": high_risk,
        "med_risk": med_risk,
        "low_risk": low_risk,
        "fraud_rate": round(fraud_rate, 2),
        "avg_amount": round(avg_amt, 2),
        "max_amount": round(max_amt, 2)
    }


def seed_demo_transactions_if_empty():
    """
    Seeds a realistic cohort of historical demo transactions if the database is brand new.
    Clearly tags records with is_demo=1 so demo records are transparently identified.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM transactions")
    count = cursor.fetchone()["count"]
    
    if count < 5:
        base_time = datetime.now() - timedelta(days=2)
        demo_txns = [
            # Genuine transactions
            {"txn_id": "TXN-DEMO-1001", "offset_min": 10, "amount": 42.50, "pred": "GENUINE", "prob": 2.1, "score": 6, "level": "LOW RISK", "feat": "Amount, V14 (Normal)", "status": "APPROVED"},
            {"txn_id": "TXN-DEMO-1002", "offset_min": 25, "amount": 118.99, "pred": "GENUINE", "prob": 3.8, "score": 11, "level": "LOW RISK", "feat": "Amount, V17 (Normal)", "status": "APPROVED"},
            {"txn_id": "TXN-DEMO-1003", "offset_min": 45, "amount": 15.00, "pred": "GENUINE", "prob": 1.2, "score": 3, "level": "LOW RISK", "feat": "V12, V10 (Normal)", "status": "APPROVED"},
            {"txn_id": "TXN-DEMO-1004", "offset_min": 70, "amount": 250.00, "pred": "GENUINE", "prob": 5.4, "score": 16, "level": "LOW RISK", "feat": "Amount, V4 (Normal)", "status": "APPROVED"},
            {"txn_id": "TXN-DEMO-1005", "offset_min": 95, "amount": 12.30, "pred": "GENUINE", "prob": 1.5, "score": 4, "level": "LOW RISK", "feat": "Amount, V14 (Normal)", "status": "APPROVED"},
            {"txn_id": "TXN-DEMO-1006", "offset_min": 120, "amount": 85.20, "pred": "GENUINE", "prob": 4.1, "score": 12, "level": "LOW RISK", "feat": "V17, V10 (Normal)", "status": "APPROVED"},
            {"txn_id": "TXN-DEMO-1007", "offset_min": 150, "amount": 34.00, "pred": "GENUINE", "prob": 2.0, "score": 6, "level": "LOW RISK", "feat": "V14 (Normal)", "status": "APPROVED"},
            
            # Suspicious / Medium Risk
            {"txn_id": "TXN-DEMO-1008", "offset_min": 180, "amount": 780.00, "pred": "GENUINE", "prob": 42.0, "score": 56, "level": "MEDIUM RISK", "feat": "V14 (-1.8), Amount", "status": "FLAGGED_REVIEW"},
            {"txn_id": "TXN-DEMO-1009", "offset_min": 210, "amount": 950.00, "pred": "GENUINE", "prob": 48.5, "score": 63, "level": "MEDIUM RISK", "feat": "V17 (-2.1), V10 (-1.9)", "status": "OTP_SENT"},
            {"txn_id": "TXN-DEMO-1010", "offset_min": 240, "amount": 540.00, "pred": "GENUINE", "prob": 35.0, "score": 48, "level": "MEDIUM RISK", "feat": "Amount, V4 (+1.6)", "status": "FLAGGED_REVIEW"},
            
            # High Risk Fraudulent
            {"txn_id": "TXN-DEMO-1011", "offset_min": 280, "amount": 1890.00, "pred": "FRAUD", "prob": 96.4, "score": 97, "level": "HIGH RISK", "feat": "V14 (-5.2), V17 (-4.8), V12 (-3.9)", "status": "BLOCKED"},
            {"txn_id": "TXN-DEMO-1012", "offset_min": 320, "amount": 1.00, "pred": "FRAUD", "prob": 92.1, "score": 94, "level": "HIGH RISK", "feat": "V14 (-4.5), V10 (-3.8) [Card Testing]", "status": "BLOCKED"},
            {"txn_id": "TXN-DEMO-1013", "offset_min": 360, "amount": 2340.50, "pred": "FRAUD", "prob": 98.2, "score": 99, "level": "HIGH RISK", "feat": "V14 (-6.1), V17 (-5.5), Amount", "status": "BLOCKED"}
        ]

        for item in demo_txns:
            ts = (base_time + timedelta(minutes=item["offset_min"])).strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO transactions (
                    txn_id, timestamp, amount, prediction, fraud_probability,
                    risk_score, risk_level, top_features, status, is_demo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                item["txn_id"], ts, item["amount"], item["pred"],
                item["prob"], item["score"], item["level"], item["feat"], item["status"]
            ))
            
        conn.commit()
    conn.close()


def clear_history():
    """Clears transaction records."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    seed_demo_transactions_if_empty()
    stats = get_transaction_stats()
    print("[DB] Initialized SQLite database successfully.")
    print(f"[DB] Stats: {stats}")
