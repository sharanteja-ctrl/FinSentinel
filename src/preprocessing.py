"""
Preprocessing and Data Ingestion Pipeline for FinSentinel.
Strictly eliminates Data Leakage by fitting scalers and SMOTE only on the training split.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


def load_and_validate_data(csv_path: str, max_samples: int = None) -> pd.DataFrame:
    """
    Loads creditcard.csv and validates feature completeness and data types.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run download_data.py first.")
        
    df = pd.read_csv(csv_path)
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        
    # Check target column
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in dataset!")

    # Optionally sample if requested for ultra-fast training, keeping all fraud cases
    if max_samples and len(df) > max_samples:
        fraud_df = df[df["Class"] == 1]
        genuine_df = df[df["Class"] == 0]
        # Keep ALL fraud cases and sample genuine proportionally
        sample_genuine_count = min(len(genuine_df), max_samples - len(fraud_df))
        sampled_genuine = genuine_df.sample(n=sample_genuine_count, random_state=42)
        df = pd.concat([fraud_df, sampled_genuine]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
    return df


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    apply_smote: bool = True,
    smote_sampling_strategy: float = 0.20
):
    """
    Prepares train and test splits with zero data leakage:
    1. Stratified split into train and test sets.
    2. RobustScaler fitted ONLY on training set (Time, Amount).
    3. Test set transformed using fitted scaler.
    4. SMOTE applied ONLY to training set.
    
    Returns:
        X_train_processed, X_test_processed, y_train_processed, y_test, scaler, feature_names
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Feature columns: Time, V1-V28, Amount
    features = [c for c in df.columns if c != "Class"]
    X = df[features].copy()
    y = df["Class"].copy()

    # Step 1: Stratified split BEFORE any scaling or oversampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Step 2: Fit RobustScaler on Time and Amount on X_train ONLY
    # RobustScaler uses median and IQR, preventing extreme transaction amounts from skewing
    scaler = RobustScaler()
    
    cols_to_scale = ["Time", "Amount"]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # Save scaler for real-time inference
    joblib.dump(scaler, SCALER_PATH)

    # Step 3: Handle class imbalance using SMOTE on training split ONLY
    if apply_smote:
        # We synthesize minority samples to bring fraud proportion to a balanced, learnable ratio (e.g., 20% or 50%)
        # without overflowing training memory
        smote = SMOTE(sampling_strategy=smote_sampling_strategy, random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    return X_train_res, X_test_scaled, y_train_res, y_test, scaler, features


def preprocess_single_transaction(input_features: dict, scaler=None) -> np.ndarray:
    """
    Transforms a single incoming transaction dictionary into the exact feature vector
    expected by the trained ML models.
    Input dictionary must have: Time, V1..V28, Amount.
    """
    if scaler is None:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Train the model first.")

    ordered_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    # Create DataFrame with exact column order
    row_data = {}
    for col in ordered_cols:
        row_data[col] = float(input_features.get(col, 0.0))
        
    df_single = pd.DataFrame([row_data])
    
    # Scale Time and Amount using fitted scaler
    df_single[["Time", "Amount"]] = scaler.transform(df_single[["Time", "Amount"]])
    
    return df_single.values, df_single
