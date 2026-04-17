import os
import numpy as np
import pandas as pd
import joblib
import torch

from model import LSTMAutoencoder
from features import build_features, aggregate_to_time_level
from train import lstm_anomaly_scores, ensemble_probs, add_iso_score

MODEL_DIR = "models"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def predict(test_path="data/test.csv", out_path="outputs/submission.csv"):
    # Load saved artefacts
    xgb_model = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    scaler    = joblib.load(f"{MODEL_DIR}/lstm_scaler.pkl")
    iso       = joblib.load(f"{MODEL_DIR}/iso_forest.pkl")
    cfg       = joblib.load(f"{MODEL_DIR}/config.pkl")

    feat_cols = cfg["feat_cols"]
    threshold = cfg["threshold"]

    lstm_model = LSTMAutoencoder(input_dim=len(feat_cols)).to(DEVICE)
    lstm_model.load_state_dict(
        torch.load(f"{MODEL_DIR}/lstm_model.pt", map_location=DEVICE,
                   weights_only=True)
    )
    lstm_model.eval()

    # Load and engineer test features
    print("Loading test data...")
    df = pd.read_csv(test_path, low_memory=False)

    # Per-channel feature engineering
    df = build_features(df)

    # Aggregate 8 channels → 1 row per timestamp
    print("Aggregating channels to time level...")
    df_agg = aggregate_to_time_level(df, has_target=False)

    # Ensure all expected feature columns exist
    for c in feat_cols:
        if c not in df_agg.columns:
            df_agg[c] = 0.0

    X = df_agg[feat_cols].values.astype(np.float32)

    # Match training: LSTM scores + iso-augmented XGBoost + 70/30 ensemble
    lstm_scores = lstm_anomaly_scores(lstm_model, scaler, X)
    X_aug = add_iso_score(iso, X, feat_cols)
    xgb_probs = xgb_model.predict_proba(X_aug)[:, 1]

    ens   = ensemble_probs(xgb_probs, lstm_scores, alpha=0.70)
    preds = (ens >= threshold).astype(int)

    print(f"Predictions — Spoofed: {preds.sum()}  Genuine: {(preds == 0).sum()}")
    print(f"Total timestamps: {len(preds)}")

    # Output: one row per timestamp with time, spoofed, confidence
    submission = pd.DataFrame({
        "time":       df_agg["time"].values,
        "spoofed":    preds,
        "confidence": ens,
    })

    os.makedirs("outputs", exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Saved {len(submission)} timestamps -> {out_path}")


if __name__ == "__main__":
    predict()