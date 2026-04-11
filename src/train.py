import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from model import LSTMAutoencoder
from features import build_features

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED        = 42
SEQ_LEN     = 20
LSTM_EPOCHS = 40
BATCH_SIZE  = 64
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR   = "models"
TARGET      = "spoofed"

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

FEATURE_COLS = [
    "correlator_symmetry", "correlator_distortion",
    "prompt_balance", "pip_pqp_ratio", "PC_magnitude",
    "residual_PD", "timing_residual",
    "pseudo_rate", "doppler_velocity",
    "phase_jump", "tcd_jump", "real_pseudo_jump",
    "Carrier_Doppler_hz_roll_std", "CN0_roll_std", "Pseudorange_m_roll_std",
    "Carrier_Doppler_hz_diff", "CN0_diff", "Pseudorange_m_diff",
    "CN0_mean_time", "CN0_std_time",
    "Carrier_Doppler_hz_std_time", "Pseudorange_m_std_time",
    "CN0_dev", "prn_count",
]


def make_sequences(X_arr, seq_len=SEQ_LEN):
    n = len(X_arr) - seq_len + 1
    if n <= 0:
        pad   = np.zeros((seq_len - len(X_arr), X_arr.shape[1]))
        X_arr = np.vstack([pad, X_arr])
        return X_arr[np.newaxis]
    return np.array([X_arr[i:i + seq_len] for i in range(n)])


def batch_reconstruction_errors(model, seqs_np, batch_size=256):
    """Compute reconstruction errors in small batches to avoid VRAM OOM."""
    all_errors = []
    for i in range(0, len(seqs_np), batch_size):
        batch = torch.FloatTensor(seqs_np[i:i + batch_size]).to(DEVICE)
        with torch.no_grad():
            recon  = model(batch)
            errors = torch.mean((recon - batch) ** 2, dim=(1, 2)).cpu().numpy()
        all_errors.append(errors)
        torch.cuda.empty_cache()
    return np.concatenate(all_errors)

def lstm_anomaly_scores(model, scaler, X_arr):
    X_norm = scaler.transform(X_arr)
    seqs   = make_sequences(X_norm)
    model.eval()
    errors = batch_reconstruction_errors(model, seqs)
    n_pad  = len(X_arr) - len(errors)
    return np.concatenate([np.full(n_pad, errors[0]), errors])

def ensemble_probs(xgb_probs, lstm_scores, alpha=0.70):
    lo, hi    = lstm_scores.min(), lstm_scores.max()
    lstm_norm = (lstm_scores - lo) / (hi - lo + 1e-9)
    return alpha * xgb_probs + (1 - alpha) * lstm_norm

def tune_threshold(probs, y_true):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.01, 0.71, 0.01):
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds, average="weighted", zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    print(f"Best threshold: {best_t:.2f}   Best weighted-F1: {best_f1:.4f}")
    return best_t

def train_lstm(X_genuine, input_dim):
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_genuine)

    seqs = make_sequences(X_norm)
    n_val   = max(1, int(len(seqs) * 0.15))
    n_train = len(seqs) - n_val

    # Keep sequences on CPU in DataLoader — move to GPU per batch
    train_ds = TensorDataset(torch.FloatTensor(seqs[:n_train]))
    loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model     = LSTMAutoencoder(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"Training LSTM on {DEVICE} — {LSTM_EPOCHS} epochs...")
    for epoch in range(LSTM_EPOCHS):
        model.train()
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{LSTM_EPOCHS}  loss={total/len(loader):.5f}")

    # Threshold at 97th percentile — computed in batches to avoid OOM
    model.eval()
    errors      = batch_reconstruction_errors(model, seqs[:n_train])
    lstm_thresh = float(np.percentile(errors, 97))
    print(f"LSTM anomaly threshold (97th pct): {lstm_thresh:.6f}")
    torch.cuda.empty_cache()

    return model, scaler, lstm_thresh

def train_xgboost(X, y):
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw      = neg / max(pos, 1)
    print(f"Class balance — genuine: {neg}  spoofed: {pos}  scale_pos_weight: {spw:.2f}")

    model = XGBClassifier(
        n_estimators          = 500,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        scale_pos_weight      = spw,
        eval_metric           = "logloss",
        early_stopping_rounds = 30,
        random_state          = SEED,
        n_jobs                = -1,
        verbosity             = 0,
    )

    # TimeSeriesSplit preserves temporal order — no future leaking into past
    from sklearn.model_selection import TimeSeriesSplit
    kf        = TimeSeriesSplit(n_splits=5)
    oof_probs = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        oof_probs[val_idx] = model.predict_proba(Xval)[:, 1]
        fold_f1 = f1_score(yval, (oof_probs[val_idx] >= 0.5).astype(int),
                           average="weighted")
        print(f"  Fold {fold}  weighted-F1: {fold_f1:.4f}")

    return model, oof_probs

def main(train_path="data/train.csv"):
    print("Loading raw data...")
    df_raw = pd.read_csv(train_path, low_memory=False)

    # ── Temporal split BEFORE feature engineering ──────────────────────────
    # Data is sequential so we split into two continuous blocks:
    #   train = first 80% of timestamps
    #   val   = last  20% of timestamps
    # This prevents:
    #   1. Future leaking into past via rolling features
    #   2. cross_satellite_features mixing train/val rows at shared timestamps
    # It also mimics real deployment where model predicts unseen future data.
    unique_times   = np.sort(df_raw['time'].unique())
    split_idx      = int(len(unique_times) * 0.80)
    train_times    = unique_times[:split_idx]
    val_times      = unique_times[split_idx:]
    df_train_raw   = df_raw[df_raw['time'].isin(train_times)].copy()
    df_val_raw     = df_raw[df_raw['time'].isin(val_times)].copy()
    print(f"Temporal split — train timestamps: {len(train_times)}  val timestamps: {len(val_times)}")

    print("Engineering features on train split...")
    df_train = build_features(df_train_raw)
    print("Engineering features on val split...")
    df_val   = build_features(df_val_raw)

    feat_cols = [c for c in FEATURE_COLS if c in df_train.columns]

    X_train = df_train[feat_cols].values.astype(np.float32)
    y_train = df_train[TARGET].values.astype(int)
    X_val   = df_val[feat_cols].values.astype(np.float32)
    y_val   = df_val[TARGET].values.astype(int)

    # Combine for full training after validation threshold is found
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    # LSTM — train on genuine-only rows from train split
    print("\n--- LSTM Autoencoder ---")
    lstm_model, scaler, lstm_thresh = train_lstm(
        X_train[y_train == 0], input_dim=len(feat_cols)
    )

    # Get LSTM scores on val split for honest threshold tuning
    lstm_val_scores = lstm_anomaly_scores(lstm_model, scaler, X_val)

    # XGBoost — train on train split, evaluate on val split
    print("\n--- XGBoost ---")
    xgb_model, xgb_oof = train_xgboost(X_train, y_train)
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]

    # Threshold tuning on held-out val split — honest evaluation
    print("\n--- Ensemble & Threshold Tuning (on val split) ---")
    val_ens     = ensemble_probs(xgb_val_probs, lstm_val_scores, alpha=0.70)
    best_thresh = tune_threshold(val_ens, y_val)

    val_f1 = f1_score(y_val, (val_ens >= best_thresh).astype(int), average="weighted")
    print(f"Honest val weighted-F1: {val_f1:.4f}")

    # Retrain XGBoost on full data for final model
    # Remove early_stopping_rounds since there is no eval_set here
    print("\nRetraining XGBoost on full data...")
    xgb_model.set_params(early_stopping_rounds=None)
    xgb_model.fit(X_all, y_all)

    # Save all artefacts
    joblib.dump(xgb_model, f"{MODEL_DIR}/xgb_model.pkl")
    joblib.dump(scaler,    f"{MODEL_DIR}/lstm_scaler.pkl")
    joblib.dump({
        "threshold":   best_thresh,
        "feat_cols":   feat_cols,
        "lstm_thresh": lstm_thresh,
    }, f"{MODEL_DIR}/config.pkl")
    torch.save(lstm_model.state_dict(), f"{MODEL_DIR}/lstm_model.pt")
    print(f"\nAll artefacts saved to '{MODEL_DIR}/'")

if __name__ == "__main__":
    main()