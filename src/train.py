import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest
from model import LSTMAutoencoder
from features import build_features, aggregate_to_time_level, get_time_level_feature_cols

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


def add_iso_score(iso: IsolationForest, X: np.ndarray, feat_cols: list) -> pd.DataFrame:
    """Append Isolation Forest anomaly score as a named column."""
    scores = iso.score_samples(X).reshape(-1, 1)
    return pd.DataFrame(np.hstack([X, scores]), columns=feat_cols + ['iso_score'])


def make_sequences(X_arr, seq_len=SEQ_LEN):
    """
    Create sequences over consecutive timestamps.
    Each row is one timestamp (already aggregated across 8 channels),
    so sliding windows capture temporal evolution directly.
    """
    if len(X_arr) < seq_len:
        return np.array([])

    sequences = []
    for i in range(len(X_arr) - seq_len + 1):
        sequences.append(X_arr[i:i + seq_len])

    return np.array(sequences)


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
    """Compute per-timestamp LSTM reconstruction error."""
    X_norm = scaler.transform(X_arr)
    seqs   = make_sequences(X_norm)

    if len(seqs) == 0:
        return np.zeros(len(X_arr))

    model.eval()
    errors = batch_reconstruction_errors(model, seqs)

    # Pad the initial timestamps that don't have full sequences
    n_pad  = len(X_arr) - len(errors)
    return np.concatenate([np.full(n_pad, errors[0]), errors])


def ensemble_probs(xgb_probs, lstm_scores, alpha=0.70):
    """Blend XGBoost probabilities with normalised LSTM anomaly scores.
    
    Uses robust percentile-based normalization (1st–99th percentile)
    to prevent extreme outliers from collapsing the score distribution.
    """
    lo = np.percentile(lstm_scores, 1)
    hi = np.percentile(lstm_scores, 99)
    lstm_clipped = np.clip(lstm_scores, lo, hi)
    lstm_norm = (lstm_clipped - lo) / (hi - lo + 1e-9)
    return alpha * xgb_probs + (1 - alpha) * lstm_norm

def tune_threshold(probs, y_true):
    """Find threshold that maximises binary F1 for spoofing detection."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.005):
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds, average="binary", zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    print(f"Best threshold: {best_t:.3f}   Best binary-F1: {best_f1:.4f}")
    return best_t

def train_lstm(X_genuine, input_dim):
    """Train LSTM autoencoder on genuine-only time-level data."""
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_genuine)

    seqs = make_sequences(X_norm, SEQ_LEN)
    if len(seqs) == 0:
        raise ValueError("Not enough genuine timestamps to create LSTM sequences")

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
        max_depth             = 7,
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

    # StratifiedKFold CV — for metric reporting only
    kf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        oof_probs[val_idx] = model.predict_proba(Xval)[:, 1]
        fold_f1 = f1_score(yval, (oof_probs[val_idx] >= 0.5).astype(int),
                           average="binary")
        print(f"  Fold {fold}  binary-F1: {fold_f1:.4f}  (spoofed in val: {yval.sum()})")

    return model, oof_probs

def main(train_path="data/train.csv"):
    print("Loading raw data...")
    df_raw = pd.read_csv(train_path, low_memory=False)

    # ── Stratified-temporal split BEFORE feature engineering ───────────────
    # Pure temporal split (first 80% / last 20%) is BROKEN for this dataset:
    # ALL spoofed data lives at timestamps 47743–63658, which falls entirely
    # within the first 80%. The val set gets 0 spoofed rows → meaningless eval.
    #
    # Fix: split WITHIN the spoofed time range and genuine-only time range
    # separately, so both train and val get spoofed samples.
    # Feature engineering is still done separately on each split.

    # Convert target to numeric early (header rows have string "0")
    df_raw[TARGET] = pd.to_numeric(df_raw[TARGET], errors='coerce').fillna(0).astype(int)

    unique_times = np.sort(df_raw['time'].unique())

    # Identify which timestamps contain spoofed data
    spoofed_times = np.sort(df_raw[df_raw[TARGET] == 1]['time'].unique())
    genuine_only_times = np.sort(np.setdiff1d(unique_times, spoofed_times))

    # Split spoofed timestamps: first 80% train, last 20% val
    sp_split = int(len(spoofed_times) * 0.80)
    spoofed_train_times = set(spoofed_times[:sp_split])
    spoofed_val_times   = set(spoofed_times[sp_split:])

    # Split genuine-only timestamps: first 80% train, last 20% val
    g_split = int(len(genuine_only_times) * 0.80)
    genuine_train_times = set(genuine_only_times[:g_split])
    genuine_val_times   = set(genuine_only_times[g_split:])

    train_times = spoofed_train_times | genuine_train_times
    val_times   = spoofed_val_times   | genuine_val_times

    df_train_raw = df_raw[df_raw['time'].isin(train_times)].copy()
    df_val_raw   = df_raw[df_raw['time'].isin(val_times)].copy()

    print(f"Stratified-temporal split:")
    print(f"  Train timestamps: {len(train_times)}  (spoofed: {len(spoofed_train_times)})")
    print(f"  Val   timestamps: {len(val_times)}  (spoofed: {len(spoofed_val_times)})")

    print("\n--- DATA CHECK ---")
    train_spoofed = (df_train_raw[TARGET] == 1).sum()
    val_spoofed   = (df_val_raw[TARGET] == 1).sum()
    print(f"Train rows: {len(df_train_raw)}  (spoofed: {train_spoofed})")
    print(f"Val   rows: {len(df_val_raw)}  (spoofed: {val_spoofed})")
    assert val_spoofed > 0, "FATAL: Validation set has 0 spoofed samples! Split is broken."

    # ── Feature engineering (per-channel) then aggregate to time level ────
    print("\nEngineering features on train split...")
    df_train_ch = build_features(df_train_raw)
    print("Engineering features on val split...")
    df_val_ch   = build_features(df_val_raw)

    print("\nAggregating channels to time level...")
    df_train = aggregate_to_time_level(df_train_ch, has_target=True)
    df_val   = aggregate_to_time_level(df_val_ch,   has_target=True)

    # Get feature columns (everything except 'time' and 'spoofed')
    feat_cols = get_time_level_feature_cols(df_train)
    # Ensure val has same columns
    for c in feat_cols:
        if c not in df_val.columns:
            df_val[c] = 0.0
    feat_cols = [c for c in feat_cols if c in df_val.columns]

    X_train = df_train[feat_cols].values.astype(np.float32)
    y_train = df_train[TARGET].values.astype(int)
    X_val   = df_val[feat_cols].values.astype(np.float32)
    y_val   = df_val[TARGET].values.astype(int)

    print(f"\nAfter time-level aggregation:")
    print(f"  Train: {len(X_train)} timestamps  (spoofed: {y_train.sum()})  features: {len(feat_cols)}")
    print(f"  Val:   {len(X_val)} timestamps  (spoofed: {y_val.sum()})  features: {len(feat_cols)}")
    assert y_val.sum() > 0, "FATAL: Val has 0 spoofed after aggregation!"

    # Combine for full training after validation threshold is found
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    # LSTM — train on genuine-only timestamps from train split
    print("\n--- LSTM Autoencoder ---")
    lstm_model, scaler, lstm_thresh = train_lstm(
        X_train[y_train == 0],
        input_dim=len(feat_cols))
    lstm_val_scores = lstm_anomaly_scores(lstm_model, scaler, X_val)

    # Isolation Forest — trained on genuine-only, score appended for XGBoost
    print("\n--- Isolation Forest ---")
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                         max_samples=0.8, random_state=SEED, n_jobs=-1)
    iso.fit(X_train[y_train == 0])
    print(f"Isolation Forest trained on {(y_train == 0).sum()} genuine timestamps.")
    X_train_aug = add_iso_score(iso, X_train, feat_cols)
    X_val_aug   = add_iso_score(iso, X_val,   feat_cols)
    aug_cols    = feat_cols + ['iso_score']

    # XGBoost — train on augmented features (with iso score)
    print("\n--- XGBoost ---")
    xgb_model, xgb_oof = train_xgboost(X_train_aug.values, y_train)
    xgb_val_probs = xgb_model.predict_proba(X_val_aug)[:, 1]

    # Ensemble — 70% XGBoost + 30% LSTM
    print("\n--- Ensemble & Threshold Tuning (on val split) ---")
    val_ens     = ensemble_probs(xgb_val_probs, lstm_val_scores, alpha=0.70)
    best_thresh = tune_threshold(val_ens, y_val)

    val_f1_binary   = f1_score(y_val, (val_ens >= best_thresh).astype(int), average="binary")
    val_f1_weighted = f1_score(y_val, (val_ens >= best_thresh).astype(int), average="weighted")
    print(f"Val binary-F1  (spoofing detection): {val_f1_binary:.4f}")
    print(f"Val weighted-F1 (overall):           {val_f1_weighted:.4f}")

    # Confusion matrix on val split
    val_preds = (val_ens >= best_thresh).astype(int)
    cm = confusion_matrix(y_val, val_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix (val split — per timestamp):")
    print(f"  TN (genuine correctly rejected) : {tn:>7}")
    print(f"  FP (genuine flagged as spoofed) : {fp:>7}")
    print(f"  FN (spoofed missed)             : {fn:>7}")
    print(f"  TP (spoofed correctly caught)   : {tp:>7}")
    print(f"\n{classification_report(y_val, val_preds, labels=[0, 1], target_names=['Genuine', 'Spoofed'], zero_division=0)}")

    # Retrain on full data for final model
    print("\nRetraining on full data...")
    iso_final = IsolationForest(n_estimators=200, contamination=0.05,
                                max_samples=0.8, random_state=SEED, n_jobs=-1)
    iso_final.fit(X_all[y_all == 0])
    X_all_aug = add_iso_score(iso_final, X_all, feat_cols)
    xgb_model.set_params(early_stopping_rounds=None)
    xgb_model.fit(X_all_aug, y_all)

    # Save all artefacts
    joblib.dump(xgb_model,  f"{MODEL_DIR}/xgb_model.pkl")
    joblib.dump(scaler,     f"{MODEL_DIR}/lstm_scaler.pkl")
    joblib.dump(iso_final,  f"{MODEL_DIR}/iso_forest.pkl")
    joblib.dump({
        "threshold":   best_thresh,
        "feat_cols":   feat_cols,
        "aug_cols":    aug_cols,
        "lstm_thresh": lstm_thresh,
    }, f"{MODEL_DIR}/config.pkl")
    torch.save(lstm_model.state_dict(), f"{MODEL_DIR}/lstm_model.pt")
    print(f"\nAll artefacts saved to '{MODEL_DIR}/'")

if __name__ == "__main__":
    main()