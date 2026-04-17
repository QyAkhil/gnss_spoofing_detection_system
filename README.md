# GNSS Spoofing Detection System

A machine learning pipeline to detect spoofed GNSS (GPS) signals. Built for the NyneOS Hackathon 2026.

## What it does

Takes raw GNSS receiver measurements (pseudorange, carrier phase, Doppler, correlator outputs, signal strength) and classifies each **timestamp** as **spoofed** or **authentic**. Each timestamp contains 8 receiver channels — since spoofing affects all channels simultaneously, the system predicts at the timestamp level rather than per-channel.

## How it works

The pipeline has four stages:

**1. Feature Engineering** (`src/features.py`)

33 physics-based features extracted per channel from raw signals:

- **Correlator features** — EC/LC ratio, Early-Late symmetry, prompt balance. A genuine signal produces a symmetric correlation peak; spoofed signals distort this.
- **Cross-measurement consistency** — Pseudorange vs carrier phase agreement, Doppler vs phase rate. These quantities are physically coupled, and a spoofer can't fake all of them consistently.
- **Temporal dynamics** — Rolling standard deviations, first differences, rate-of-change acceleration. Spoofing attacks produce abrupt signal transitions that stand out against smooth genuine behaviour.
- **Cross-satellite statistics** — Per-timestamp mean/std of CN0 and Doppler across all tracked satellites. A single-antenna spoofer produces unnaturally uniform signals.

**2. Time-Level Aggregation**

The 33 per-channel features are aggregated across all 8 channels at each timestamp using mean, std, min, and max — producing ~120 aggregated features per timestamp. This captures both the central tendency and the spread across channels (spoofed signals are unnaturally uniform).

**3. Anomaly Detection**

- **LSTM Autoencoder** (`src/model.py`) — Trained only on genuine timestamps. Learns the normal temporal patterns of GNSS measurements. At inference, high reconstruction error = anomalous = likely spoofed.
- **Isolation Forest** — Unsupervised outlier detector trained on genuine-only data. Its anomaly score is appended as an extra feature for XGBoost.

**4. Classification**

- **XGBoost** — Gradient-boosted classifier trained on the aggregated features + Isolation Forest score. Uses StratifiedKFold cross-validation (5 folds) with early stopping.
- **Ensemble** — Final prediction is a weighted blend: `0.70 * XGBoost + 0.30 * LSTM anomaly score`. The threshold is tuned on a held-out validation split to maximise binary F1.

## Project structure

```
gnss_spoofing_detection_system/
|
|-- data/
|   |-- train.csv          # Training data (891K rows = 111K timestamps × 8 channels)
|   |-- test.csv           # Test data (382K rows = 47K timestamps × 8 channels)
|
|-- src/
|   |-- features.py        # Feature engineering + time-level aggregation
|   |-- model.py           # LSTM autoencoder architecture
|   |-- train.py           # Training script (main entry point)
|   |-- predict.py         # Generate submission from trained models
|   |-- check.py           # Diagnostic plots and reports
|
|-- models/                # Saved model artefacts (after training)
|   |-- xgb_model.pkl
|   |-- lstm_model.pt
|   |-- lstm_scaler.pkl
|   |-- iso_forest.pkl
|   |-- config.pkl
|
|-- outputs/
|   |-- submission.csv     # Generated predictions (1 row per timestamp)
|
|-- requirements.txt
```

## Setup

```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA optional (LSTM trains on GPU if available, falls back to CPU).

## Usage

**Train:**
```bash
cd src
python train.py
```
Saves all artefacts to `models/`.

**Predict:**
```bash
cd src
python predict.py
```
Writes `outputs/submission.csv` with columns: `time`, `spoofed`, `confidence` — one row per timestamp (~47K rows).

## Data split strategy

The training data contains spoofing only at timestamps 47,743–63,658 (out of 1–111,401). A naive 80/20 temporal split puts all spoofing in the train set and none in validation, which breaks evaluation.

The current approach splits spoofed and genuine timestamp ranges separately, then takes the last 20% of each for validation. This ensures both train and val contain spoofed data while preserving temporal ordering.

## Key design decisions

- **Time-level prediction.** Spoofing affects all 8 channels at a given timestamp simultaneously (0 mixed-label timestamps in training data). Aggregating channels gives richer per-timestamp features and produces the correct output granularity.
- **Channel aggregation with mean/std/min/max.** The std across channels is itself a strong feature — spoofed signals show unnaturally low variance across channels because they come from a single source.
- **LSTM as anomaly detector, not classifier.** Trained only on genuine data, so it doesn't need spoofed examples to generalise. The reconstruction error becomes a feature for XGBoost rather than a direct prediction.
- **Isolation Forest score as a feature.** Gives XGBoost an independent unsupervised signal to weigh alongside the physics features.
- **70/30 ensemble blending.** XGBoost handles the structured feature relationships well; LSTM captures temporal patterns. The blend is a simple weighted average — nothing fancy, but it works.
- **Physics-motivated features over raw signal values.** Ratios and consistency metrics (e.g. pseudorange/carrier_phase) generalise better across time periods than absolute measurements like pseudorange_m, which change as satellites orbit.
