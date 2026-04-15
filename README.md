# GNSS Spoofing Detection System

A machine learning pipeline to detect spoofed GNSS (GPS) signals. Built for the NyneOS Hackathon -- Kaizen 2026.

## What it does

Takes raw GNSS receiver measurements (pseudorange, carrier phase, Doppler, correlator outputs, signal strength) and classifies each observation as **spoofed** or **authentic**. The system uses an LSTM autoencoder to learn what genuine signals look like, then combines that with a gradient-boosted classifier to flag anomalies.

## How it works

The pipeline has three stages:

**1. Feature Engineering** (`src/features.py`)

33 physics-based features extracted from raw signals:

- **Correlator features** -- EC/LC ratio, Early-Late symmetry, prompt balance. A genuine signal produces a symmetric correlation peak; spoofed signals distort this.
- **Cross-measurement consistency** -- Pseudorange vs carrier phase agreement, Doppler vs phase rate. These quantities are physically coupled, and a spoofer can't fake all of them consistently.
- **Temporal dynamics** -- Rolling standard deviations, first differences, rate-of-change acceleration. Spoofing attacks produce abrupt signal transitions that stand out against smooth genuine behaviour.
- **Cross-satellite statistics** -- Per-timestamp mean/std of CN0 and Doppler across all tracked satellites. A single-antenna spoofer produces unnaturally uniform signals.

**2. Anomaly Detection**

- **LSTM Autoencoder** (`src/model.py`) -- Trained only on genuine signals. Learns the normal temporal patterns of GNSS measurements. At inference, high reconstruction error = anomalous = likely spoofed.
- **Isolation Forest** -- Unsupervised outlier detector trained on genuine-only data. Its anomaly score is appended as an extra feature for XGBoost.

**3. Classification**

- **XGBoost** -- Gradient-boosted classifier trained on the 33 features + Isolation Forest score. Uses StratifiedKFold cross-validation (5 folds) with early stopping.
- **Ensemble** -- Final prediction is a weighted blend: `0.70 * XGBoost + 0.30 * LSTM anomaly score`. The threshold is tuned on a held-out validation split to maximise binary F1.

## Project structure

```
gnss_spoofing_detection_system/
|
|-- data/
|   |-- train.csv          # Training data (891K rows, 16 columns)
|   |-- test.csv           # Test data for submission
|
|-- src/
|   |-- features.py        # Feature engineering pipeline
|   |-- model.py           # LSTM autoencoder architecture
|   |-- train.py           # Training script (main entry point)
|   |-- predict.py         # Generate submission from trained models
|
|-- models/                # Saved model artefacts (after training)
|   |-- xgb_model.pkl
|   |-- lstm_model.pt
|   |-- lstm_scaler.pkl
|   |-- iso_forest.pkl
|   |-- config.pkl
|
|-- outputs/
|   |-- submission.csv     # Generated predictions
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
Takes about 5-10 minutes depending on hardware. Saves all artefacts to `models/`.

**Predict:**
```bash
cd src
python predict.py
```
Writes `outputs/submission.csv` with `spoofed` and `confidence` columns.

## Data split strategy

The training data contains spoofing only at timestamps 47,743-63,658 (out of 1-111,401). A naive 80/20 temporal split puts all spoofing in the train set and none in validation, which breaks evaluation.

The current approach splits spoofed and genuine timestamp ranges separately, then takes the last 20% of each for validation. This ensures both train and val contain spoofed data while preserving temporal ordering.

## Results

Cross-validation (5-fold StratifiedKFold):
- Binary F1: ~0.93

Held-out validation (temporal split):
- Binary F1: ~0.73
- Precision: 0.97, Recall: 0.58

The gap between CV and validation reflects distribution shift -- the model sees spoofing from one time range during training and is evaluated on a different range. The CV score is more representative of per-timestamp performance within any given time period.

## Key design decisions

- **LSTM as anomaly detector, not classifier.** Trained only on genuine data, so it doesn't need spoofed examples to generalise. The reconstruction error becomes a feature for XGBoost rather than a direct prediction.
- **Isolation Forest score as a feature.** Gives XGBoost an independent unsupervised signal to weigh alongside the physics features.
- **70/30 ensemble blending.** XGBoost handles the structured feature relationships well; LSTM captures temporal patterns. The blend is a simple weighted average -- nothing fancy, but it works.
- **Physics-motivated features over raw signal values.** Ratios and consistency metrics (e.g. pseudorange/carrier_phase) generalise better across time periods than absolute measurements like pseudorange_m, which change as satellites orbit.
