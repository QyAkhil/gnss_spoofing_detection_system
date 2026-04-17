import pandas as pd
import numpy as np

LAMBDA_L1 = 299792458 / 1575420000  # ~0.1903 m
SPEED_OF_LIGHT = 299792458

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- CLEANING DEBUG ---")
    print(f"Initial rows: {len(df)}")

    # Step 1: Remove header artefacts
    before = len(df)
    df = df[~df['PRN'].astype(str).str.startswith('ch')].copy()
    print(f"After removing header artefacts: {len(df)}  (dropped {before - len(df)})")

    # Step 2: Convert numeric
    num_cols = [
        'PRN', 'Carrier_Doppler_hz', 'Pseudorange_m', 'RX_time', 'TOW',
        'Carrier_phase', 'EC', 'LC', 'PC', 'PIP', 'PQP', 'TCD', 'CN0'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Step 3: Check NaNs after conversion
    # print("NaN counts after numeric conversion:")
    # print(df[num_cols].isna().sum())

    # Step 4: Remove invalid signals
    # Step 4: Remove invalid signals (FIXED)
    before = len(df)

    # Keep only valid signal strength
    df = df[df['CN0'] >= 0].copy()

    # KEEP pseudorange = 0 and encode it
    df['pseudo_zero_flag'] = (df['Pseudorange_m'] == 0).astype(int)

    print(f"After CN0 filter: {len(df)}  (dropped {before - len(df)})")
    # Step 5: Channel cleaning
    if df['channel'].dtype == object:
        df['channel'] = df['channel'].str.replace('ch', '').astype(int)

    df = df.sort_values(['PRN', 'RX_time']).reset_index(drop=True)

    print(f"Final rows after cleaning: {len(df)}")
    print("--- END CLEANING ---\n")

    return df
    
def correlator_features(df: pd.DataFrame) -> pd.DataFrame:
    # E-L asymmetry — most direct correlator spoofing indicator
    df['correlator_symmetry']   = (df['EC'] - df['LC']) / (df['PC'] + 1e-9)
    df['correlator_distortion'] = (df['EC'] - df['LC']).abs() / (df['PC'] + 1e-6)
    df['prompt_balance']        = df['PC'] / (df['EC'] + df['LC'] + 1e-6)
    df['pip_pqp_ratio']         = df['PIP'] / (df['PQP'] + 1e-6)
    df['PC_magnitude']          = (df['PIP']**2 + df['PQP']**2)**0.5

    # NEW: Direct EC/LC ratio — should be ~1 for genuine signals.
    # Spoofers using a single antenna create asymmetric correlator outputs.
    df['ec_lc_ratio'] = df['EC'] / (df['LC'] + 1e-9)

    # NEW: Correlator power ratio — total correlator energy vs prompt
    df['el_power_ratio'] = (df['EC']**2 + df['LC']**2) / (df['PC']**2 + 1e-9)

    return df

def physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df['time_delta']  = df.groupby('PRN')['RX_time'].diff()
    df['pseudo_delta'] = df.groupby('PRN')['Pseudorange_m'].diff()

    valid_dt = df['time_delta'].abs() > 1e-6
    df['pseudo_rate'] = 0.0
    df.loc[valid_dt, 'pseudo_rate'] = (
        df.loc[valid_dt, 'pseudo_delta'] / df.loc[valid_dt, 'time_delta']
    )

    # PIR — should be ~0 for genuine signals
    df['doppler_velocity'] = LAMBDA_L1 * df['Carrier_Doppler_hz']
    df['residual_PD']      = (df['pseudo_rate'] + df['doppler_velocity']).abs()

    # Timing consistency
    df['timing_residual'] = (
        (df['TOW'] - df['RX_time']) - df['Pseudorange_m'] / SPEED_OF_LIGHT
    ).abs()

    # NEW: Carrier-phase to pseudorange consistency
    # For genuine signals, carrier_phase * wavelength should closely track pseudorange.
    # Spoofed signals often have mismatches because the spoofer generates
    # pseudorange and carrier phase independently.
    df['carrier_pseudo_consistency'] = (
        df['Carrier_phase'] * LAMBDA_L1 - df['Pseudorange_m']
    ).abs()

    # NEW: Doppler vs carrier-phase-rate consistency
    # d(carrier_phase)/dt should match Carrier_Doppler_hz for genuine signals.
    phase_rate = df.groupby('PRN')['Carrier_phase'].diff() / df['time_delta']
    df['doppler_phase_consistency'] = (
        phase_rate - df['Carrier_Doppler_hz']
    ).abs().fillna(0)

    return df

def temporal_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    # ─────────────────────────────────────────────
    # Temporal statistics (STRICTLY CAUSAL)
    # We use shift(1) to ensure features depend ONLY on past values.
    # This prevents leakage from current timestep into features.
    # ─────────────────────────────────────────────
    for col in ['Carrier_Doppler_hz', 'CN0', 'Pseudorange_m']:
        grp = df.groupby('PRN')[col]

        # Rolling std using ONLY past values
        # shift(1) → excludes current value
        # rolling → captures temporal variability (spoofing causes instability)
        df[f'{col}_roll_std'] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )

        # First-order difference (current - previous)
        # Captures sudden jumps (common in spoofing)
        df[f'{col}_diff'] = grp.transform(
            lambda x: x.diff().fillna(0)
        )

    # ─────────────────────────────────────────────
    # Signal discontinuity indicators
    # Large jumps often indicate spoofing or signal takeover
    # ─────────────────────────────────────────────
    df['phase_jump'] = df.groupby('PRN')['Carrier_phase'].diff().abs().fillna(0)
    df['tcd_jump']   = df.groupby('PRN')['TCD'].diff().abs().fillna(0)

    # ─────────────────────────────────────────────
    # Gap-aware pseudorange jump
    # We estimate "typical interval" using ONLY past values (causal median)
    # This avoids using future timestamps → prevents subtle leakage
    # ─────────────────────────────────────────────
    typical_interval = df.groupby('PRN')['time_delta'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).median()
    )

    # Detect gaps (missing data / irregular sampling)
    df['is_gap'] = (df['time_delta'] > 2 * typical_interval).astype(int)

    # Only consider pseudorange jumps when NOT caused by gaps
    # This isolates real anomalies from missing-data artifacts
    df['real_pseudo_jump'] = df['pseudo_delta'].abs() * (1 - df['is_gap'])

    # NEW: Pseudorange rate acceleration (2nd derivative)
    # Genuine signals have smooth dynamics; spoofed signals show sudden changes.
    df['pseudo_rate_accel'] = df.groupby('PRN')['pseudo_rate'].diff().abs().fillna(0)

    # NEW: TOW consistency — TOW should increment smoothly.
    # Spoofed signals may show irregular TOW jumps.
    tow_diff = df.groupby('PRN')['TOW'].diff()
    expected_tow_diff = df['time_delta']  # TOW should advance at ~1 sec/sec
    df['tow_jump'] = (tow_diff - expected_tow_diff).abs().fillna(0)

    return df


def cross_satellite_features(df: pd.DataFrame) -> pd.DataFrame:
    # At each timestamp — disagreement across PRNs is the key spoofing signal
    # NOTE: these must be computed WITHIN a single split (train or val separately)
    # to avoid data leakage. build_features() should never be called on the
    # full combined train+val dataframe.
    for col in ['CN0', 'Carrier_Doppler_hz', 'Pseudorange_m']:
        df[f'{col}_mean_time'] = df.groupby('RX_time')[col].transform('mean')
        df[f'{col}_std_time']  = df.groupby('RX_time')[col].transform('std').fillna(0)

    df['CN0_dev']   = df['CN0'] - df['CN0_mean_time']
    df['prn_count'] = df.groupby('RX_time')['PRN'].transform('count')

    # NEW: Doppler deviation from cross-satellite mean
    # Single-antenna spoofers produce similar Doppler shifts across all PRNs,
    # so per-PRN deviation from the mean is abnormally small.
    df['doppler_dev'] = (df['Carrier_Doppler_hz'] - df['Carrier_Doppler_hz_mean_time']).abs()

    # NEW: Pseudorange deviation from cross-satellite mean
    df['pseudo_dev'] = (df['Pseudorange_m'] - df['Pseudorange_m_mean_time']).abs()

    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean(df)
    df = correlator_features(df)
    df = physics_features(df)
    df = temporal_features(df)
    df = cross_satellite_features(df)
    df = df.fillna(0)
    print(f"Done. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# TIME-LEVEL AGGREGATION
# ─────────────────────────────────────────────
# Each timestamp has exactly 8 channels with identical spoofing labels.
# We aggregate per-channel features across the 8 channels using
# mean / std / min / max to produce ONE row per timestamp.

# Features that are already cross-satellite (computed at time level)
# — just take the first value since they're identical across channels.
_TIME_LEVEL_COLS = {
    'CN0_mean_time', 'CN0_std_time',
    'Carrier_Doppler_hz_mean_time', 'Carrier_Doppler_hz_std_time',
    'Pseudorange_m_mean_time', 'Pseudorange_m_std_time',
    'prn_count',
}

# Per-channel features to aggregate with mean/std/min/max
_PER_CHANNEL_FEATURES = [
    'correlator_symmetry', 'correlator_distortion',
    'prompt_balance', 'pip_pqp_ratio', 'PC_magnitude',
    'ec_lc_ratio', 'el_power_ratio',
    'residual_PD', 'timing_residual',
    'carrier_pseudo_consistency', 'doppler_phase_consistency',
    'pseudo_rate', 'doppler_velocity',
    'phase_jump', 'tcd_jump', 'real_pseudo_jump',
    'pseudo_rate_accel', 'tow_jump',
    'Carrier_Doppler_hz_roll_std', 'CN0_roll_std', 'Pseudorange_m_roll_std',
    'Carrier_Doppler_hz_diff', 'CN0_diff', 'Pseudorange_m_diff',
    'CN0_dev', 'pseudo_zero_flag',
    'doppler_dev', 'pseudo_dev',
]


def aggregate_to_time_level(df: pd.DataFrame,
                            has_target: bool = True) -> pd.DataFrame:
    """
    Collapse 8 channel-rows per timestamp into 1 row per timestamp.

    Parameters
    ----------
    df : DataFrame with per-channel features (output of build_features)
    has_target : True for training data (has 'spoofed' column)

    Returns
    -------
    DataFrame with one row per 'time', aggregated features.
    """
    per_ch = [c for c in _PER_CHANNEL_FEATURES if c in df.columns]
    time_lvl = [c for c in _TIME_LEVEL_COLS if c in df.columns]

    # Build aggregation dict
    agg_dict = {}

    # Per-channel features → mean, std, min, max
    for col in per_ch:
        agg_dict[col] = ['mean', 'std', 'min', 'max']

    # Time-level features → just take first (identical across channels)
    for col in time_lvl:
        agg_dict[col] = 'first'

    if has_target and 'spoofed' in df.columns:
        agg_dict['spoofed'] = 'max'  # all same within a timestamp

    grouped = df.groupby('time').agg(agg_dict)

    # Flatten multi-level column names: ('correlator_symmetry', 'mean') → 'correlator_symmetry_mean'
    new_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            # Time-level cols and target: keep original name (no suffix)
            if col[1] == 'first' or col[0] == 'spoofed':
                new_cols.append(col[0])
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
        else:
            new_cols.append(col)

    grouped.columns = new_cols
    grouped = grouped.reset_index()

    # Replace NaN std (can happen if a channel feature is constant across 8 channels)
    grouped = grouped.fillna(0)

    print(f"Aggregated to time level: {grouped.shape[0]} timestamps × {grouped.shape[1]} columns")
    return grouped


def get_time_level_feature_cols(df: pd.DataFrame) -> list:
    """Return the list of feature column names from an aggregated DataFrame."""
    exclude = {'time', 'spoofed'}
    return [c for c in df.columns if c not in exclude]