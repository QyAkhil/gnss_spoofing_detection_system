import pandas as pd
import numpy as np

LAMBDA_L1 = 299792458 / 1575420000  # ~0.1903 m
SPEED_OF_LIGHT = 299792458

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df['PRN'].astype(str).str.startswith('ch')].copy()

    num_cols = [
        'PRN', 'Carrier_Doppler_hz', 'Pseudorange_m', 'RX_time', 'TOW',
        'Carrier_phase', 'EC', 'LC', 'PC', 'PIP', 'PQP', 'TCD', 'CN0'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df[(df['CN0'] > 0) & (df['Pseudorange_m'] != 0)].copy()

    if df['channel'].dtype == object:
        df['channel'] = df['channel'].str.replace('ch', '').astype(int)

    df = df.sort_values(['PRN', 'RX_time']).reset_index(drop=True)
    return df

def correlator_features(df: pd.DataFrame) -> pd.DataFrame:
    # E-L asymmetry — most direct correlator spoofing indicator
    df['correlator_symmetry']   = (df['EC'] - df['LC']) / (df['PC'] + 1e-9)
    df['correlator_distortion'] = (df['EC'] - df['LC']).abs() / (df['PC'] + 1e-6)
    df['prompt_balance']        = df['PC'] / (df['EC'] + df['LC'] + 1e-6)
    df['pip_pqp_ratio']         = df['PIP'] / (df['PQP'] + 1e-6)
    df['PC_magnitude']          = (df['PIP']**2 + df['PQP']**2)**0.5
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

    return df

def temporal_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    # Rolling stats on key signals
    for col in ['Carrier_Doppler_hz', 'CN0', 'Pseudorange_m']:
        grp = df.groupby('PRN')[col]
        df[f'{col}_roll_std']  = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f'{col}_diff']      = grp.transform(lambda x: x.diff().fillna(0))

    # Phase and TCD jumps
    df['phase_jump'] = df.groupby('PRN')['Carrier_phase'].diff().abs().fillna(0)
    df['tcd_jump']   = df.groupby('PRN')['TCD'].diff().abs().fillna(0)

    # Gap-aware pseudorange jump
    typical_interval   = df.groupby('PRN')['time_delta'].transform('median')
    df['is_gap']       = (df['time_delta'] > 2 * typical_interval).astype(int)
    df['real_pseudo_jump'] = df['pseudo_delta'].abs() * (1 - df['is_gap'])

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