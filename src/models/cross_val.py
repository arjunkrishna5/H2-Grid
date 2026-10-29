# H2-Grid India: Time-Series Cross-Validation & Dataset Slicing Harness
# Implements 80/20 chronological train/test split (2019-2022 train, 2023 holdout test).
# Provides 6-fold expanding-window walk-forward backtesting across seasonal transitions.
# Generates 2D tabular matrices for LightGBM and 3D sliding-window sequences for BiLSTM.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

HUBS = ["bhadla", "pavagada", "ramanathapuram", "palakkad"]
TRAIN_END_TIMESTAMP = "2022-12-31 23:00:00"
TEST_START_TIMESTAMP = "2023-01-01 00:00:00"


@dataclass
class SeasonalFold:
    """Represents a single time-series cross-validation fold with metadata."""
    fold_index: int
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_size: int
    val_size: int


def load_hub_data(hub_id: str, scaled: bool = True) -> pd.DataFrame:
    """
    Loads preprocessed feature parquet for a designated hub.
    
    Args:
        hub_id: One of 'bhadla', 'pavagada', 'ramanathapuram', 'palakkad'.
        scaled: If True, loads 0-1 normalized features (for BiLSTM).
                If False, loads unscaled physical units (for LightGBM/Tree models).
    Returns:
        pd.DataFrame with DatetimeIndex (43,824 hourly rows).
    """
    hub_id = hub_id.lower().strip()
    if hub_id not in HUBS:
        raise ValueError(f"Unknown hub '{hub_id}'. Must be one of {HUBS}")
    
    suffix = "scaled" if scaled else "unscaled"
    parquet_path = PROCESSED_DIR / f"{hub_id}_features_{suffix}.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing processed data: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def train_test_split_timeseries(
    df: pd.DataFrame,
    train_end: str = TRAIN_END_TIMESTAMP,
    test_start: str = TEST_START_TIMESTAMP,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slices dataset chronologically into 80% train (2019-2022) and 20% test (2023 holdout).
    Guarantees zero future temporal leakage.
    
    Args:
        df: Processed DataFrame indexed by hourly timestamps.
        train_end: Inclusive cutoff for training set (default: 2022-12-31 23:00:00).
        test_start: Inclusive start for holdout test set (default: 2023-01-01 00:00:00).
    Returns:
        (df_train, df_test)
    """
    df_train = df.loc[:train_end].copy()
    df_test = df.loc[test_start:].copy()
    
    if len(df_train) == 0:
        raise ValueError(f"Training split is empty with cutoff {train_end}")
    if len(df_test) == 0:
        raise ValueError(f"Holdout test split is empty with start {test_start}")
    
    # Assert zero overlap
    assert df_train.index.max() < df_test.index.min(), (
        f"Temporal leakage detected: train max {df_train.index.max()} >= test min {df_test.index.min()}"
    )
    return df_train, df_test


def extract_X_y(
    df: pd.DataFrame,
    target_col: str = "ghi",
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates feature matrix X from target vector y.
    By default, drops 'dni' as well to prevent concurrent solar leakage.
    
    Args:
        df: DataFrame containing features and target.
        target_col: Target column to forecast (default: 'ghi').
        drop_cols: Additional columns to drop from X (default: ['dni']).
    Returns:
        (X, y)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns.")
    
    cols_to_drop = [target_col]
    if drop_cols is None:
        if "dni" in df.columns:
            cols_to_drop.append("dni")
    else:
        for c in drop_cols:
            if c in df.columns and c not in cols_to_drop:
                cols_to_drop.append(c)
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    return X, y


def get_seasonal_expanding_folds(df_train: pd.DataFrame) -> List[SeasonalFold]:
    """
    Builds a 6-fold expanding-window cross-validation schedule across the 4-year
    training horizon (2019-2022 = 35,064 hours).
    
    Folds test across distinct Indian climatic transitions (Monsoon, Winter, Summer):
      Fold 1: Train 2019 (8,760 hrs)       -> Val: 2020 Winter/Spring (4,368 hrs)
      Fold 2: Train 2019-2020 May (12,408) -> Val: 2020 SW Monsoon (Jun-Sep, 2,928 hrs)
      Fold 3: Train 2019-2020 Sep (15,336) -> Val: 2020-21 Post-Monsoon/Winter (3,624 hrs)
      Fold 4: Train 2019-2021 Feb (18,960) -> Val: 2021 Summer Pre-Monsoon (2,208 hrs)
      Fold 5: Train 2019-2021 May (21,168) -> Val: 2021 SW Monsoon (Jun-Sep, 2,928 hrs)
      Fold 6: Train 2019-2021 Dec (26,304) -> Val: 2022 Full Annual Walk-Forward (8,760 hrs)
    """
    fold_specs = [
        ("Fold 1 (2020 Winter/Spring)", "2019-01-01 00:00:00", "2019-12-31 23:00:00", "2020-01-01 00:00:00", "2020-05-31 23:00:00"),
        ("Fold 2 (2020 SW Monsoon)", "2019-01-01 00:00:00", "2020-05-31 23:00:00", "2020-06-01 00:00:00", "2020-09-30 23:00:00"),
        ("Fold 3 (2020-21 Post-Monsoon/Winter)", "2019-01-01 00:00:00", "2020-09-30 23:00:00", "2020-10-01 00:00:00", "2021-02-28 23:00:00"),
        ("Fold 4 (2021 Summer/Pre-Monsoon)", "2019-01-01 00:00:00", "2021-02-28 23:00:00", "2021-03-01 00:00:00", "2021-05-31 23:00:00"),
        ("Fold 5 (2021 SW Monsoon)", "2019-01-01 00:00:00", "2021-05-31 23:00:00", "2021-06-01 00:00:00", "2021-09-30 23:00:00"),
        ("Fold 6 (2022 Full-Year Walk-Forward)", "2019-01-01 00:00:00", "2021-12-31 23:00:00", "2022-01-01 00:00:00", "2022-12-31 23:00:00"),
    ]
    
    folds = []
    for idx, (name, tr_start, tr_end, val_start, val_end) in enumerate(fold_specs, start=1):
        tr_mask = (df_train.index >= tr_start) & (df_train.index <= tr_end)
        val_mask = (df_train.index >= val_start) & (df_train.index <= val_end)
        
        tr_size = int(tr_mask.sum())
        val_size = int(val_mask.sum())
        
        folds.append(
            SeasonalFold(
                fold_index=idx,
                name=name,
                train_start=pd.Timestamp(tr_start),
                train_end=pd.Timestamp(tr_end),
                val_start=pd.Timestamp(val_start),
                val_end=pd.Timestamp(val_end),
                train_size=tr_size,
                val_size=val_size,
            )
        )
    return folds


def get_expanding_window_indices(
    n_samples: int,
    n_splits: int = 6,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Standard programmatic expanding window CV using scikit-learn TimeSeriesSplit.
    
    Args:
        n_samples: Total number of rows in the training set.
        n_splits: Number of backtesting folds (default: 6).
    Yields:
        (train_indices, val_indices) for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(n_samples)
    for train_idx, val_idx in tscv.split(indices):
        yield train_idx, val_idx


def create_sequences(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    lookback: int = 24,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D tabular features into 3D sequential sliding windows for BiLSTM.
    
    Args:
        X: Feature matrix of shape (N, F).
        y: Target series of shape (N,).
        lookback: Number of past hourly steps in sequence window (default: 24).
        horizon: Number of steps ahead to predict (default: 1).
    Returns:
        X_seq: 3D array of shape (N - lookback - horizon + 1, lookback, F)
        y_seq: 1D or 2D array of target values. If horizon=1, shape is (N - lookback,).
               If horizon > 1, shape is (N - lookback - horizon + 1, horizon).
    """
    X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    
    num_samples = len(X_arr) - lookback - horizon + 1
    if num_samples <= 0:
        raise ValueError(
            f"Dataset length ({len(X_arr)}) is insufficient for lookback={lookback} and horizon={horizon}"
        )
    
    num_features = X_arr.shape[1]
    X_seq = np.empty((num_samples, lookback, num_features), dtype=np.float32)
    
    if horizon == 1:
        y_seq = np.empty((num_samples,), dtype=np.float32)
        for i in range(num_samples):
            X_seq[i] = X_arr[i : i + lookback]
            y_seq[i] = y_arr[i + lookback]
    else:
        y_seq = np.empty((num_samples, horizon), dtype=np.float32)
        for i in range(num_samples):
            X_seq[i] = X_arr[i : i + lookback]
            y_seq[i] = y_arr[i + lookback : i + lookback + horizon]
            
    return X_seq, y_seq


def get_tabular_splits(
    hub_id: str,
    scaled: bool = False,
    drop_dni: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Convenience method returning tabular train/test splits for LightGBM / SARIMA.
    
    Args:
        hub_id: Hub identifier.
        scaled: Whether to use scaled features (False for LightGBM trees).
        drop_dni: Whether to drop 'dni' from features.
    Returns:
        (X_train, y_train, X_test, y_test, df_train, df_test)
    """
    df = load_hub_data(hub_id=hub_id, scaled=scaled)
    df_train, df_test = train_test_split_timeseries(df)
    
    drop_cols = ["dni"] if drop_dni else []
    X_train, y_train = extract_X_y(df_train, target_col="ghi", drop_cols=drop_cols)
    X_test, y_test = extract_X_y(df_test, target_col="ghi", drop_cols=drop_cols)
    
    return X_train, y_train, X_test, y_test, df_train, df_test


def get_sequence_splits(
    hub_id: str,
    scaled: bool = True,
    lookback: int = 24,
    horizon: int = 1,
    drop_dni: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Convenience method returning sequential tensor arrays for BiLSTM.
    
    Args:
        hub_id: Hub identifier.
        scaled: Whether to use scaled features (True for BiLSTM).
        lookback: Past context window length in hours (default: 24).
        horizon: Forecast horizon (default: 1).
        drop_dni: Whether to drop 'dni' from features.
    Returns:
        (X_train_seq, y_train_seq, X_test_seq, y_test_seq, df_train, df_test)
    """
    X_train, y_train, X_test, y_test, df_train, df_test = get_tabular_splits(
        hub_id=hub_id, scaled=scaled, drop_dni=drop_dni
    )
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback=lookback, horizon=horizon)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback=lookback, horizon=horizon)
    
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, df_train, df_test


if __name__ == "__main__":
    print("=" * 78)
    print("H2-Grid India: Subphase 3.1 Verification - Cross-Validation Harness")
    print("=" * 78)
    
    for hub in HUBS:
        print(f"\n>>> Verifying Hub: {hub.upper()}")
        
        # 1. Tabular Split (Unscaled for LightGBM)
        X_tr, y_tr, X_te, y_te, df_tr, df_te = get_tabular_splits(hub, scaled=False)
        print(f"  [Tabular] Train records: {len(X_tr):,} rows | Test records: {len(X_te):,} rows")
        print(f"  [Tabular] Features count: {X_tr.shape[1]} (Target: {y_tr.name})")
        print(f"  [Tabular] Date bounds: Train [{df_tr.index.min()} to {df_tr.index.max()}]")
        print(f"                         Test  [{df_te.index.min()} to {df_te.index.max()}]")
        
        assert len(X_tr) == 35064, f"Expected 35,064 train rows, got {len(X_tr)}"
        assert len(X_te) == 8760, f"Expected 8,760 test rows, got {len(X_te)}"
        assert X_tr.shape[1] == 27, f"Expected 27 features (29 minus ghi and dni), got {X_tr.shape[1]}"
        
        # 2. Sequential Split (Scaled for BiLSTM)
        X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, _, _ = get_sequence_splits(hub, scaled=True, lookback=24)
        print(f"  [Sequence] BiLSTM Train Tensor: {X_tr_seq.shape} | Targets: {y_tr_seq.shape}")
        print(f"  [Sequence] BiLSTM Test Tensor:  {X_te_seq.shape} | Targets: {y_te_seq.shape}")
        
        expected_tr_samples = 35064 - 24
        expected_te_samples = 8760 - 24
        assert X_tr_seq.shape == (expected_tr_samples, 24, 27)
        assert X_te_seq.shape == (expected_te_samples, 24, 27)
    
    # 3. Verify Seasonal Folds on Bhadla
    print("\n" + "=" * 78)
    print(">>> Verifying 6-Fold Seasonal Expanding-Window Walk-Forward Schedule")
    print("=" * 78)
    df_bhadla = load_hub_data("bhadla", scaled=True)
    df_train, _ = train_test_split_timeseries(df_bhadla)
    folds = get_seasonal_expanding_folds(df_train)
    
    for f in folds:
        print(f"  {f.name}:")
        print(f"    Train: {f.train_start.strftime('%Y-%m-%d')} to {f.train_end.strftime('%Y-%m-%d')} ({f.train_size:,} hrs)")
        print(f"    Val:   {f.val_start.strftime('%Y-%m-%d')} to {f.val_end.strftime('%Y-%m-%d')} ({f.val_size:,} hrs)")
        assert f.train_size > 0 and f.val_size > 0
        assert f.train_end < f.val_start, "Temporal overlap between train and validation!"
        
    print("\n[SUCCESS] Subphase 3.1: All cross-validation and splitting checks passed!")

