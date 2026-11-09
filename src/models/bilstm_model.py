# H2-Grid India: Tier 2 Deep Learning — Bidirectional LSTM (BiLSTM) Forecasting Engine
# Implements a 2-layer BiLSTM network with sequence memory and dropout regularization.
# Trains on 4-year sliding sequences (2019-2022) and evaluates on 2023 holdout test set.
# Employs early stopping, checkpointing, and un-scaling to real physical solar units (W/m^2).

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.cross_val import HUBS, get_sequence_splits

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@dataclass
class BiLSTMMetrics:
    """Evaluation metrics for the BiLSTM neural network."""
    hub_id: str
    architecture: str
    r2_overall: float
    rmse_overall: float
    mae_overall: float
    r2_daytime: float
    rmse_daytime: float
    mae_daytime: float
    best_val_loss: float
    epochs_trained: int
    train_time_sec: float
    eval_time_sec: float
    test_hours: int
    daytime_hours: int


class BiLSTMForecaster(nn.Module):
    """
    2-Layer Bidirectional LSTM network with sequence feature projection.
    
    Processes (batch_size, seq_len=24, num_features=27) sequences in both forward
    and backward temporal directions, outputting a continuous 1-step solar irradiance forecast.
    """
    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Bidirectional concatenates forward and backward hidden states (hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Use final time-step representation for 1-step ahead prediction
        last_step = lstm_out[:, -1, :]
        out = self.head(last_step)
        return out.squeeze(-1)


def train_bilstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 12,
    learning_rate: float = 0.002,
    patience: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, float, int, float]:
    """
    Trains the BiLSTM network with early stopping and best-weight checkpointing.
    
    Returns:
        (best_model, best_val_loss, epochs_trained, total_train_time)
    """
    t0 = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    actual_epochs = 0

    for epoch in range(1, num_epochs + 1):
        actual_epochs = epoch
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                val_loss += criterion(preds, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    train_time = time.time() - t0
    return model, best_loss, actual_epochs, train_time


def predict_and_unscale(
    model: nn.Module,
    X_test_seq: np.ndarray,
    hub_id: str,
    solar_elevation: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, float]:
    """
    Generates BiLSTM predictions, un-scales them to W/m^2, and enforces nocturnal clipping.
    """
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        y_pred_scaled = model(x_tensor).cpu().numpy()

    # Load scaler bounds to invert normalization
    scaler_path = PROCESSED_DIR / f"{hub_id}_scaler.joblib"
    scaler = joblib.load(scaler_path)
    ghi_min = float(scaler.data_min_[0])
    ghi_max = float(scaler.data_max_[0])

    # Invert min-max scaling back to physical W/m^2
    y_pred_unscaled = np.clip(y_pred_scaled * (ghi_max - ghi_min) + ghi_min, 0.0, None)

    # Nocturnal zero-clipping when sun is below horizon
    if solar_elevation is not None:
        night_mask = solar_elevation <= 0.0
        y_pred_unscaled[night_mask] = 0.0

    eval_time = time.time() - t0
    return y_pred_unscaled, eval_time


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    daytime_threshold: float = 5.0,
) -> Dict[str, float]:
    """Computes overall and daytime regression metrics."""
    rmse_overall = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_overall = float(mean_absolute_error(y_true, y_pred))
    r2_overall = float(r2_score(y_true, y_pred))

    day_mask = y_true > daytime_threshold
    if day_mask.sum() > 0:
        rmse_day = float(np.sqrt(mean_squared_error(y_true[day_mask], y_pred[day_mask])))
        mae_day = float(mean_absolute_error(y_true[day_mask], y_pred[day_mask]))
        r2_day = float(r2_score(y_true[day_mask], y_pred[day_mask]))
    else:
        rmse_day, mae_day, r2_day = rmse_overall, mae_overall, r2_overall

    return {
        "rmse_overall": round(rmse_overall, 2),
        "mae_overall": round(mae_overall, 2),
        "r2_overall": round(r2_overall, 4),
        "rmse_daytime": round(rmse_day, 2),
        "mae_daytime": round(mae_day, 2),
        "r2_daytime": round(r2_day, 4),
        "test_hours": int(len(y_true)),
        "daytime_hours": int(day_mask.sum()),
    }


def run_bilstm_benchmark(
    hub_id: str,
    val_hours: int = 4000,
    num_epochs: int = 12,
    batch_size: int = 256,
    save_results: bool = True,
) -> Tuple[BiLSTMMetrics, np.ndarray]:
    """
    Runs the complete BiLSTM benchmark pipeline for a given renewable hub.
    """
    hub_id = hub_id.lower().strip()
    X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, df_tr, df_te = get_sequence_splits(hub_id, scaled=True, lookback=24)

    # Train/Validation split from 4-year sequence baseline
    X_train, y_train = X_tr_seq[:-val_hours], y_tr_seq[:-val_hours]
    X_val, y_val = X_tr_seq[-val_hours:], y_tr_seq[-val_hours:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize model
    model = BiLSTMForecaster(input_dim=X_tr_seq.shape[2], hidden_dim=64, num_layers=2, dropout=0.2)
    model, best_val_loss, epochs_trained, train_time = train_bilstm(
        model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=0.002, patience=3
    )

    # Predict on unseen 2023 holdout test set (8,736 hours)
    elev_test = df_te["solar_elevation"].iloc[24:].to_numpy()
    y_pred_unscaled, eval_time = predict_and_unscale(model, X_te_seq, hub_id, solar_elevation=elev_test)

    # Unscale ground truth
    scaler = joblib.load(PROCESSED_DIR / f"{hub_id}_scaler.joblib")
    ghi_min, ghi_max = float(scaler.data_min_[0]), float(scaler.data_max_[0])
    y_true_unscaled = y_te_seq * (ghi_max - ghi_min) + ghi_min

    scores = evaluate_regression(y_true_unscaled, y_pred_unscaled)

    metrics = BiLSTMMetrics(
        hub_id=hub_id,
        architecture="2-Layer BiLSTM (64 hidden, dropout=0.2)",
        r2_overall=scores["r2_overall"],
        rmse_overall=scores["rmse_overall"],
        mae_overall=scores["mae_overall"],
        r2_daytime=scores["r2_daytime"],
        rmse_daytime=scores["rmse_daytime"],
        mae_daytime=scores["mae_daytime"],
        best_val_loss=round(float(best_val_loss), 6),
        epochs_trained=epochs_trained,
        train_time_sec=round(train_time, 2),
        eval_time_sec=round(eval_time, 2),
        test_hours=scores["test_hours"],
        daytime_hours=scores["daytime_hours"],
    )

    if save_results:
        # Save PyTorch weights
        model_path = PROCESSED_DIR / f"{hub_id}_bilstm.pt"
        torch.save(model.state_dict(), model_path)

        # Save metrics
        metrics_file = PROCESSED_DIR / f"{hub_id}_bilstm_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

        # Save predictions
        preds_df = pd.DataFrame(
            {"y_true": y_true_unscaled, "y_pred_bilstm": y_pred_unscaled},
            index=df_te.index[24:],
        )
        preds_file = PROCESSED_DIR / f"{hub_id}_bilstm_preds.parquet"
        preds_df.to_parquet(preds_file)

    return metrics, y_pred_unscaled


def benchmark_all_hubs() -> Dict[str, Dict]:
    """Runs the Tier 2 BiLSTM benchmark across all 4 national renewable hubs."""
    print("\n" + "=" * 80)
    print("H2-Grid India: Tier 2 Deep Learning (BiLSTM) Benchmark")
    print("=" * 80)

    all_metrics = {}
    for hub in HUBS:
        print(f"\n>>> Training & Benchmarking BiLSTM on Hub: {hub.upper()}...")
        metrics, _ = run_bilstm_benchmark(hub, save_results=True)
        all_metrics[hub] = asdict(metrics)
        print(f"    Trained {metrics.epochs_trained} epochs in {metrics.train_time_sec:.1f}s | Val Loss: {metrics.best_val_loss:.5f}")
        print(f"    Overall:  R2 = {metrics.r2_overall:.4f} | RMSE = {metrics.rmse_overall:.2f} W/m2 | MAE = {metrics.mae_overall:.2f} W/m2")
        print(f"    Daytime:  R2 = {metrics.r2_daytime:.4f} | RMSE = {metrics.rmse_daytime:.2f} W/m2 | MAE = {metrics.mae_daytime:.2f} W/m2")

    summary_path = PROCESSED_DIR / "bilstm_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print(f"{'Hub':<16} | {'Overall R2':<10} | {'RMSE (W/m2)':<11} | {'Daytime R2':<10} | {'Day RMSE':<10} | {'Train Time':<10}")
    print("-" * 80)
    for hub, m in all_metrics.items():
        print(
            f"{hub.capitalize():<16} | {m['r2_overall']:<10.4f} | {m['rmse_overall']:<11.2f} | "
            f"{m['r2_daytime']:<10.4f} | {m['rmse_daytime']:<10.2f} | {m['train_time_sec']:<8.1f}s"
        )
    print("=" * 80)
    print(f"[SUCCESS] Saved BiLSTM benchmark summary -> {summary_path}")
    return all_metrics


if __name__ == "__main__":
    benchmark_all_hubs()

