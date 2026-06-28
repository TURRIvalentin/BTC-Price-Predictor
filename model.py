"""
model.py — LSTM predictor and dataset utilities for BTC log-return forecasting.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

WINDOW  = 60   # input sequence length (trading days)
HORIZON = 28   # forecast horizon (direct multi-output, one output per day)


# ── Dataset ────────────────────────────────────────────────────────────────────

class BTCDataset(Dataset):
    """
    Sliding-window dataset built from an array of (scaled) log-returns.

    Accepts either:
        log_returns : (N,)    — single feature (log return only)
        log_returns : (N, F)  — multiple features (log return + extras)

    Each sample:
        X : (window, input_size)  — past feature sequence
        y : (horizon,)            — next `horizon` values of column 0 (log return)

    The attribute `self.input_size` reflects the actual feature count and is
    intended to be passed directly to BTCPredictor(input_size=...) so the
    feature dimension is never implicit or buried in the array shape.
    """

    def __init__(
        self,
        log_returns: np.ndarray,
        window: int = WINDOW,
        horizon: int = HORIZON,
    ) -> None:
        # Normalise to 2-D: (N, n_features)
        arr = log_returns.reshape(-1, 1) if log_returns.ndim == 1 else log_returns
        n, self.input_size = arr.shape

        if n < window + horizon:
            raise ValueError(
                f"Series length {n} is too short for "
                f"window={window} + horizon={horizon} (need >= {window + horizon})."
            )

        X_list, y_list = [], []
        for i in range(n - window - horizon + 1):
            X_list.append(arr[i : i + window, :])                        # (window, input_size)
            y_list.append(arr[i + window : i + window + horizon, 0])     # target = col 0

        self.X = torch.tensor(np.stack(X_list), dtype=torch.float32)    # (N, window, input_size)
        self.y = torch.tensor(np.stack(y_list), dtype=torch.float32)    # (N, horizon)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── Model ──────────────────────────────────────────────────────────────────────

class BTCPredictor(nn.Module):
    """
    Stacked LSTM that directly predicts `horizon` future daily log-returns from
    a `window`-length input sequence (direct multi-output forecasting).

    Dropout operates at two levels:
      1. Between LSTM layers   — via the built-in `dropout` parameter of nn.LSTM.
                                 Has no effect when num_layers == 1.
      2. On the last hidden state before the linear head — via an explicit
         nn.Dropout layer (`self.drop`).  This layer is the one kept active
         during Monte Carlo Dropout inference; see `enable_mc_dropout()`.

    Parameters
    ----------
    input_size : int
        Number of input features per time step. Must match the `input_size`
        attribute of the BTCDataset used for training.
    """

    def __init__(
        self,
        input_size:  int   = 1,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        horizon:     int   = HORIZON,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Explicit dropout on the final hidden state — kept active in MC mode.
        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, window, input_size)
        Returns:
            (batch, horizon) — predicted log-returns for the next `horizon` days
        """
        lstm_out, _  = self.lstm(x)         # (batch, window, hidden_size)
        last_hidden  = lstm_out[:, -1, :]   # (batch, hidden_size)
        last_hidden  = self.drop(last_hidden)
        return self.head(last_hidden)       # (batch, horizon)

    def enable_mc_dropout(self) -> None:
        """
        Switch to Monte Carlo Dropout inference mode.

        Puts the whole network in eval() (so BatchNorm uses running statistics
        and the inter-LSTM dropout is disabled) then flips every nn.Dropout
        sub-module back to train() so stochastic dropout fires on each
        forward pass.  Call this once before running MC sampling loops.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


# ── Data-loading + normalisation ───────────────────────────────────────────────

def build_dataloaders(
    log_returns: np.ndarray,
    window:      int   = WINDOW,
    horizon:     int   = HORIZON,
    val_ratio:   float = 0.15,
    batch_size:  int   = 64,
    num_workers: int   = 0,
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Scale inputs and create a time-ordered train/val split.

    Normalisation policy (no data leakage):
        - StandardScaler is fitted on the training slice ONLY.
        - The same fitted scaler is then applied to the validation slice.
        - The scaler is returned so it can be saved alongside the model
          and used in predict.py to inverse-transform predictions.

    Returns
    -------
    train_loader, val_loader, scaler
    """
    split = int(len(log_returns) * (1 - val_ratio))

    train_raw = log_returns[:split]
    val_raw   = log_returns[split:]

    # Fit on train, transform both — never the other way around.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).ravel()
    val_scaled   = scaler.transform(val_raw.reshape(-1, 1)).ravel()

    train_ds = BTCDataset(train_scaled, window=window, horizon=horizon)
    val_ds   = BTCDataset(val_scaled,   window=window, horizon=horizon)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, scaler


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    fake = rng.normal(0.0, 0.02, size=600)

    train_dl, val_dl, scaler = build_dataloaders(fake)
    X_batch, y_batch = next(iter(train_dl))
    print(f"X batch      : {tuple(X_batch.shape)}")    # (64, 60, 1)
    print(f"y batch      : {tuple(y_batch.shape)}")    # (64, 28)
    print(f"input_size   : {train_dl.dataset.input_size}")

    model = BTCPredictor(input_size=train_dl.dataset.input_size)
    print(f"\nModel:\n{model}")

    preds = model(X_batch)
    print(f"\nOutput (train mode) : {tuple(preds.shape)}")   # (64, 28)

    model.enable_mc_dropout()
    mc = torch.stack([model(X_batch) for _ in range(10)])
    print(f"MC samples          : {tuple(mc.shape)}")        # (10, 64, 28)
