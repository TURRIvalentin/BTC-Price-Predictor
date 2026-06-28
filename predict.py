"""
predict.py — Monte Carlo Dropout inference for BTCPredictor.

Date convention: future dates use CALENDAR days (freq="D"), not business days.
BTC trades 24/7 with no market closures, so excluding weekends would create
gaps that don't reflect actual price exposure.

Reconversion pipeline (order is load-bearing — do not rearrange):
    [model output]  (n_mc, 28)  scaled log returns
         |
    (1) inverse_transform         → raw log returns     (n_mc, 28)
         |
    (2) exp(cumsum(..., axis=1))  → price relatives     (n_mc, 28)
         |
    (3) * last_close              → absolute prices     (n_mc, 28)
         |
    (4) mean / percentile(10,90)  → forecast stats      (28,) each

Stats are always computed ACROSS trajectories for each day, never across
days or on the intermediate log-return arrays.
"""

import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from data import get_clean_data
from model import BTCPredictor, WINDOW, HORIZON

MODELS_DIR = "models"
MC_SAMPLES = 100


# ── Artifact loading ───────────────────────────────────────────────────────────

def _load_artifacts(
    models_dir: str = MODELS_DIR,
) -> tuple[BTCPredictor, object, dict]:
    """
    Load model weights, scaler, and hyperparameters from `models_dir`.
    Raises FileNotFoundError with a clear message if any artifact is missing.
    """
    paths = {
        "hparams": os.path.join(models_dir, "hparams.json"),
        "weights": os.path.join(models_dir, "btc_predictor.pt"),
        "scaler":  os.path.join(models_dir, "scaler.pkl"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts: {missing}\n"
            "Run `python train.py` to generate them first."
        )

    with open(paths["hparams"]) as f:
        hparams = json.load(f)

    model = BTCPredictor(
        input_size=hparams["input_size"],
        hidden_size=hparams["hidden_size"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
        horizon=hparams["horizon"],
    )
    model.load_state_dict(
        torch.load(paths["weights"], map_location="cpu", weights_only=True)
    )

    scaler = joblib.load(paths["scaler"])

    return model, scaler, hparams


# ── Input preparation ──────────────────────────────────────────────────────────

def _build_input_tensor(
    log_returns: np.ndarray,
    scaler,
    window: int = WINDOW,
) -> torch.Tensor:
    """
    Take the last `window` log returns, apply the pre-fitted scaler, and
    return a (1, window, 1) tensor.

    Uses .transform() exclusively — the scaler is NEVER re-fitted here.
    Refitting would use future data to define the scale, which is leakage.
    """
    if len(log_returns) < window:
        raise ValueError(
            f"Need at least {window} log returns; got {len(log_returns)}."
        )
    tail   = log_returns[-window:]                           # (window,)
    scaled = scaler.transform(tail.reshape(-1, 1)).ravel()  # (window,)  — transform only
    # Shape: (batch=1, window, input_size=1)
    return torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)


# ── Monte Carlo inference ──────────────────────────────────────────────────────

def _mc_forward(
    model: BTCPredictor,
    x: torch.Tensor,
    n_samples: int = MC_SAMPLES,
) -> np.ndarray:
    """
    Run `n_samples` stochastic forward passes with dropout active.

    enable_mc_dropout() sets the network to eval() (disabling inter-LSTM
    dropout and BatchNorm randomness) while keeping nn.Dropout in train()
    so every pass produces a different stochastic sample.

    torch.no_grad() is orthogonal: it suppresses gradient tracking for
    speed and memory, but does not affect dropout behaviour.

    Returns (n_samples, horizon) array of SCALED log-return predictions.
    """
    model.enable_mc_dropout()
    with torch.no_grad():
        # Each call: (1, horizon) — concatenate along batch dim.
        samples = torch.cat(
            [model(x) for _ in range(n_samples)], dim=0
        )  # (n_samples, horizon)
    return samples.cpu().numpy()


# ── Reconversion ───────────────────────────────────────────────────────────────

def _reconvert_to_prices(
    mc_scaled: np.ndarray,
    scaler,
    last_close: float,
) -> np.ndarray:
    """
    Convert (n_samples, horizon) SCALED log returns to (n_samples, horizon)
    absolute prices. Steps must execute in this order:

        (1) inverse_transform: undo StandardScaler → raw log returns
        (2) cumsum along time axis → cumulative log return per day
        (3) exp() → price relative to the last known close
        (4) multiply by last_close → absolute USD prices

    Aggregation (mean, percentiles) is intentionally NOT done here — it
    happens in predict() on the price matrix, not on returns.
    """
    n_samples, horizon = mc_scaled.shape

    # (1) De-normalise: scaler expects (N, 1); flatten, transform, reshape back.
    mc_log_returns = scaler.inverse_transform(
        mc_scaled.reshape(-1, 1)                 # (n_samples * horizon, 1)
    ).reshape(n_samples, horizon)                # (n_samples, horizon)

    # (2) + (3) + (4): price_t = last_close * exp(r_1 + r_2 + ... + r_t)
    mc_prices = last_close * np.exp(
        np.cumsum(mc_log_returns, axis=1)        # sum along time, not samples
    )                                            # (n_samples, horizon)

    return mc_prices


# ── Public API ─────────────────────────────────────────────────────────────────

def predict(
    horizon_days: int = HORIZON,
    models_dir: str = MODELS_DIR,
    n_mc_samples: int = MC_SAMPLES,
    source: str = "csv",
) -> pd.DataFrame:
    """
    Run MC Dropout inference and return a forecast DataFrame.

    Parameters
    ----------
    horizon_days : int
        Days to forecast; must satisfy 1 <= horizon_days <= HORIZON (28).
        Slice values: 7, 14, 21, 28.
    models_dir : str
        Directory containing btc_predictor.pt, scaler.pkl, hparams.json.
    n_mc_samples : int
        Number of stochastic forward passes (default 100).
    source : {"csv", "snowflake"}
        Data source for historical prices (passed through to get_clean_data).

    Returns
    -------
    pd.DataFrame with DatetimeIndex (calendar days) and columns:
        mean  — mean predicted price across MC trajectories
        p10   — 10th-percentile price (lower uncertainty bound)
        p90   — 90th-percentile price (upper uncertainty bound)

    Index starts at last_historical_date + 1 day.
    """
    if not (1 <= horizon_days <= HORIZON):
        raise ValueError(
            f"horizon_days must be between 1 and {HORIZON}; got {horizon_days}."
        )

    # ── Load ──────────────────────────────────────────────────────────────────
    model, scaler, hparams = _load_artifacts(models_dir)
    window = hparams["window"]

    # ── Data ──────────────────────────────────────────────────────────────────
    df         = get_clean_data(source=source)
    log_returns = df["Log_Return"].to_numpy()
    last_close  = float(df["Close"].iloc[-1])
    last_date   = df.index[-1]

    # ── Build input ───────────────────────────────────────────────────────────
    x = _build_input_tensor(log_returns, scaler, window)   # (1, window, 1)

    # ── MC inference → (n_mc, 28) scaled log returns ──────────────────────────
    mc_scaled = _mc_forward(model, x, n_samples=n_mc_samples)

    # ── Reconvert → (n_mc, 28) absolute prices ────────────────────────────────
    mc_prices = _reconvert_to_prices(mc_scaled, scaler, last_close)

    # Clip to the requested horizon (axis 1 = days). Stats are computed on
    # axis 0 (trajectories), so these two operations are order-independent —
    # clipping first is a code-clarity choice, not a mathematical requirement.
    mc_prices = mc_prices[:, :horizon_days]                # (n_mc, horizon_days)

    # ── Aggregate over trajectories (axis=0), one stat per future day ─────────
    mean_price = mc_prices.mean(axis=0)                    # (horizon_days,)
    p10        = np.percentile(mc_prices, 10, axis=0)      # (horizon_days,)
    p90        = np.percentile(mc_prices, 90, axis=0)      # (horizon_days,)

    # ── Future dates (calendar days — BTC trades 24/7) ────────────────────────
    future_dates = pd.date_range(
        start   = last_date + pd.Timedelta(days=1),
        periods = horizon_days,
        freq    = "D",
    )

    return pd.DataFrame(
        {"mean": mean_price, "p10": p10, "p90": p90},
        index=future_dates,
    )


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("Loading latest BTC data...")
    df_hist = get_clean_data()
    last_close = float(df_hist["Close"].iloc[-1])
    last_date  = df_hist.index[-1].date()
    print(f"  Last close : ${last_close:,.2f}  ({last_date})")
    print(f"  MC samples : {MC_SAMPLES}\n")

    print("Running 7-day forecast...")
    forecast = predict(horizon_days=7)

    pd.set_option("display.float_format", "${:,.2f}".format)
    print(forecast.rename(columns={"mean": "Mean", "p10": "P10", "p90": "P90"}).to_string())

    day7 = forecast.iloc[-1]
    pct_change = (day7["mean"] / last_close - 1) * 100
    print(
        f"\n  Day-7 implied move : {pct_change:+.1f}%"
        f"  (P10 ${day7['p10']:,.0f}  |  mean ${day7['mean']:,.0f}"
        f"  |  P90 ${day7['p90']:,.0f})"
    )
