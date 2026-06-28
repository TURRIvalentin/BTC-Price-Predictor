"""
train.py — training loop with early stopping for BTCPredictor.

Usage:
    python train.py
    python train.py --hidden-size 256 --num-layers 3 --patience 30
    python train.py --source snowflake
"""

import argparse
import json
import os

import joblib
import torch
import torch.nn as nn
from torch.optim import Adam

from data import get_clean_data
from model import BTCPredictor, build_dataloaders, WINDOW, HORIZON

# ── Default hyperparameters ────────────────────────────────────────────────────

HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 64
MAX_EPOCHS   = 200
PATIENCE     = 20
VAL_RATIO    = 0.15
MODELS_DIR   = "models"


# ── Checkpoint helper ──────────────────────────────────────────────────────────

def _save_checkpoint(
    model: BTCPredictor,
    scaler,
    hparams: dict,
    models_dir: str,
) -> None:
    """Persist model weights, scaler, and hyperparameters to `models_dir`."""
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "btc_predictor.pt"))
    joblib.dump(scaler,            os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    hidden_size:  int   = HIDDEN_SIZE,
    num_layers:   int   = NUM_LAYERS,
    dropout:      float = DROPOUT,
    lr:           float = LR,
    weight_decay: float = WEIGHT_DECAY,
    batch_size:   int   = BATCH_SIZE,
    max_epochs:   int   = MAX_EPOCHS,
    patience:     int   = PATIENCE,
    val_ratio:    float = VAL_RATIO,
    models_dir:   str   = MODELS_DIR,
    source:       str   = "csv",
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = get_clean_data(source=source)
    log_returns = df["Log_Return"].to_numpy()
    print(
        f"  {len(log_returns)} observations  "
        f"({df.index[0].date()} -> {df.index[-1].date()})"
    )

    train_loader, val_loader, scaler = build_dataloaders(
        log_returns,
        window=WINDOW,
        horizon=HORIZON,
        val_ratio=val_ratio,
        batch_size=batch_size,
    )
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"  Train samples: {n_train}  |  Val samples: {n_val}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # input_size comes from the dataset so it is always consistent with the
    # actual feature array — never hardcoded independently.
    input_size = train_loader.dataset.input_size
    model = BTCPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=HORIZON,
    ).to(device)

    hparams = {
        "input_size":  input_size,
        "hidden_size": hidden_size,
        "num_layers":  num_layers,
        "dropout":     dropout,
        "horizon":     HORIZON,
        "window":      WINDOW,
    }
    print(f"  Params: {hparams}\n")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0

    header = f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  Status"
    print(header)
    print("-" * len(header))

    for epoch in range(1, max_epochs + 1):

        # Train
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            # Gradient clipping prevents exploding gradients common in LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= n_train

        # Validate (no dropout, no gradients)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item() * len(X)
        val_loss /= n_val

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            _save_checkpoint(model, scaler, hparams, models_dir)
            status = "[saved]"
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{patience}"

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {status}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nBest val loss : {best_val_loss:.6f}")
    print(f"Artifacts     : {models_dir}/btc_predictor.pt  |  scaler.pkl  |  hparams.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BTCPredictor LSTM")
    p.add_argument("--hidden-size",  type=int,   default=HIDDEN_SIZE,  help="LSTM hidden units")
    p.add_argument("--num-layers",   type=int,   default=NUM_LAYERS,   help="LSTM stacked layers")
    p.add_argument("--dropout",      type=float, default=DROPOUT,      help="Dropout probability")
    p.add_argument("--lr",           type=float, default=LR,           help="Adam learning rate")
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Adam L2 regularisation")
    p.add_argument("--batch-size",   type=int,   default=BATCH_SIZE,   help="Mini-batch size")
    p.add_argument("--max-epochs",   type=int,   default=MAX_EPOCHS,   help="Training epoch cap")
    p.add_argument("--patience",     type=int,   default=PATIENCE,     help="Early-stopping patience")
    p.add_argument("--val-ratio",    type=float, default=VAL_RATIO,    help="Validation fraction")
    p.add_argument("--models-dir",   type=str,   default=MODELS_DIR,   help="Output directory")
    p.add_argument("--source",       type=str,   default="csv",
                   choices=["csv", "snowflake"],                        help="Data source")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_ratio=args.val_ratio,
        models_dir=args.models_dir,
        source=args.source,
    )
