# PROJECT_CONTEXT.md

> Permanent technical context document for the **BTC Price Predictor** repository.
> Written so that an LLM assistant with no prior exposure to this codebase can answer
> questions about it, reason about changes, and understand design intent without
> re-reading the source. Everything below was derived directly from the code and
> configuration files in this repository as of the analysis date; nothing is invented.

---

## 1. Project Overview

**What the project does.** This repository implements an end-to-end machine learning
pipeline that forecasts future Bitcoin (BTC-USD) prices. It downloads historical daily
OHLCV (Open/High/Low/Close/Volume) data, trains a stacked LSTM neural network (in
PyTorch) to predict future **daily log returns** over a horizon of up to 28 days, and
serves the result through an interactive Streamlit web dashboard with a Plotly chart.
Uncertainty around each prediction is quantified using **Monte Carlo (MC) Dropout**,
producing a P10–P90 confidence band around the mean forecast, displayed as a shaded
region on the chart.

**Main objective.** The stated objective (from the README) is explicitly **not** to
build a trading tool or provide investment advice. It is "an end-to-end ML engineering
exercise covering data ingestion, sequence modeling, uncertainty quantification, and
interactive deployment." It functions as a portfolio/demonstration project for AI/ML
engineering skills — the repository is literally located in a folder named
`AI Engineer Portfolio`.

**Main users.** Portfolio reviewers, recruiters, or other engineers evaluating the
author's ML engineering capability, and the author themself using it as a template or
live demo. There is a public live deployment linked from the README running on AWS ECS
Fargate. It is a single-user, unauthenticated, read-only dashboard — there are no user
accounts, no multi-tenant behavior, and no persistence of user interactions.

**Current development stage.** Mature/complete for its scope: training pipeline,
inference pipeline, dashboard, Docker containerization, and CI/CD (GitHub Actions →
AWS ECR/ECS) are all implemented and working. Trained model artifacts
(`models/btc_predictor.pt`, `scaler.pkl`, `hparams.json`) are already present in the
repo working directory (though `models/` and `data/` are `.gitignore`d, so they are
locally generated/cached, not committed to git — except that the Docker build
explicitly does ship them, see §16). There is no automated test suite (no `tests/`
directory, no pytest/unittest files) — verification is done via `if __name__ ==
"__main__"` sanity-check blocks in `data.py`, `model.py`, and `predict.py`.

---

## 2. High-Level Architecture

The system is a **linear, single-process ML pipeline** rather than a distributed or
microservice architecture. It has two independent execution modes that share the same
core modules:

1. **Offline training mode** (`train.py`, run manually / on demand): loads data,
   builds a supervised sliding-window dataset, trains an LSTM with early stopping, and
   persists artifacts (weights, scaler, hyperparameters) to `models/`.
2. **Online serving mode** (`app.py`, run via `streamlit run`): loads cached/fresh
   historical data, loads the persisted model artifacts, runs MC Dropout inference,
   and renders an interactive chart plus a metrics summary and data table.

**Main components:**
- `data.py` — data acquisition/caching/cleaning layer (Yahoo Finance or Snowflake).
- `model.py` — model architecture (`BTCPredictor`), dataset windowing (`BTCDataset`),
  and dataloader/scaler construction (`build_dataloaders`).
- `train.py` — training loop, early stopping, checkpointing.
- `predict.py` — inference-time artifact loading, MC Dropout sampling, reconversion of
  scaled log-returns back into absolute USD price percentiles.
- `app.py` — Streamlit UI that composes `data.py` and `predict.py` into a visual
  dashboard.

**Data flow (end to end):**
```
Yahoo Finance API (or Snowflake DW)
        │  yf.download() / SQL query
        ▼
 data.py: _from_yfinance()/_from_snowflake() → raw OHLCV DataFrame
        │  cached to data/btc_ohlcv.csv
        ▼
 data.py: _add_log_returns() → adds Log_Return column, drops leading NaN
        │
        ▼
 get_clean_data() → canonical DataFrame [Close, High, Low, Volume, Log_Return]
        │
        ├──────────────► train.py: build_dataloaders() → BTCDataset (sliding windows)
        │                       │
        │                       ▼
        │                 model.py: BTCPredictor (2-layer LSTM) trained via Adam/MSE
        │                       │
        │                       ▼
        │                 checkpoint saved: btc_predictor.pt, scaler.pkl, hparams.json
        │
        └──────────────► predict.py: loads checkpoint, takes last 60 log returns,
                                scales them, runs 100 stochastic forward passes
                                (MC Dropout), inverse-transforms + cumsum + exp to
                                reconstruct 100 candidate price trajectories,
                                aggregates to mean/P10/P90 per future day
                                       │
                                       ▼
                                app.py: Streamlit + Plotly renders historical prices,
                                forecast mean, and shaded P10–P90 band
```

**How components interact.** All interaction is via direct Python function imports —
there is no network API between components, no message queue, no database beyond a
flat CSV cache and a Snowflake option. `app.py` imports `get_clean_data` from `data.py`
and `predict` from `predict.py`. `predict.py` imports `get_clean_data` from `data.py`
and `BTCPredictor`/`WINDOW`/`HORIZON` from `model.py`. `train.py` imports
`get_clean_data` from `data.py` and `BTCPredictor`/`build_dataloaders`/`WINDOW`/
`HORIZON` from `model.py`. This creates a simple, acyclic dependency graph (formalized
in §19).

---

## 3. Folder Structure

```
RNN BTC Price Predictor/               (repo root)
├── .github/
│   └── workflows/
│       └── deploy.yml                 CI/CD: build Docker image, push to ECR, deploy to ECS on push to main
├── data/
│   └── btc_ohlcv.csv                  Local cache of downloaded OHLCV data (gitignored; present locally)
├── docs/
│   └── screenshot.png                 App screenshot embedded in README
├── models/
│   ├── btc_predictor.pt               Trained PyTorch model weights (state_dict), ~800KB (gitignored; present locally)
│   ├── hparams.json                   Hyperparameters used for the currently trained model (gitignored; present locally)
│   └── scaler.pkl                     Fitted sklearn StandardScaler, joblib-serialized (gitignored; present locally)
├── app.py                             Streamlit dashboard entry point
├── data.py                            Data ingestion, caching, and cleaning module
├── model.py                           Model architecture + dataset + dataloader utilities
├── predict.py                         MC Dropout inference pipeline
├── train.py                           Training script / CLI
├── requirements.txt                   Python dependency pins
├── Dockerfile                         Container build definition for deployment
├── .dockerignore                      Files excluded from the Docker build context
├── .gitignore                         Files excluded from version control
└── README.md                          Project documentation (features, design decisions, usage, deployment)
```

Notes on gitignore/dockerignore interaction (important, non-obvious):
- `.gitignore` excludes `data/`, `models/`, `.env`, caches, and virtual envs from git.
- `.dockerignore` excludes `.git/`, `.gitignore` itself, `__pycache__`, `.env`, `venv`
  variants, `data/`, `Dockerfile`, `.dockerignore`, `README.md`, and `docs/` — but
  **does NOT exclude `models/`**. This is intentional per a comment in the `Dockerfile`:
  the Docker build context includes the local `models/` directory so the built image
  ships with a pre-trained checkpoint and the container can serve predictions
  immediately without running `train.py` inside the container. This means whoever
  builds the Docker image must have already run `train.py` locally (or otherwise
  populated `models/`) before `docker build`.
- `data/` is excluded from the Docker build context, so the container will download
  fresh data from Yahoo Finance on first request (via `data.py`'s cache-miss path)
  rather than shipping a stale CSV.

---

## 4. Important Files

### `data.py`
- **Purpose:** Single source of truth for acquiring, caching, and cleaning BTC-USD
  OHLCV data, from either Yahoo Finance (default) or Snowflake (optional).
- **Responsibilities:** Downloading via `yfinance`; local CSV caching to avoid repeated
  API calls; converting yfinance's period strings to explicit start dates; computing
  log returns; dropping NaNs; asserting no NaN/inf values remain; providing a uniform
  output schema regardless of source; querying Snowflake via `snowflake-connector-python`
  reading credentials only from environment variables (via `python-dotenv`'s
  `load_dotenv()`, imported defensively with a try/except so the dependency is
  effectively optional); graceful fallback from Snowflake to CSV/yfinance with a
  `UserWarning` if credentials or the connector are missing.
- **Interactions:** Imported by `train.py`, `predict.py`, and `app.py`. Its single
  public entry point is `get_clean_data(...)`.

### `model.py`
- **Purpose:** Defines the neural network architecture, the windowed dataset
  abstraction, and helper functions to turn a raw log-return series into
  train/validation PyTorch DataLoaders with proper (leak-free) scaling.
- **Responsibilities:** `BTCDataset` builds sliding windows of length `WINDOW=60` from
  the log-return array, pairing each with the next `HORIZON=28` returns as the
  prediction target. `BTCPredictor` implements a stacked LSTM + dropout + linear head
  with a special `enable_mc_dropout()` mode. `build_dataloaders()` performs a
  time-ordered (non-shuffled-split) train/val split, fits a `StandardScaler` only on
  the training slice, and constructs `DataLoader`s.
- **Interactions:** Imported by `train.py` (for `BTCPredictor`, `build_dataloaders`,
  `WINDOW`, `HORIZON`) and by `predict.py` (for `BTCPredictor`, `WINDOW`, `HORIZON`).

### `train.py`
- **Purpose:** CLI-driven training entry point that produces the three artifacts
  needed for inference.
- **Responsibilities:** Parses CLI hyperparameter overrides via `argparse`; loads data
  via `data.py`; builds dataloaders/scaler via `model.py`; instantiates `BTCPredictor`;
  runs an Adam-optimized MSE training loop with gradient clipping (`max_norm=1.0`);
  implements early stopping (patience-based) and only checkpoints when validation loss
  improves; writes `models/btc_predictor.pt` (state dict), `models/scaler.pkl`
  (joblib-serialized `StandardScaler`), and `models/hparams.json` (architecture +
  window/horizon config) every time a new best validation loss is found.
- **Interactions:** Depends on `data.py` and `model.py`. Produces the artifacts
  consumed by `predict.py`.

### `predict.py`
- **Purpose:** Turns a trained checkpoint into a probabilistic USD price forecast via
  Monte Carlo Dropout.
- **Responsibilities:** Loads `hparams.json`, model weights, and the fitted scaler
  (raises `FileNotFoundError` with an actionable message if any is missing); builds the
  most recent 60-day input window and scales it (transform-only, never re-fit); runs
  `n_mc_samples` (default 100) stochastic forward passes with dropout active via
  `BTCPredictor.enable_mc_dropout()`; reconverts each of the 100 sampled log-return
  trajectories into absolute price trajectories via
  `last_close * exp(cumsum(inverse_transform(scaled_returns)))`; computes mean, P10,
  and P90 **across trajectories** (axis 0) for each future day, only after
  reconversion to price space (not on raw returns); builds a DatetimeIndex using
  calendar days (`freq="D"`) because BTC trades 24/7; exposes the single public
  function `predict(horizon_days, models_dir, n_mc_samples, source)`.
- **Interactions:** Depends on `data.py` (for fresh historical data) and `model.py`
  (for the `BTCPredictor` class and `WINDOW`/`HORIZON` constants). Consumed by
  `app.py`.

### `app.py`
- **Purpose:** The Streamlit web application — the only user-facing surface in the
  project.
- **Responsibilities:** Page configuration (wide layout, ₿ icon); sidebar controls for
  forecast horizon (7/14/21/28-day selectbox) and historical days to display
  (30–365 slider); cached (`st.cache_data(ttl=3600)`) historical data loading with
  error handling (`st.error` + `st.stop()` on failure); calls `predict.predict()` inside
  a spinner, catching `FileNotFoundError` specifically to show an actionable "run
  `python train.py`" message when no model artifacts exist; renders four summary
  metrics (last close, day-N mean with % delta, P10, P90); builds a Plotly figure with
  historical close line (Bitcoin-orange `#F7931A`), a filled P10–P90 uncertainty band
  (semi-transparent blue polygon), a dotted forecast-mean line, thin P10/P90 boundary
  lines, and a vertical dashed line marking the historical/forecast boundary; uses
  `hovermode="x unified"` and `plotly_dark` template; shows a collapsible forecast data
  table via `st.expander`.
- **Interactions:** Only imports from `data.py` and `predict.py`. This is the
  application's entry point when deployed (see §6, §16).

### `requirements.txt`
- Pinned (minimum-version) Python dependencies: `yfinance`, `pandas`, `numpy`, `torch`,
  `scikit-learn`, `joblib`, `snowflake-connector-python`, `python-dotenv`, `streamlit`,
  `plotly`.

### `Dockerfile`
- Base image `python:3.11-slim`; sets `PYTHONDONTWRITEBYTECODE=1` and
  `PYTHONUNBUFFERED=1`; installs dependencies from `requirements.txt` in a separate
  layer for build caching; copies the entire remaining build context (including
  `models/`, per the `.dockerignore` behavior above) into `/app`; exposes port 8501;
  default command runs `streamlit run app.py --server.address=0.0.0.0
  --server.headless=true`.

### `.github/workflows/deploy.yml`
- GitHub Actions workflow, triggered on push to `main`. Uses OIDC
  (`aws-actions/configure-aws-credentials@v5` with `id-token: write` permission and an
  IAM role ARN) to authenticate to AWS without long-lived secrets. Logs into ECR, builds
  and tags the Docker image with both the git SHA and `latest`, pushes both tags to ECR
  repository `btc-predictor`, and then deploys via
  `aws-actions/amazon-ecs-deploy-express-service@v1` to an ECS service named
  `btc-predictor-1f5d` in the `default` cluster, region `us-east-2`, container port
  8501. This is "ECS Express" — a simplified deploy action, not a full Terraform/CDK
  infrastructure-as-code setup; infrastructure (cluster, ALB, task execution roles) is
  assumed pre-provisioned outside this repo.

### `README.md`
- Comprehensive human-facing documentation: feature list, explicit "Design Decisions"
  section (log returns vs. raw prices, direct vs. recursive multi-output forecasting,
  MC Dropout rationale, percentile computation order, interchangeable data source,
  leak-free scaler fitting), a "Limitations" section acknowledging BTC's near-random-walk
  behavior and the project's non-financial-advice status, a stack table, deployment
  description, and getting-started instructions including the Snowflake `.env` template.

### `data/btc_ohlcv.csv`
- Local cache artifact (not committed to git via `.gitignore`, though present in the
  working tree at analysis time). CSV columns: `Date, Open, High, Low, Close, Volume`.
  Currently spans from 2022-06-29 onward (~1461 rows ≈ 4 years), consistent with the
  default `PERIOD = "4y"` in `data.py`.

### `models/hparams.json`
- Currently persisted hyperparameters from the last successful training run:
  `input_size=1, hidden_size=128, num_layers=2, dropout=0.3, horizon=28, window=60`.
  These match `train.py`'s defaults exactly, implying the shipped checkpoint was
  trained with default CLI arguments.

### `models/btc_predictor.pt` / `models/scaler.pkl`
- Binary artifacts: PyTorch `state_dict` (~800 KB) and a joblib-pickled
  `sklearn.preprocessing.StandardScaler` (~0.6 KB, single-feature scaler fitted on
  training-slice log returns).

---

## 5. Technologies

| Category | Technology | Role |
|---|---|---|
| Language | Python 3.11 | Entire codebase |
| Deep learning | PyTorch (`torch>=2.2`) | LSTM model definition, training, MC Dropout inference |
| Data acquisition | `yfinance>=0.2.40` | Yahoo Finance OHLCV download |
| Data acquisition (optional) | `snowflake-connector-python>=3.0` | Alternative data warehouse source |
| Env config | `python-dotenv>=1.0` | Loads `.env` for Snowflake credentials; optional import |
| Data manipulation | `pandas>=2.0`, `numpy>=1.26` | DataFrame handling, numeric ops, log-return math |
| Preprocessing | `scikit-learn>=1.3` (`StandardScaler`) | Feature scaling with leak-free fit/transform discipline |
| Serialization | `joblib>=1.3` | Persisting/loading the fitted scaler |
| Web dashboard | `streamlit>=1.35` | Entire UI framework |
| Charting | `plotly>=5.20` (`graph_objects`) | Interactive historical + forecast chart |
| Containerization | Docker (`python:3.11-slim` base) | Packaging for deployment |
| CI/CD | GitHub Actions | Build/push/deploy automation |
| Cloud | AWS (ECR, ECS Fargate, IAM OIDC role, presumably an ALB) | Hosting the live demo |

No JavaScript/TypeScript, no relational database, no REST/GraphQL API framework (e.g.
FastAPI/Flask) — Streamlit itself is the "server."

---

## 6. Entry Points

There are three distinct executable entry points, each a plain Python script with a
`if __name__ == "__main__":` block — no `setup.py`/`pyproject.toml` console-script
entry points, no `main.py`.

1. **`streamlit run app.py`** — the production/user-facing entry point. This is what
   the `Dockerfile`'s `CMD` runs, and what a developer runs locally to view the
   dashboard. Streamlit executes `app.py` top-to-bottom on every user interaction
   (Streamlit's rerun model), re-evaluating cached functions only when their cache is
   still valid.
2. **`python train.py [--flags]`** — offline training entry point. Not run inside the
   Docker container (there is no training service); expected to be run manually by a
   developer to (re)generate `models/` artifacts before building/deploying the image.
3. **`python data.py`** / **`python model.py`** / **`python predict.py`** — each module
   also has a self-test/demo `__main__` block (printing DataFrame shape/stats, running
   a fake dataloader batch through the model, or running a 7-day sanity forecast
   printed to stdout). These are developer diagnostics, not part of the production
   flow.

**What happens after `streamlit run app.py` starts:** Streamlit sets page config →
renders sidebar controls → calls the cached `_load_history()` (which calls
`get_clean_data()` from `data.py`, hitting the CSV cache or downloading fresh data via
yfinance) → calls `predict()` from `predict.py` (which loads model artifacts from
`models/`, builds the latest input window, runs 100 MC Dropout forward passes,
reconverts to price space) → builds and renders the Plotly figure → renders the
collapsible data table. Any exception in data loading or prediction is caught and shown
as a Streamlit error message, and a missing-model-artifact scenario shows a specific
actionable instruction to run `train.py`.

---

## 7. Application Flow

**Training flow (`python train.py`):**
1. Parse CLI args (or use defaults: hidden_size=128, num_layers=2, dropout=0.3,
   lr=1e-3, weight_decay=1e-5, batch_size=64, max_epochs=200, patience=20,
   val_ratio=0.15).
2. Select device (`cuda` if available, else `cpu`).
3. Load clean OHLCV+Log_Return data via `get_clean_data(source=...)`.
4. Call `build_dataloaders()`: time-order split (85% train / 15% val by default, no
   shuffling across the split boundary), fit `StandardScaler` on train only, transform
   both splits, wrap each in a `BTCDataset` (sliding windows of 60 in, 28 out), wrap in
   `DataLoader`s (train shuffled, val not).
5. Instantiate `BTCPredictor` with `input_size` taken from the dataset (always 1 in the
   current setup, since only `Log_Return` is fed in — see §17 for why this could be
   extended).
6. Train up to `max_epochs`, each epoch: forward pass, MSE loss vs. the 28-day target
   vector, backward pass, gradient-norm clipping to 1.0, optimizer step. Then a no-grad
   validation pass in `eval()` mode.
7. Early stopping: track best validation loss; every time it improves, immediately
   persist a checkpoint (weights + scaler + hparams) — meaning the **saved** checkpoint
   is always the best-validation one, not the final-epoch one. If validation loss fails
   to improve for `patience` consecutive epochs, stop early.
8. Print a per-epoch table to stdout; print a final summary.

**Inference flow (`predict.predict(horizon_days=...)`), invoked from `app.py`:**
1. Validate `1 <= horizon_days <= 28`.
2. Load `hparams.json`, model weights (`torch.load(..., weights_only=True)`), and the
   fitted scaler; raise `FileNotFoundError` if any file is missing.
3. Reconstruct `BTCPredictor` with the exact architecture recorded in `hparams.json`
   (so the served model always matches the checkpoint's true shape, even if
   `model.py`'s defaults have since changed).
4. Fetch current OHLCV data via `get_clean_data(source=...)` (same caching/fallback
   logic as training).
5. Take the last `window` (60) log returns, scale them with `scaler.transform()`
   (never re-fit), shape into a `(1, 60, 1)` tensor.
6. Call `enable_mc_dropout()` on the model (puts it in `eval()` globally, then flips
   every `nn.Dropout` submodule back to `train()` so dropout still fires
   stochastically) and run 100 forward passes under `torch.no_grad()`, producing a
   `(100, 28)` array of scaled log-return predictions.
7. Reconvert to price space per trajectory, in this fixed order: inverse-transform the
   scaler → cumulative sum along the time axis → exponentiate → multiply by the last
   known close price. Result: `(100, 28)` array of absolute USD price trajectories.
8. Slice to the first `horizon_days` columns.
9. Aggregate across the 100 trajectories (axis 0) to get `mean`, `p10` (10th
   percentile), `p90` (90th percentile) for each future day.
10. Build a `pd.DataFrame` indexed by calendar-day dates starting the day after the
    last historical date, columns `mean`, `p10`, `p90`.

**Dashboard rendering flow (`app.py`), continuing from inference:**
1. Compute last close, last date, percentage delta of day-N mean vs. last close.
2. Render four `st.metric` tiles.
3. Prepend the last historical point to every forecast series so the chart lines
   connect seamlessly with no visual gap.
4. Build four/five Plotly traces (historical line, P10–P90 filled polygon, dotted mean
   line, thin P10/P90 boundary lines) plus a vertical marker line at the
   history/forecast boundary.
5. Render with `plotly_dark` template, unified hover, and a horizontal legend above the
   chart.
6. Render a collapsible formatted data table of the forecast.

---

## 8. Core Modules

### Data module (`data.py`)
Acts as an anti-corruption layer between two heterogeneous data sources (Yahoo Finance
REST/yfinance library, and a Snowflake SQL table) and the rest of the pipeline, which
only ever sees one canonical schema: `["Close", "High", "Low", "Volume", "Log_Return"]`
indexed by a `DatetimeIndex` named `Date`. Internally:
- `_period_to_start()` translates period shorthand (e.g., `"4y"`) into an explicit
  start date, because yfinance's `period=` parameter only accepts a fixed enum of
  strings and `"4y"` is not one of them — the code instead uses `start=`.
- `_add_log_returns()` computes `log(Close_t / Close_{t-1})`, drops the first row
  (undefined return), and asserts no NaN/Inf values leak through — a defensive
  correctness check, not exception-driven control flow.
- `_from_yfinance()` implements the CSV cache: if the cache file exists and
  `force_refresh` is False, load from disk; otherwise download and overwrite the cache.
  Handles yfinance's occasional `MultiIndex` column return format (when tickers are
  passed as a list) by flattening to the first level.
- `_from_snowflake()` returns `None` (a sentinel, not an exception) if credentials are
  incomplete or the connector isn't installed, signaling the caller to fall back
  silently (with a warning) rather than crash the pipeline. This is a deliberate design
  choice documented in the README ("Interchangeable data source").
- `get_clean_data()` is the only function other modules should call; it dispatches to
  the right backend, normalizes column dtypes to `float`, and always applies
  `_add_log_returns()` regardless of source.

### Model module (`model.py`)
Defines the ML core:
- `BTCDataset(Dataset)` — a sliding-window supervised-learning framer. Given an
  `(N,)` or `(N, F)` array, produces `X: (window, input_size)` / `y: (horizon,)` pairs
  for every valid starting index. Raises `ValueError` if the series is too short.
  Stores `input_size` as an instance attribute specifically so downstream code (model
  construction) never needs to hardcode or separately infer the feature count.
- `BTCPredictor(nn.Module)` — architecture: `nn.LSTM` (configurable input/hidden
  size/layers/dropout) → take only the **last timestep's** hidden state → explicit
  `nn.Dropout` → `nn.Linear(hidden_size, horizon)` producing all 28 future log-return
  predictions in a single forward pass (direct multi-output, not autoregressive/
  recursive). The inter-layer LSTM dropout (only active when `num_layers > 1`) and the
  final explicit dropout are two independent dropout mechanisms; only the latter is
  toggled on during MC Dropout inference via `enable_mc_dropout()`, which calls
  `self.eval()` first (disabling inter-layer dropout and any batchnorm, though there is
  no batchnorm in this model) and then iterates `self.modules()` to call `.train()` on
  every `nn.Dropout` instance specifically.
- `build_dataloaders()` — orchestrates the leak-free preprocessing pipeline: splits
  the raw (unscaled) log-return array chronologically (no shuffling across the
  boundary, since shuffling before splitting would leak future information into
  training), fits `StandardScaler` on the training portion only, transforms both
  portions with that same fitted scaler, wraps each into a `BTCDataset`, and returns
  `DataLoader`s plus the scaler object itself (so callers can persist it).

### Training module (`train.py`)
A conventional PyTorch training loop with two engineering safeguards worth calling
out: (1) gradient-norm clipping at `max_norm=1.0` on every step, mitigating LSTM
exploding-gradient risk; (2) checkpoint-on-improvement early stopping, meaning the
persisted model is always the historically-best one on validation loss, and training
can be interrupted/resumed conceptually at any point without losing the best result
(though there is no actual resume-from-checkpoint logic implemented — restarting
`train.py` always trains from scratch). CLI argument parsing via `argparse` exposes
every architecture and optimization hyperparameter, but the `WINDOW=60` and
`HORIZON=28` sequence-shape constants are imported from `model.py` and are **not**
CLI-overridable — the underlying model architecture assumptions are treated as fixed,
while capacity (hidden size, depth, dropout) and optimization (lr, batch size,
patience) are treated as tunable.

### Inference module (`predict.py`)
The most mathematically careful module in the codebase, with inline comments and a
docstring explicitly calling out that operation order is "load-bearing." The key
insight documented (and enforced in code) is that reconversion from scaled log-returns
to prices must happen *fully, per-trajectory*, before any percentile aggregation — you
cannot take percentiles of returns and then convert those percentile *returns* into
percentile *prices*, because `exp(cumsum(...))` is a nonlinear transform and
percentile-of-a-nonlinear-function ≠ nonlinear-function-of-a-percentile. This is
implemented by keeping all 100 MC trajectories as full `(100, 28)` matrices all the
way through inverse-scaling, cumulative summation, and exponentiation, and only calling
`np.percentile(..., axis=0)` at the very end.

### Application module (`app.py`)
A single linear Streamlit script (no custom components, no multipage app, no session
state usage beyond Streamlit's own caching). Uses `st.cache_data(ttl=3600)` to avoid
re-downloading/re-parsing historical data on every widget interaction (Streamlit
reruns the whole script on each interaction) — cache expires after one hour. Error
handling is coarse-grained but user-friendly: broad `except Exception` blocks around
both data loading and prediction, converted into `st.error()` messages, with a special
case for `FileNotFoundError` during prediction that gives the exact remediation command
(`python train.py`).

---

## 9. Classes

### `BTCDataset` (in `model.py`)
- **Responsibility:** Convert a flat (or multi-feature) time series of log returns
  into supervised `(X, y)` sliding-window pairs suitable for a `DataLoader`.
- **Attributes:**
  - `input_size: int` — number of features per timestep (inferred from input array
    shape; currently always 1 in this project's actual usage, since only `Log_Return`
    is passed in, but the class supports more).
  - `X: torch.Tensor` of shape `(num_samples, window, input_size)`, dtype float32.
  - `y: torch.Tensor` of shape `(num_samples, horizon)`, dtype float32 — always
    derived from column 0 of the input array regardless of how many features there are.
- **Important methods:**
  - `__init__(log_returns, window=60, horizon=28)` — validates minimum series length
    (`n >= window + horizon`), builds all sliding windows in a Python loop, stacks them
    into tensors.
  - `__len__`, `__getitem__` — standard PyTorch `Dataset` protocol.
- **Relationships:** Instantiated inside `build_dataloaders()`. Its `input_size`
  attribute directly parameterizes `BTCPredictor.__init__`, coupling dataset shape to
  model shape without hardcoding.

### `BTCPredictor(nn.Module)` (in `model.py`)
- **Responsibility:** The forecasting model itself — a stacked LSTM with dropout
  regularization and a linear projection head that emits all `horizon` future
  log-return predictions in one forward pass.
- **Attributes:**
  - `self.lstm: nn.LSTM` — `input_size`, `hidden_size` (default 128), `num_layers`
    (default 2), `dropout` (inter-layer, default 0.3, only active if `num_layers > 1`),
    `batch_first=True`.
  - `self.drop: nn.Dropout` — explicit dropout (default p=0.3) applied to the final
    hidden state before the linear head; this is the layer manipulated by MC Dropout.
  - `self.head: nn.Linear(hidden_size, horizon)` — maps final hidden state to the
    28-length output vector.
- **Important methods:**
  - `forward(x)` — `x: (batch, window, input_size)` → runs LSTM → takes
    `lstm_out[:, -1, :]` (last timestep's hidden state across all layers) → applies
    `self.drop` → applies `self.head` → returns `(batch, horizon)`.
  - `enable_mc_dropout()` — puts the entire model in `eval()` mode, then iterates
    `self.modules()` and calls `.train()` specifically on every `nn.Dropout` instance,
    so only dropout randomness (not inter-layer LSTM dropout, not any batchnorm) is
    active during MC sampling.
- **Relationships:** Constructed identically (same hyperparameters) in `train.py`
  (fresh, randomly initialized) and in `predict.py` (rebuilt from `hparams.json`, then
  `load_state_dict()`-ed from `btc_predictor.pt`). Consumes `BTCDataset`-shaped tensors.
  Its architecture constants (`WINDOW`, `HORIZON`) are module-level globals in
  `model.py` reused by both `train.py` and `predict.py`.

No other custom classes exist in the codebase — `StandardScaler` and `DataLoader` are
used directly from their respective libraries without subclassing.

---

## 10. Functions

Summarized by module (grouping trivial dunder methods with their owning class in §9
above; this section covers standalone/module-level functions).

**`data.py`**
- `_period_to_start(period)` — string period → ISO start-date string; raises
  `ValueError` on unrecognized period.
- `_add_log_returns(df)` — adds `Log_Return` column, drops leading NaN row, asserts
  data cleanliness.
- `_from_yfinance(ticker, period, cache_path, force_refresh)` — download-or-load-from-
  cache OHLCV data; handles yfinance MultiIndex columns; writes cache CSV.
- `_from_snowflake(period)` — credential-gated Snowflake query; returns `None` (not an
  exception) on missing credentials/connector, signaling fallback.
- `get_clean_data(source, ticker, period, cache_path, force_refresh)` — **public API**;
  dispatches to the right backend, normalizes dtypes/columns, always adds log returns.

**`model.py`**
- `build_dataloaders(log_returns, window, horizon, val_ratio, batch_size, num_workers)`
  — chronological split, leak-free scaler fit/transform, returns
  `(train_loader, val_loader, scaler)`.

**`train.py`**
- `_save_checkpoint(model, scaler, hparams, models_dir)` — writes all three artifacts
  to disk (creates `models_dir` if absent).
- `train(hidden_size, num_layers, dropout, lr, weight_decay, batch_size, max_epochs,
  patience, val_ratio, models_dir, source)` — the full training loop described in §7.
- `_parse_args()` — builds the `argparse.ArgumentParser` and returns parsed CLI args.

**`predict.py`**
- `_load_artifacts(models_dir)` — loads and validates presence of hparams/weights/
  scaler; raises `FileNotFoundError` with the exact missing paths and remediation
  instructions if anything is absent.
- `_build_input_tensor(log_returns, scaler, window)` — extracts last `window` returns,
  scales (transform only), shapes into `(1, window, 1)` tensor.
- `_mc_forward(model, x, n_samples)` — runs `n_samples` stochastic forward passes with
  dropout active, returns `(n_samples, horizon)` numpy array of scaled predictions.
- `_reconvert_to_prices(mc_scaled, scaler, last_close)` — the four-step
  inverse-transform → cumsum → exp → multiply pipeline, per trajectory; returns
  `(n_samples, horizon)` numpy array of absolute prices.
- `predict(horizon_days, models_dir, n_mc_samples, source)` — **public API**;
  orchestrates the full inference flow described in §7 and returns the forecast
  DataFrame.

**`app.py`**
- `_load_history()` — `st.cache_data`-wrapped wrapper around `get_clean_data()`, cached
  for one hour.
- No other standalone functions; the rest of `app.py` is top-level Streamlit
  script logic executed sequentially.

---

## 11. APIs

**There is no HTTP/REST/GraphQL API in this project.** All "APIs" are internal Python
function-call interfaces between modules (documented above in §4/§8/§10). The
Streamlit app itself is the only network-facing surface, and it exposes a browser UI
(port 8501), not a programmatic API — there are no JSON endpoints, no request/response
schemas, and no authentication layer of any kind (the deployed app is publicly
accessible with no login, per the live demo URL in the README).

**External API dependency:** the project's *client-side* consumption of the Yahoo
Finance API happens transitively through the `yfinance` Python library
(`yf.download(ticker, start=..., interval="1d", auto_adjust=True, progress=False)`).
This is an unauthenticated public data endpoint; yfinance scrapes/queries Yahoo
Finance's undocumented endpoints internally, so there are no API keys involved for this
path.

**Optional external API/integration:** Snowflake, via `snowflake.connector.connect()`
using credentials read exclusively from environment variables
(`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_WAREHOUSE`,
`SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, optionally `SNOWFLAKE_TABLE` defaulting to
`BTC_OHLCV`). The SQL query executed is a static parameterized-by-string-formatting
`SELECT DATE, HIGH, LOW, CLOSE, VOLUME FROM {table} WHERE TICKER = 'BTC-USD' AND DATE
>= DATEADD(DAY, -{lookback_days}, CURRENT_DATE()) ORDER BY DATE ASC`. Note:
`{table}` is interpolated directly from an environment variable into the SQL string
without parameterization — this is a latent SQL-injection surface *if* an untrusted
party could ever control the `SNOWFLAKE_TABLE` environment variable, though in this
project's deployment model env vars are operator-controlled, not user-controlled, so
practical risk is low but worth flagging if this pattern is ever reused elsewhere.

---

## 12. Database

There is no relational/document database. Persistence is entirely file-based:
- **CSV file** `data/btc_ohlcv.csv` — flat cache of OHLCV rows, columns
  `Date, Open, High, Low, Close, Volume` (note: `Open` is fetched and cached but
  dropped by `get_clean_data()`'s canonical output, which only keeps `Close, High, Low,
  Volume, Log_Return`).
- **Optional Snowflake table** (external, not managed by this repo) — expected schema
  per `data.py`'s docstring: `DATE DATE, TICKER VARCHAR, HIGH FLOAT, LOW FLOAT, CLOSE
  FLOAT, VOLUME FLOAT`, default table name `BTC_OHLCV`. No DDL/migration files exist in
  this repo — the table is assumed to already exist in the target Snowflake account.
- **Model artifacts** (`models/*.pt`, `*.pkl`, `*.json`) function as a simple
  "artifact store" but are just local files, not a database.

No ORM, no SQLAlchemy, no schema migrations tooling.

---

## 13. Machine Learning / AI

This is the heart of the project.

**Task formulation.** Multi-horizon, direct (non-recursive) time-series regression:
given the last 60 daily log returns of BTC-USD, predict the next 28 daily log returns
in a single forward pass. The output is then reconverted to absolute price space for
presentation, but the model itself is trained and evaluated purely in log-return space
via MSE loss.

**Why log returns, not raw prices** (from README, corroborated by code): BTC price is
non-stationary (trending, heteroscedastic). Log returns
`r_t = log(Close_t / Close_{t-1})` are approximately stationary and put the training
signal on a consistent scale across the whole window, rather than forcing the model to
memorize an absolute price range that won't generalize.

**Datasets.** A single feature series (`Log_Return`) derived from ~4 years (1460 days,
`PERIOD="4y"`) of daily BTC-USD OHLCV data from Yahoo Finance (or equivalently shaped
data from Snowflake). No exogenous features (macro data, on-chain metrics, sentiment,
other assets) are currently included — `BTCDataset`/`BTCPredictor` support multi-feature
input (`input_size` is a first-class parameter) but the pipeline as wired today only
ever passes the single `Log_Return` column, i.e., `input_size=1` in practice (confirmed
by `hparams.json`).

**Preprocessing / feature engineering.**
1. Compute log returns from raw close prices; drop the first (NaN) row.
2. Chronological train/validation split (85/15 by default) — no shuffling across the
   split boundary, since this is time-series data and shuffling would leak future
   information.
3. Fit `StandardScaler` (zero mean, unit variance) on the training split's log returns
   only; apply `.transform()` (never re-fit) to the validation split and, at inference
   time, to the latest 60-day input window.
4. Build sliding windows: input `(window=60, features=1)`, target `(horizon=28,)` —
   the target is the *next* 28 log returns after the input window, taken from column 0.

**Model architecture.** Stacked LSTM (`nn.LSTM`, default 2 layers, 128 hidden units,
inter-layer dropout 0.3 when `num_layers>1`) → take the last timestep's hidden state
→ explicit `nn.Dropout(0.3)` → `nn.Linear(128 → 28)`. Total parameters are modest
(checkpoint file is ~800 KB), consistent with a small-to-medium LSTM. No attention
mechanism, no Transformer, no convolutional layers — a straightforward recurrent
architecture chosen for a moderate-size univariate sequence problem.

**Training pipeline.** Adam optimizer (`lr=1e-3`, `weight_decay=1e-5` — L2
regularization), `nn.MSELoss()` against the 28-length target vector, gradient-norm
clipping at 1.0 per step, up to 200 epochs, early stopping with patience 20 epochs
(stop if validation loss doesn't improve for 20 consecutive epochs), checkpoint saved
every time validation loss reaches a new best value (so the final saved checkpoint is
always the best-on-validation model, never a later possibly-overfit one). Batch size
64. No learning-rate scheduler, no k-fold cross-validation, no hyperparameter search
harness (though CLI flags allow manual sweeps).

**Uncertainty quantification: Monte Carlo Dropout.** Instead of training an ensemble of
models (expensive) or a Bayesian neural network directly, the project exploits
dropout-as-approximate-Bayesian-inference (the well-known MC Dropout technique): at
inference, dropout layers are kept stochastic (`train()` mode) while everything else
(inter-layer LSTM dropout, if it existed at eval time, and any batchnorm) is frozen
(`eval()` mode). Running 100 independent forward passes on the identical input yields
100 different output samples purely due to the injected dropout noise; their empirical
spread approximates the model's predictive uncertainty. This is implemented precisely
via `BTCPredictor.enable_mc_dropout()` and `predict._mc_forward()`.

**Inference / reconversion pipeline.** Each of the 100 sampled scaled log-return
trajectories is independently: inverse-scaled (`StandardScaler.inverse_transform`),
cumulatively summed across the 28 future days (`np.cumsum(axis=1)`), exponentiated
(`np.exp`), and multiplied by the last known close price — reconstructing 100 full
candidate price paths. Only after this full reconversion are `mean`, `np.percentile(...,
10)`, and `np.percentile(..., 90)` computed **across the 100 trajectories** (axis=0)
for each future day independently. This ordering is explicitly called out in code
comments as mathematically required: nonlinear transforms (`exp`, `cumsum`) do not
commute with percentile computation, so percentiles must be taken on the final price
trajectories, not on intermediate return-space percentiles.

**Evaluation metrics.** Only MSE (via `nn.MSELoss`) on the validation split, tracked
per epoch and used purely for early-stopping/checkpoint-selection. There is no
holdout/test-set evaluation script, no backtesting harness, no reported directional
accuracy, Sharpe ratio, or calibration metric (e.g., checking whether ~80% of realized
outcomes actually fall within the P10–P90 band). This is an acknowledged limitation
(see §18).

**Known behavioral limitation (explicitly documented in README):** BTC daily returns
are close to a random walk, so the model's predictions tend toward smooth,
mean-reverting forecasts that under-represent real crypto volatility; the uncertainty
band widens with horizon but won't capture sharp real-world jumps. The README is
explicit that this is not intended as a trading signal.

---

## 14. Configuration

**Environment variables** (all optional; only needed for the Snowflake data-source
path):
| Variable | Purpose | Required? |
|---|---|---|
| `SNOWFLAKE_ACCOUNT` | Snowflake account identifier | Only if `source="snowflake"` |
| `SNOWFLAKE_USER` | Snowflake username | Only if `source="snowflake"` |
| `SNOWFLAKE_PASSWORD` | Snowflake password | Only if `source="snowflake"` |
| `SNOWFLAKE_WAREHOUSE` | Snowflake warehouse name | Only if `source="snowflake"` |
| `SNOWFLAKE_DATABASE` | Snowflake database name | Only if `source="snowflake"` |
| `SNOWFLAKE_SCHEMA` | Snowflake schema name | Only if `source="snowflake"` |
| `SNOWFLAKE_TABLE` | Table name to query | Optional, defaults to `BTC_OHLCV` |

These are loaded from a local `.env` file via `python-dotenv`'s `load_dotenv()` (called
at import time in `data.py`, wrapped in a try/except so the app still works if
`python-dotenv` isn't installed — env vars can then be set directly in the shell
instead). `.env` is git-ignored. If **any** of the six required Snowflake variables is
missing, `_from_snowflake()` returns `None` and the caller silently (with a
`UserWarning`) falls back to the yfinance/CSV path — there is no hard failure mode for
missing Snowflake config.

**Configuration files:**
- `models/hparams.json` — runtime model configuration (architecture shape) read by
  `predict.py` at inference time; **not** meant to be hand-edited — it's a training
  output that keeps the reconstructed model architecture in sync with the saved
  weights.
- No `config.yaml`/`settings.py`/`pydantic-settings` — all other configuration is via
  Python function default arguments and CLI flags (`argparse` in `train.py`).

**Secrets expected:** Only the Snowflake credential set above. No API key is needed for
Yahoo Finance (`yfinance` uses an unauthenticated endpoint). AWS credentials for
deployment are handled entirely inside GitHub Actions via OIDC role assumption
(`arn:aws:iam::750907156542:role/github-actions-btc-deploy`) — no AWS access keys are
stored as GitHub secrets in the visible workflow file.

**Runtime configuration (module-level constants, not environment-driven):**
- `data.py`: `TICKER="BTC-USD"`, `PERIOD="4y"`, `CACHE_PATH="data/btc_ohlcv.csv"`.
- `model.py`: `WINDOW=60`, `HORIZON=28`.
- `train.py`: `HIDDEN_SIZE=128, NUM_LAYERS=2, DROPOUT=0.3, LR=1e-3,
  WEIGHT_DECAY=1e-5, BATCH_SIZE=64, MAX_EPOCHS=200, PATIENCE=20, VAL_RATIO=0.15,
  MODELS_DIR="models"` — all overridable via CLI flags.
- `predict.py`: `MODELS_DIR="models"`, `MC_SAMPLES=100`.

---

## 15. Dependencies

| Package | Why it's used |
|---|---|
| `yfinance` | Fetches free historical BTC-USD OHLCV data from Yahoo Finance without needing an API key |
| `pandas` | DataFrame-based data manipulation throughout (`data.py`, `predict.py`, `app.py`) |
| `numpy` | Numeric array operations: log returns, cumulative sums, percentiles, exponentials |
| `torch` | Defines and trains the LSTM (`BTCPredictor`), runs MC Dropout inference |
| `scikit-learn` | `StandardScaler` for leak-free feature normalization |
| `joblib` | Efficient serialization/deserialization of the fitted `StandardScaler` |
| `snowflake-connector-python` | Optional alternative data source for production/enterprise data-warehouse deployments |
| `python-dotenv` | Loads Snowflake credentials from a local `.env` file during development |
| `streamlit` | The entire web dashboard framework — layout, widgets, caching, state |
| `plotly` | Interactive charting library used for the historical + forecast visualization |

All versions are specified as minimums (`>=`) rather than pinned exact versions, so
reproducibility relies on whatever latest-compatible versions are available at install
time — there is no `requirements.lock`, `Pipfile.lock`, or `poetry.lock` in the repo.

---

## 16. Deployment

**Local development / running:**
```bash
pip install -r requirements.txt
python train.py              # generates models/btc_predictor.pt, scaler.pkl, hparams.json
streamlit run app.py         # launches the dashboard on localhost, default Streamlit port 8501
```
Training on first run also triggers a fresh yfinance download (cached thereafter to
`data/btc_ohlcv.csv`).

**Docker:** The `Dockerfile` builds on `python:3.11-slim`, installs
`requirements.txt` in a cache-friendly layer, then `COPY . .` — which, due to the
`.dockerignore` configuration analyzed in §3, **includes** the local `models/`
directory (pre-trained artifacts) but **excludes** `data/`, `.git`, `README.md`,
`docs/`, and env/venv artifacts. This means:
- Anyone building the Docker image must first run `train.py` locally (or otherwise
  populate `models/`) so the image ships with a working checkpoint.
- The container will re-download fresh price data from Yahoo Finance on first
  Streamlit interaction (since `data/` isn't shipped), populating its own internal
  ephemeral cache — this cache does not persist across container restarts/redeploys
  since there's no volume mount defined.
- Exposes port 8501; runs `streamlit run app.py --server.address=0.0.0.0
  --server.headless=true` so it's reachable from outside the container and doesn't try
  to open a local browser or prompt for the Streamlit telemetry opt-in.

**CI/CD (`.github/workflows/deploy.yml`):** Triggered automatically on every push to
`main`. Steps: checkout → assume an AWS IAM role via OIDC (no static AWS keys stored in
GitHub) → log into Amazon ECR → build the Docker image, tag with both the commit SHA
and `latest` → push both tags to the `btc-predictor` ECR repository → deploy to AWS ECS
using the "ECS Express" GitHub Action (`amazon-ecs-deploy-express-service@v1`) against
service `btc-predictor-1f5d` in the `default` cluster, region `us-east-2`, mapping
container port 8501.

**Production infrastructure (per README + workflow):** AWS ECS Fargate (serverless
containers, no EC2 management), fronted by an Application Load Balancer with HTTPS
termination (per README: "served via an Application Load Balancer with HTTPS"). The
README notes the task "scales to zero when idle" — consistent with ECS Express/Fargate
capabilities — and that the whole deployment is "fully reproducible from the
`Dockerfile`." The live demo URL
(`https://bt-7dff3b7ba8f246528d893ace7e964833.ecs.us-east-2.on.aws`) is an
AWS-generated ECS Express domain, not a custom domain.

**No Streamlit Community Cloud / Vercel deployment** is used or referenced — deployment
is exclusively the Docker → ECR → ECS pipeline described above.

---

## 17. Design Decisions

These are explicitly documented in the README's "Design Decisions" section and
corroborated by the code; each represents a deliberate engineering trade-off worth
preserving context on:

1. **Log returns instead of raw prices** — removes trend/non-stationarity, keeps the
   training signal on a consistent scale. Prices are reconstructed post-hoc via
   `last_close × exp(cumsum(returns))`.
2. **Direct multi-output forecasting instead of recursive/autoregressive** — a single
   forward pass emits all 28 future steps, bounding error accumulation to one model
   pass instead of compounding errors across 28 recursive steps, at the cost of a wider
   output head (28 units) fixed at training time (can't dynamically extend horizon
   without retraining).
3. **Monte Carlo Dropout over a full model ensemble** — much cheaper computationally
   (one trained model, 100 fast forward passes) than training/maintaining N separate
   models, while still producing a meaningful uncertainty estimate. Implemented via a
   dedicated `nn.Dropout` layer kept independent from the LSTM's built-in inter-layer
   dropout so MC sampling can be toggled surgically.
4. **Percentiles computed strictly on price trajectories, never on raw return
   percentiles** — because the return→price transform (`exp(cumsum(...))`) is
   nonlinear, and percentile aggregation does not commute with nonlinear functions.
5. **Interchangeable, source-agnostic data layer** (`get_clean_data(source=...)`) —
   allows swapping Yahoo Finance for a managed Snowflake warehouse in production
   without touching model or app code; credentials are environment-variable-only
   (never hardcoded), and missing credentials degrade gracefully to the CSV/yfinance
   path with a warning rather than crashing.
6. **StandardScaler fit strictly on the training split** — a leak-prevention discipline
   applied consistently: fit once during training, persisted via `joblib`, and reused
   with `.transform()`-only calls both for the validation split and for every inference
   call — the scaler is never re-fit at inference time.
7. **Checkpoint-on-best-validation-loss, not checkpoint-on-final-epoch** — ensures the
   artifact shipped to `predict.py`/the Docker image reflects the best generalizing
   model observed during training, not whatever the model looked like after
   `max_epochs` or when training happened to stop.
8. **`hparams.json` as the source of truth for model shape at inference time** —
   `predict.py` reconstructs `BTCPredictor` using the persisted hyperparameters rather
   than `model.py`'s current defaults, decoupling "what shape was this checkpoint
   trained with" from "what are the current code defaults," so changing defaults in
   `model.py`/`train.py` for a future training run cannot silently break inference on
   an already-shipped checkpoint.
9. **Calendar-day (not business-day) date indexing for forecasts** — BTC trades 24/7,
   so `predict.py` explicitly uses `freq="D"` rather than pandas' default business-day
   frequency, which would incorrectly skip weekends.
10. **Docker image ships pre-trained weights** — deliberately excludes `models/` from
    `.gitignore`'s effect on the Docker build context (only `.dockerignore` matters for
    Docker) so the container serves predictions immediately, decoupling
    "build/deploy the app" from "train the model," at the cost of requiring a manual
    training step before each image build if the model should be refreshed.

---

## 18. Current Limitations

Explicitly acknowledged in the README:
- BTC daily returns are close to a random walk; predictive signal from past returns
  alone is limited, and the model's smooth, mean-reverting forecasts will tend to
  underestimate real volatility, especially for a market known for sharp intraday
  moves. The confidence band widens with horizon but doesn't capture true tail risk.
- Explicitly **not** a trading tool or investment-advice system.

Additional limitations observed from the code (not explicitly stated in README, but
verifiable from the source, flagged here as inferred rather than documented):
- **No automated test suite** — no `tests/` directory, no `pytest`/`unittest` files.
  Verification relies on manual sanity-check blocks in `data.py`, `model.py`, and
  `predict.py`.
- **No model evaluation beyond validation MSE** — no backtesting script, no
  calibration check for whether the P10–P90 band actually contains ~80% of realized
  outcomes, no comparison against a naive baseline (e.g., last-price-repeated), despite
  the README itself noting that such a baseline is often competitive.
- **Single-feature input in practice** — although `BTCDataset`/`BTCPredictor` support
  multi-feature (`input_size > 1`) input architecturally, the actual pipeline as wired
  only ever passes the univariate `Log_Return` series; no exogenous features (volume,
  macro indicators, on-chain data, sentiment) are actually used despite `Volume`/
  `High`/`Low` being fetched and available in the cleaned DataFrame.
- **No training resume/checkpoint-loading in `train.py`** — every run trains from a
  freshly initialized model; there's no `--resume-from` flag or logic to continue from
  an existing `models/` checkpoint.
- **No dependency pinning** — `requirements.txt` uses `>=` minimums only, risking
  future reproducibility drift as upstream libraries release breaking changes.
- **No data validation/drift monitoring** — if Yahoo Finance changes its response
  format beyond the already-handled `MultiIndex` case, or if the ticker is delisted/
  renamed, the pipeline has no monitoring or alerting; only the CSV cache masks
  transient failures.
- **Ephemeral Docker data cache** — no persistent volume is defined for `data/` in the
  container, so every container restart triggers a fresh yfinance download.
- **Latent SQL string-interpolation pattern** in `_from_snowflake()`'s query
  construction (table name interpolated via an f-string) — not currently exploitable
  given env vars are operator-controlled, but worth flagging as a pattern to avoid if
  ever extended to accept user-supplied table/ticker values.
- **No authentication/access control** on the deployed Streamlit app — it's fully
  public with no login, consistent with a portfolio/demo purpose but not appropriate if
  ever repurposed for anything sensitive.
- **No infrastructure-as-code** (Terraform/CDK) for the AWS side — the ECS
  cluster/service/ALB/IAM roles are assumed pre-provisioned and are not defined
  anywhere in this repository; the GitHub Actions workflow only builds/pushes/deploys
  against pre-existing infrastructure.

---

## 19. Complete Dependency Graph

Import-level dependency graph (arrows mean "imports from"):

```
app.py ────────► data.py   (get_clean_data)
app.py ────────► predict.py (predict)

predict.py ────► data.py   (get_clean_data)
predict.py ────► model.py  (BTCPredictor, WINDOW, HORIZON)

train.py ──────► data.py   (get_clean_data)
train.py ──────► model.py  (BTCPredictor, build_dataloaders, WINDOW, HORIZON)

model.py ──────► (no internal project dependencies; only external libs)
data.py ───────► (no internal project dependencies; only external libs)
```

`model.py` and `data.py` are the two foundational, dependency-free modules. `predict.py`
and `train.py` are both "middle-tier" orchestrators that depend on both foundational
modules but not on each other. `app.py` is the sole top-level consumer, depending on
`data.py` (for historical chart data) and `predict.py` (for the forecast), but never
importing `model.py` or `train.py` directly — the app never touches the model class or
training loop, only the already-serialized artifacts via `predict.py`.

There are **no circular imports** and no shared mutable global state between modules
(each module's "constants" like `WINDOW`/`HORIZON`/`MODELS_DIR` are simple immutable
values, re-imported by value where needed).

**Artifact-level dependency (files, not imports):**
```
train.py  ──writes──►  models/btc_predictor.pt
train.py  ──writes──►  models/scaler.pkl
train.py  ──writes──►  models/hparams.json

predict.py ──reads───►  models/btc_predictor.pt
predict.py ──reads───►  models/scaler.pkl
predict.py ──reads───►  models/hparams.json

data.py   ──writes/reads──►  data/btc_ohlcv.csv   (cache)
```
This means `predict.py` (and transitively `app.py`) has a **runtime** dependency on
`train.py` having been run at least once — this is not an import dependency but a
filesystem-artifact dependency, explicitly guarded against via the `FileNotFoundError`
handling in `predict._load_artifacts()` and surfaced as a friendly Streamlit error in
`app.py`.

---

## 20. End-to-End Summary

This repository is a compact, single-purpose ML engineering portfolio project that
forecasts Bitcoin's price 7–28 days out, complete with calibrated-looking uncertainty
bands, and serves the result through a polished Streamlit dashboard deployed on AWS.
The engineering is deliberately disciplined for a project of this size: data is
transformed into stationary log returns before modeling; the train/validation split and
feature scaler strictly respect chronological order to avoid look-ahead leakage; the
model performs direct (not recursive) multi-step forecasting to bound error
accumulation; uncertainty is estimated cheaply via Monte Carlo Dropout rather than an
expensive ensemble; and the return-to-price reconversion math is ordered correctly so
that percentile bands are mathematically valid rather than an approximation. The
codebase cleanly separates concerns into four small modules — data acquisition
(`data.py`), model/dataset definitions (`model.py`), training (`train.py`), and
inference (`predict.py`) — all consumed by a single-file Streamlit app (`app.py`) that
has no knowledge of model internals, only of the `predict()` and `get_clean_data()`
public interfaces. Deployment is fully containerized (Docker) and automated (GitHub
Actions → ECR → ECS Fargate via OIDC, no long-lived AWS credentials), with the
container shipping a pre-trained model checkpoint so it can serve predictions
immediately after deployment. The project explicitly and repeatedly disclaims any
trading/investment-advice intent, framing itself instead as a demonstration of
end-to-end ML engineering practice: ingestion, leak-free preprocessing, sequence
modeling, principled uncertainty quantification, and interactive deployment. Its
primary gaps — no automated tests, no backtesting/calibration evaluation, no pinned
dependency versions, single-feature input despite architectural support for more, and
reliance on manually pre-provisioned AWS infrastructure outside the repo — are
consistent with a portfolio-scale project rather than a production trading system, and
should be treated as expected scope boundaries rather than defects, unless the user
indicates they want to extend the project toward production-grade rigor.
