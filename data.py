"""
data.py — BTC-USD OHLCV download, caching, and preprocessing.

Data sources (selectable via get_clean_data(source=...)):
    "csv"       — download from Yahoo Finance via yfinance, cache to local CSV
    "snowflake" — query a Snowflake table (credentials from .env / env vars);
                  falls back to "csv" with a warning if credentials are missing.
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; env vars can be set directly in the shell

# ── Constants ──────────────────────────────────────────────────────────────────

TICKER = "BTC-USD"
PERIOD = "4y"
CACHE_PATH = "data/btc_ohlcv.csv"

# Canonical output column order — identical regardless of source
_OUTPUT_COLS = ["Close", "High", "Low", "Volume", "Log_Return"]

# yfinance only accepts specific period strings; we convert custom ones to start dates
_PERIOD_DAYS: dict[str, int] = {
    "1mo": 30, "3mo": 91, "6mo": 182,
    "1y": 365, "2y": 730, "3y": 1095, "4y": 1460, "5y": 1825,
}

# Snowflake credential env-var names
_SF_KEYS = {
    "account":   "SNOWFLAKE_ACCOUNT",
    "user":      "SNOWFLAKE_USER",
    "password":  "SNOWFLAKE_PASSWORD",
    "warehouse": "SNOWFLAKE_WAREHOUSE",
    "database":  "SNOWFLAKE_DATABASE",
    "schema":    "SNOWFLAKE_SCHEMA",
}
_SF_TABLE_ENV     = "SNOWFLAKE_TABLE"
_SF_TABLE_DEFAULT = "BTC_OHLCV"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _period_to_start(period: str) -> str:
    """Convert a period string (e.g. '4y') to a YYYY-MM-DD start date."""
    days = _PERIOD_DAYS.get(period)
    if days is None:
        raise ValueError(
            f"Unsupported period {period!r}. "
            f"Valid options: {sorted(_PERIOD_DAYS)}"
        )
    return (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")


def _add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Append Log_Return = log(Close_t / Close_{t-1}) and drop the leading NaN."""
    df = df.copy()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["Log_Return"])
    assert df.isnull().sum().sum() == 0, "Unexpected NaN values after cleaning"
    assert np.isfinite(df[["Close", "Log_Return"]].values).all(), "Unexpected ±inf values"
    return df


def _from_yfinance(
    ticker: str = TICKER,
    period: str = PERIOD,
    cache_path: str = CACHE_PATH,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV from Yahoo Finance and persist to a local CSV.
    Subsequent calls load from cache unless force_refresh=True.

    The download uses an explicit start date derived from `period` to guarantee
    exactly ~N years of data (yfinance only supports a fixed set of period
    strings; '4y' is not among them).
    """
    if not force_refresh and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col="Date", parse_dates=True)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    start = _period_to_start(period)
    raw = yf.download(
        ticker,
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns MultiIndex columns when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index.name = "Date"
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df = raw[cols].copy()
    df.to_csv(cache_path)
    return df


def _from_snowflake(period: str = PERIOD) -> Optional[pd.DataFrame]:
    """
    Query BTC OHLCV data from Snowflake.

    Expected table schema (column names case-insensitive):
        DATE DATE, TICKER VARCHAR, HIGH FLOAT, LOW FLOAT, CLOSE FLOAT, VOLUME FLOAT

    Returns None if any required credential is missing or the connector is not
    installed — the caller should fall back to the yfinance/CSV source.
    """
    creds = {k: os.environ.get(env) for k, env in _SF_KEYS.items()}
    missing_vars = [env for k, env in _SF_KEYS.items() if not creds[k]]
    if missing_vars:
        return None  # signal to caller: no credentials available

    try:
        import snowflake.connector  # type: ignore[import]
    except ImportError:
        warnings.warn(
            "snowflake-connector-python is not installed. "
            "Run: pip install snowflake-connector-python",
            stacklevel=4,
        )
        return None

    lookback_days = _PERIOD_DAYS.get(period, 1460)
    table = os.environ.get(_SF_TABLE_ENV, _SF_TABLE_DEFAULT)

    query = f"""
        SELECT DATE, HIGH, LOW, CLOSE, VOLUME
        FROM {table}
        WHERE TICKER = 'BTC-USD'
          AND DATE >= DATEADD(DAY, -{lookback_days}, CURRENT_DATE())
        ORDER BY DATE ASC
    """

    conn = snowflake.connector.connect(**{k: v for k, v in creds.items()})
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        col_names = [desc[0].capitalize() for desc in cursor.description]
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=col_names)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.index.name = "Date"
    return df[["High", "Low", "Close", "Volume"]]


# ── Public API ─────────────────────────────────────────────────────────────────

def get_clean_data(
    source: str = "csv",
    ticker: str = TICKER,
    period: str = PERIOD,
    cache_path: str = CACHE_PATH,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a clean, model-ready DataFrame with a DatetimeIndex.

    Columns (always in this order):
        Close       — adjusted daily close price (USD)
        High        — daily high (USD)
        Low         — daily low (USD)
        Volume      — daily volume
        Log_Return  — log(Close_t / Close_{t-1})

    Parameters
    ----------
    source : {"csv", "snowflake"}
        "csv"       → yfinance download, cached to `cache_path`.
        "snowflake" → Snowflake query via env-var credentials.
                      Falls back to "csv" with a UserWarning if credentials
                      are absent or the connector is not installed.
    period : str
        Lookback window, e.g. "4y", "2y", "6mo".
    force_refresh : bool
        If True, bypass the CSV cache and re-download from yfinance.
    """
    if source == "snowflake":
        raw = _from_snowflake(period=period)
        if raw is None:
            warnings.warn(
                "Snowflake source unavailable (missing credentials or connector). "
                "Falling back to yfinance/CSV.",
                UserWarning,
                stacklevel=2,
            )
            raw = _from_yfinance(
                ticker=ticker, period=period,
                cache_path=cache_path, force_refresh=force_refresh,
            )
    elif source == "csv":
        raw = _from_yfinance(
            ticker=ticker, period=period,
            cache_path=cache_path, force_refresh=force_refresh,
        )
    else:
        warnings.warn(f"Unknown source={source!r}; defaulting to 'csv'.", stacklevel=2)
        raw = _from_yfinance(
            ticker=ticker, period=period,
            cache_path=cache_path, force_refresh=force_refresh,
        )

    df = pd.DataFrame(index=raw.index)
    df["Close"]  = raw["Close"].astype(float)
    df["High"]   = raw["High"].astype(float)
    df["Low"]    = raw["Low"].astype(float)
    df["Volume"] = raw["Volume"].astype(float)

    df = _add_log_returns(df)
    return df[_OUTPUT_COLS]


if __name__ == "__main__":
    df = get_clean_data()
    print(f"Shape     : {df.shape}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast  5 rows:\n{df.tail()}")
    print(f"\nBasic stats:\n{df.describe()}")
