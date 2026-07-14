"""
app.py — Streamlit + Plotly dashboard for BTC price prediction.

Run with:
    streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data import get_clean_data
from predict import predict

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BTC Price Predictor",
    page_icon="₿",
    layout="wide",
)

# ── Cached data loader ───────────────────────────────────────────────────────
# Defined before the sidebar so the "Actualizar datos" button below can call
# `_load_history.clear()` on this same function later in the script.

@st.cache_data(ttl=3600, show_spinner="Fetching BTC price history…")
def _load_history() -> pd.DataFrame:
    return get_clean_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("₿ BTC Predictor")
    st.markdown("LSTM · Monte Carlo Dropout")
    st.divider()

    horizon = st.selectbox(
        "Forecast horizon",
        options=[7, 14, 21, 28],
        index=3,
        help="Calendar days ahead — BTC trades 24/7, no market closures.",
    )
    hist_days = st.slider(
        "Historical days to display",
        min_value=30, max_value=365, value=120, step=10,
    )

    st.divider()

    # Manual refresh: re-download from yfinance (bypassing the CSV cache) up
    # to today, then invalidate Streamlit's in-memory cache and rerun. Only
    # get_clean_data() runs again — the trained model artifacts are never
    # touched, so this never retrains anything.
    if st.button("🔄 Actualizar datos", use_container_width=True):
        try:
            with st.spinner("Descargando datos actualizados desde Yahoo Finance…"):
                get_clean_data(force_refresh=True)
            _load_history.clear()
            st.session_state["refresh_error"] = None
            st.rerun()
        except Exception as exc:
            st.session_state["refresh_error"] = str(exc)

    if st.session_state.get("refresh_error"):
        st.error(
            "⚠️ No se pudieron actualizar los datos desde Yahoo Finance. "
            "Se muestran los últimos datos disponibles.\n\n"
            f"Detalle: {st.session_state['refresh_error']}"
        )

    st.divider()
    st.caption(
        "Architecture: 2-layer LSTM + Dropout  \n"
        "Inference: 100 MC forward passes  \n"
        "Input window: 60 days  |  Max horizon: 28 days"
    )

# ── Load historical data ──────────────────────────────────────────────────────

try:
    df_hist = _load_history()
except Exception as exc:
    st.error(f"Failed to load price history: {exc}")
    st.stop()

hist_slice = df_hist["Close"].tail(hist_days)
last_close = float(hist_slice.iloc[-1])
last_date  = hist_slice.index[-1]

st.sidebar.caption(f"📅 Datos actualizados al {last_date.strftime('%d/%m/%Y')}")

# ── Run forecast ──────────────────────────────────────────────────────────────

try:
    with st.spinner("Running Monte Carlo Dropout inference (100 passes)…"):
        forecast = predict(horizon_days=horizon)

except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "No trained model found in `models/`.  \n"
        "Generate the artifacts by running:"
    )
    st.code("python train.py", language="bash")
    st.stop()

except Exception as exc:
    st.error(f"Prediction error: {exc}")
    st.stop()

# ── Metrics row ───────────────────────────────────────────────────────────────

st.header(f"BTC/USD — {horizon}-day Price Forecast")

end_pred  = forecast.iloc[-1]
pct_delta = (end_pred["mean"] / last_close - 1) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close",          f"${last_close:,.0f}")
c2.metric(f"Day-{horizon} mean", f"${end_pred['mean']:,.0f}",
          delta=f"{pct_delta:+.1f}%")
c3.metric("Lower bound (P10)",   f"${end_pred['p10']:,.0f}")
c4.metric("Upper bound (P90)",   f"${end_pred['p90']:,.0f}")

# ── Build Plotly figure ───────────────────────────────────────────────────────

# Prepend the last historical point so all forecast traces originate exactly
# where the historical line ends — no visual gap, no duplicated data point.
fc_dates = [last_date]  + list(forecast.index)
fc_mean  = [last_close] + list(forecast["mean"])
fc_p10   = [last_close] + list(forecast["p10"])
fc_p90   = [last_close] + list(forecast["p90"])

fig = go.Figure()

# Historical close (Bitcoin orange)
fig.add_trace(go.Scatter(
    x    = hist_slice.index,
    y    = hist_slice.values,
    name = "Historical close",
    mode = "lines",
    line = dict(color="#F7931A", width=2),
))

# P10–P90 shaded band — polygon built by going up along p90 then back down
# along p10; hover suppressed so it doesn't clutter the unified tooltip.
fig.add_trace(go.Scatter(
    x           = fc_dates + fc_dates[::-1],
    y           = fc_p90   + fc_p10[::-1],
    fill        = "toself",
    fillcolor   = "rgba(99, 149, 255, 0.15)",
    line        = dict(width=0),
    mode        = "lines",
    name        = "P10–P90 band",
    hoverinfo   = "skip",
    showlegend  = True,
))

# Forecast mean (dotted blue)
fig.add_trace(go.Scatter(
    x    = fc_dates,
    y    = fc_mean,
    name = "Forecast mean",
    mode = "lines",
    line = dict(color="#6395FF", width=2.5, dash="dot"),
))

# P10 and P90 boundary lines — thin, visible in unified hover for exact values
for label, values in [("P10", fc_p10), ("P90", fc_p90)]:
    fig.add_trace(go.Scatter(
        x    = fc_dates,
        y    = values,
        name = label,
        mode = "lines",
        line = dict(color="rgba(99, 149, 255, 0.45)", width=1, dash="dot"),
    ))

# Vertical dashed line at the historical/forecast boundary
fig.add_vline(
    x                    = last_date,
    line                 = dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"),
    annotation_text      = f"Last close  ({last_date.date()})",
    annotation_position  = "top left",
    annotation_font      = dict(color="rgba(255,255,255,0.45)", size=11),
)

fig.update_layout(
    template  = "plotly_dark",
    hovermode = "x unified",
    xaxis     = dict(title="Date", showgrid=False),
    yaxis     = dict(
        title      = "Price (USD)",
        tickprefix = "$",
        tickformat = ",.0f",
        gridcolor  = "rgba(255,255,255,0.05)",
    ),
    legend = dict(
        orientation = "h",
        yanchor     = "bottom",
        y           = 1.01,
        xanchor     = "left",
        x           = 0,
    ),
    margin = dict(t=60, b=40),
    height = 520,
)

st.plotly_chart(fig, use_container_width=True)

# ── Forecast table (collapsible) ──────────────────────────────────────────────

with st.expander("Forecast data table"):
    display = forecast.copy()
    display.index = display.index.strftime("%Y-%m-%d")
    display.index.name = "Date"
    st.dataframe(
        display,
        column_config={
            "mean": st.column_config.NumberColumn("Mean price ($)", format="$%.0f"),
            "p10":  st.column_config.NumberColumn("P10 ($)",        format="$%.0f"),
            "p90":  st.column_config.NumberColumn("P90 ($)",        format="$%.0f"),
        },
        use_container_width=True,
    )
