 ============================================================
# DASHBOARD
# File: app.py
# Run: streamlit run app.py
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title = "Google Stock Forecast",
    page_icon = "📈",
    layout = "wide"
)

st.title("📈 Google Stock Price Forecasting")
st.markdown("**Powered by LSTM Deep Learning**")
st.markdown("---")

# ── LOAD MODEL ───────────────────────────────────────────────
@st.cache_resource
def load_lstm():
    import tensorflow as tf
    model = tf.keras.models.load_model(
        'google_model.h5',
        compile=False,
        safe_mode=False
    )
    return model

# ── LOAD LIVE DATA ───────────────────────────────────────────
@st.cache_data
def load_data():
    df = yf.download(
        'GOOG',
        start='2018-01-01',
        end=pd.Timestamp.today().strftime('%Y-%m-%d')
    )
    df = df.reset_index()
    df.columns = [
        'date', 'open', 'high',
        'low', 'close', 'adjclose', 'volume'
    ]
    df = df.sort_values('date').reset_index(drop=True)
    df.dropna(inplace=True)
    return df

# ── LOADING ──────────────────────────────────────────────────
with st.spinner('Loading model and live data...'):
    model = load_lstm()
    df = load_data()

st.success('Model loaded successfully!')

# ── SHOW LATEST DATA DATE ────────────────────────────────────
st.markdown(f"📅 **Data updated up to:** {df['date'].iloc[-1].strftime('%B %d, %Y')}")

# ── INITIALIZE ───────────────────────────────────────────────
FEATURES = [
    'open', 'high', 'low',
    'close', 'volume', 'adjclose'
]
TARGET_IDX = FEATURES.index('adjclose')
LOOKBACK = 90
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[FEATURES].values)

# ── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value = 7,
    max_value = 60,
    value = 30
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.markdown("- **Algorithm:** LSTM")
st.sidebar.markdown("- **Lookback:** 90 days")
st.sidebar.markdown("- **Features:** 6 columns")
st.sidebar.markdown("- **RMSE:** $6.06")
st.sidebar.markdown("- **MAE:** $4.80")
st.sidebar.markdown("- **Built by:** Habiba")

# ── FORECAST ─────────────────────────────────────────────────
with st.spinner('Generating forecast...'):
    current_seq = scaled_data[-LOOKBACK:].copy()
    future_preds = []

    for _ in range(forecast_days):
        inp = current_seq.reshape(1, LOOKBACK, len(FEATURES))
        next_val = model.predict(inp, verbose=0)[0, 0]
        future_preds.append(next_val)
        new_row = current_seq[-1].copy()
        new_row[TARGET_IDX] = next_val
        current_seq = np.vstack([current_seq[1:], new_row])

    dummy = np.zeros((forecast_days, len(FEATURES)))
    dummy[:, TARGET_IDX] = future_preds
    future_prices = scaler.inverse_transform(dummy)[:, TARGET_IDX]
    future_dates = pd.bdate_range(
        start = df['date'].iloc[-1] + pd.Timedelta(days=1),
        periods = forecast_days
    )

# ── METRICS ──────────────────────────────────────────────────
last_price = df['adjclose'].iloc[-1]
first_pred = future_prices[0]
last_pred = future_prices[-1]
change = last_pred - last_price
change_pct = (change / last_price) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Real Price", f"${last_price:.2f}")
col2.metric("Day +1 Forecast", f"${first_pred:.2f}")
col3.metric(f"Day +{forecast_days}", f"${last_pred:.2f}")
col4.metric("Expected Change",
            f"${change:.2f}",
            f"{change_pct:.2f}%")

st.markdown("---")

# ── CHART ────────────────────────────────────────────────────
st.subheader("📊 Price History + Forecast")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(
    df['date'].iloc[-180:],
    df['adjclose'].iloc[-180:],
    color='#2196F3', linewidth=1.8,
    label='Actual Price'
)
ax.plot(
    future_dates, future_prices,
    color='#E91E63', linewidth=2,
    linestyle='--', marker='o',
    markersize=3,
    label=f'Forecast ({forecast_days} days)'
)
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ── FORECAST TABLE ───────────────────────────────────────────
st.subheader("📋 Forecast Table")

forecast_df = pd.DataFrame({
    'Date' : future_dates.strftime('%Y-%m-%d'),
    'Predicted Price' : [f"${p:.2f}" for p in future_prices],
    'Change from Today': [f"${p - last_price:+.2f}"
                          for p in future_prices]
})
st.dataframe(forecast_df, use_container_width=True)

csv = forecast_df.to_csv(index=False)
st.download_button(
    label = "⬇️ Download Forecast CSV",
    data = csv,
    file_name= "google_forecast.csv",
    mime = "text/csv"
)

st.markdown("---")
st.markdown("Built by **Habiba** | LSTM Stock Forecasting")

