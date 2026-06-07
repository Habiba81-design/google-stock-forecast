# ============================================================
# DASHBOARD
# File: app.py
# Run: streamlit run app.py
# ============================================================

# ── IMPORTS ──────────────────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ── PAGE SETUP ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Google Stock Forecast",
    page_icon = "📈",
    layout = "wide"
)

st.title("📈 Google Stock Price Forecasting")
st.markdown("**Powered by LSTM Deep Learning**")
st.markdown("---")

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Google_Stock_Price.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    return df

# ── LOAD MODEL ───────────────────────────────────────────────
@st.cache_resource
def load_lstm():
    return load_model(
        'google_model.h5',
        compile=False
    )

# ── INITIALIZE ───────────────────────────────────────────────
df = load_data()
model = load_lstm()

FEATURES = [
    'open', 'high', 'low',
    'close', 'volume', 'adjclose'
]
TARGET_IDX = FEATURES.index('adjclose')
LOOKBACK = 90
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[FEATURES].values)

# ── SIDEBAR ───────────────────────
