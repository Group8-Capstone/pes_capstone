# ================================
# IMPORTS
# ================================
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
import requests
from streamlit_autorefresh import st_autorefresh

from agents.detection_agent import DetectionAgent
from agents.ueba_agent import UEBAAgent
from agents.anomaly_agent import AnomalyAgent

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Enterprise SOC", layout="wide")
st.title("Enterprise AI SOC Dashboard")

st_autorefresh(interval=2000, key="soc_refresh")

# ================================
# LOAD MODELS (SAFE)
# ================================
@st.cache_resource
def load_models():
    scaler = joblib.load("outputs/models/scaler.pkl")
    selector = joblib.load("outputs/models/selector.pkl")

    detector = DetectionAgent()
    detector.load("outputs/models/xgb_model.json")

    ueba = UEBAAgent()
    ueba.model.load("outputs/models/ueba_model.pkl")

    anomaly = AnomalyAgent()
    anomaly.load("outputs/models/anomaly_model.pkl")

    raw_dim = scaler.n_features_in_

    return scaler, selector, detector, ueba, anomaly, raw_dim

scaler, selector, detector, ueba, anomaly, raw_dim = load_models()

# ================================
# LOAD STREAM DATA
# ================================
@st.cache_data
def load_stream():
    path = "outputs/test_stream.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

stream_df = load_stream()

# ================================
# SESSION STATE
# ================================
if "events" not in st.session_state:
    st.session_state.events = []

if "pointer" not in st.session_state:
    st.session_state.pointer = 0

# ================================
# SIDEBAR
# ================================
st.sidebar.header("SOC Controls")

run = st.sidebar.toggle("Live Monitoring", True)
threshold = st.sidebar.slider("Threat Threshold", 0.1, 0.9, 0.5)

# ================================
# GEO FUNCTIONS
# ================================
def generate_ip():
    return f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"

def get_geo(ip):
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}", timeout=1).json()
        if r["status"] == "success":
            return r["country"], r["lat"], r["lon"]
    except:
        pass
    return "India", 12.97, 77.59

# ================================
# CLASSIFICATION
# ================================
def classify(prob, an):
    if prob > 0.8: return "DoS"
    if an > 0.85: return "Probe"
    if prob > 0.6: return "R2L"
    if an > 0.6: return "U2R"
    return "Normal"

def severity(score):
    if score > 0.85: return "CRITICAL"
    if score > 0.65: return "HIGH"
    if score > 0.5: return "MEDIUM"
    return "LOW"

# ================================
# EVENT GENERATION (REAL STREAM)
# ================================
if run:

    if not stream_df.empty:

        for _ in range(5):

            # loop over real dataset
            row = stream_df.iloc[st.session_state.pointer % len(stream_df)]
            st.session_state.pointer += 1

            X_raw = pd.DataFrame([row.values])

            # PIPELINE
            X_scaled = scaler.transform(X_raw)
            X_sel = selector.transform(X_scaled)
            X_sel = pd.DataFrame(X_sel)

            # AGENTS
            prob = detector.predict_proba(X_sel)[0]
            fraud = ueba.detect(X_sel)[0]
            an = anomaly.detect(X_sel)[0]

            score = 0.5*prob + 0.2*fraud + 0.3*an
            decision = "BLOCK" if score > threshold else "ALLOW"

            ip = generate_ip()
            country, lat, lon = get_geo(ip)

            st.session_state.events.append({
                "Time": pd.Timestamp.now(),
                "IP": ip,
                "Country": country,
                "Prob": prob,
                "Fraud": fraud,
                "Anomaly": an,
                "Score": score,
                "Decision": decision,
                "Type": classify(prob, an),
                "Severity": severity(score),
                "lat": lat,
                "lon": lon
            })

    else:
        # FALLBACK DATA (prevents blank screen)
        st.warning("⚠️ No stream data found → using dummy data")

        for _ in range(3):
            st.session_state.events.append({
                "Time": pd.Timestamp.now(),
                "IP": "192.168.1.1",
                "Country": "India",
                "Prob": 0.9,
                "Fraud": 1,
                "Anomaly": 1,
                "Score": 0.9,
                "Decision": "BLOCK",
                "Type": "DoS",
                "Severity": "CRITICAL",
                "lat": 12.97,
                "lon": 77.59
            })

# ================================
# DATAFRAME
# ================================
df = pd.DataFrame(st.session_state.events)

if df.empty:
    st.warning("No data yet...")
    st.stop()

# ================================
# KPI PANEL
# ================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Threats", int((df["Decision"]=="BLOCK").sum()))
c2.metric("Allowed", int((df["Decision"]=="ALLOW").sum()))
c3.metric("Countries", df["Country"].nunique())
c4.metric("Avg Risk", round(df["Score"].mean(), 2))

# ================================
# ALERT FEED
# ================================
st.subheader("Live SOC Alerts")

for _, row in df.sort_values("Time", ascending=False).head(6).iterrows():
    msg = f"{row['IP']} | {row['Type']} | Score={round(row['Score'],2)}"

    if row["Severity"] == "CRITICAL":
        st.error("🔴 " + msg)
    elif row["Severity"] == "HIGH":
        st.warning("🟠 " + msg)
    else:
        st.info("🟢 " + msg)

# ================================
# MAP
# ================================
st.subheader("Global Threat Map")
st.map(df[["lat","lon"]])

# ================================
# CHARTS
# ================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Attack Types")
    st.bar_chart(df["Type"].value_counts())

with col2:
    st.subheader("Severity")
    st.bar_chart(df["Severity"].value_counts())

with col3:
    st.subheader("Decisions")
    st.bar_chart(df["Decision"].value_counts())

# ================================
# TREND
# ================================
st.subheader("Threat Trend")
trend = df.set_index("Time").resample("1min")["Score"].mean()
st.line_chart(trend)

# ================================
# TABLE
# ================================
st.subheader("📜 Event Logs")
st.dataframe(df.tail(20), width="stretch")