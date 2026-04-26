# ================================
# IMPORTS
# ================================
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
import random
import joblib
import requests
from streamlit_autorefresh import st_autorefresh

from agents.detection_agent import DetectionAgent
from agents.ueba_agent import UEBAAgent
from agents.anomaly_agent import AnomalyAgent

import warnings
warnings.filterwarnings("ignore")

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="SOC Dashboard", layout="wide")
st.title("🛡️ AI-Powered SOC Dashboard")

st_autorefresh(interval=2000, key="refresh")

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_all():
    scaler = joblib.load("outputs/models/scaler.pkl")
    selector = joblib.load("outputs/models/selector.pkl")

    detector = DetectionAgent()
    detector.load("outputs/models/xgb_model.json")

    ueba = UEBAAgent()
    ueba.model.load("outputs/models/ueba_model.pkl")

    anomaly = AnomalyAgent()
    anomaly.load("outputs/models/anomaly_model.pkl")

    return scaler, selector, detector, ueba, anomaly

scaler, selector, detector, ueba, anomaly = load_all()

# ================================
# SESSION STATE
# ================================
if "data" not in st.session_state:
    st.session_state.data = []

# ================================
# SIDEBAR
# ================================
st.sidebar.header("⚙️ Controls")
run = st.sidebar.checkbox("▶ Enable Monitoring")
threshold = st.sidebar.slider("Threat Threshold", 0.1, 0.9, 0.3)

# ================================
# GEO + IP
# ================================
geo_cache = {}

def generate_ip():
    ranges = [(8,8,8),(1,1,1),(52,95,110),(142,250,0)]
    b = random.choice(ranges)
    return f"{b[0]}.{b[1]}.{random.randint(0,255)}.{random.randint(0,255)}"

def get_geo(ip):
    if ip in geo_cache:
        return geo_cache[ip]
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}", timeout=1).json()
        if r["status"] == "success":
            geo_cache[ip] = (r["country"], r["lat"], r["lon"])
            return geo_cache[ip]
    except:
        pass
    return ("India", 12.97, 77.59)

# ================================
# HELPERS
# ================================
def classify(prob, an):
    if prob > 0.8: return "DoS"
    if an > 0.85: return "Probe"
    if prob > 0.5: return "R2L"
    if an > 0.6: return "U2R"
    return "Normal"

def severity(prob, an):
    if prob > 0.8: return "CRITICAL"
    if prob > 0.5: return "HIGH"
    if an > 0.6: return "MEDIUM"
    return "LOW"

# ================================
# DATA GENERATION
# ================================
if run:
    for _ in range(8):
        sample = np.random.rand(78)
        X = selector.transform(scaler.transform(sample.reshape(1, -1)))
        X = pd.DataFrame(X)

        prob = detector.predict_proba(X)[0]
        fraud = ueba.detect(X)[0]
        an = anomaly.detect(X)[0]

        score = 0.5*prob + 0.2*fraud + 0.3*an
        final = 1 if score > threshold or random.random() > 0.85 else 0

        ip = generate_ip()
        country, lat, lon = get_geo(ip)

        st.session_state.data.append({
            "Time": pd.Timestamp.now(),
            "IP": ip,
            "Country": country,
            "Prob": prob,
            "Type": classify(prob, an),
            "Severity": severity(prob, an),
            "Anomaly": an,
            "Final": final,
            "lat": lat,
            "lon": lon
        })

    if len(st.session_state.data) > 300:
        st.session_state.data = st.session_state.data[-300:]

# ================================
# DATAFRAME
# ================================
df = pd.DataFrame(st.session_state.data)

if df.empty:
    st.info("👈 Enable Monitoring to start SOC system")
    st.stop()

# ================================
# KPI PANEL
# ================================
col1, col2, col3, col4 = st.columns(4)

attacks = int((df["Final"] == 1).sum())
normal = int((df["Final"] == 0).sum())
countries = df["Country"].nunique()
risk_score = round(df["Prob"].mean(), 2)

col1.metric("🚨 Attacks", attacks)
col2.metric("✅ Normal", normal)
col3.metric("🌍 Countries", countries)
col4.metric("🎯 Risk Score", risk_score)

# ================================
# ALERT FEED (SOC STYLE)
# ================================
st.subheader("🚨 Live Threat Feed")

latest = df.sort_values("Time", ascending=False).head(10)

for _, row in latest.iterrows():
    if row["Severity"] == "CRITICAL":
        st.error(f"🔴 {row['IP']} | {row['Type']} | {row['Country']}")
    elif row["Severity"] == "HIGH":
        st.warning(f"🟠 {row['IP']} | {row['Type']} | {row['Country']}")
    else:
        st.info(f"🟢 {row['IP']} | {row['Type']}")

# ================================
# MAP
# ================================
st.subheader("🌍 Global Attack Map")
st.map(df[["lat","lon"]])

# ================================
# CHARTS
# ================================
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Attack Types")
    st.bar_chart(df["Type"].value_counts())

with c2:
    st.subheader("Severity")
    st.bar_chart(df["Severity"].value_counts())

with c3:
    st.subheader("Attack vs Normal")
    st.bar_chart(df["Final"].value_counts())

# ================================
# TIME TREND
# ================================
st.subheader("📈 Attack Trend")
trend = df.set_index("Time").resample("1min")["Final"].sum()
st.line_chart(trend)

# ================================
# SCATTER
# ================================
st.subheader("🔍 Detection Confidence vs Anomaly")
st.scatter_chart(df[["Prob","Anomaly"]])

# ================================
# TOP IPs
# ================================
st.subheader("🔥 Top Attackers")
top = df.groupby("IP")["Final"].sum().sort_values(ascending=False).head(5)
st.bar_chart(top)

# ================================
# TABLE
# ================================
st.subheader("📜 Logs")
st.dataframe(df.tail(20), width="stretch")