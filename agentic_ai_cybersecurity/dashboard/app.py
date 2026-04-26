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

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="SOC Dashboard", layout="wide")
st.title("🔴 AI Cybersecurity SOC Dashboard (Real-Time SOC)")

# 🔥 AUTO REFRESH (2 sec)
st_autorefresh(interval=2000, key="soc_refresh")

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

threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.3, step=0.01)

st.sidebar.subheader("🔍 Filters")

filter_status = st.sidebar.selectbox(
    "Traffic", ["All", "Attacks Only", "Normal Only"]
)

filter_type = st.sidebar.multiselect(
    "Attack Type",
    ["DoS", "Probe", "R2L", "U2R", "Normal"],
    default=["DoS", "Probe", "R2L", "U2R", "Normal"]
)

prob_range = st.sidebar.slider(
    "Probability",
    0.0, 1.0,
    (0.0, 1.0),
    step=0.01
)

search_ip = st.sidebar.text_input("🔎 Search IP")
time_filter = st.sidebar.slider("Last Minutes", 1, 60, 10)

# ================================
# GEO
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
            res = (r["country"], r["lat"], r["lon"])
            geo_cache[ip] = res
            return res
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

def apply_filters(df):
    if df.empty:
        return df

    cutoff = pd.Timestamp.now() - pd.Timedelta(minutes=time_filter)
    df = df[df["Time"] >= cutoff]

    if filter_status == "Attacks Only":
        df = df[df["Final"] == 1]
    elif filter_status == "Normal Only":
        df = df[df["Final"] == 0]

    df = df[df["Type"].isin(filter_type)]

    df = df[(df["Prob"] >= prob_range[0]) & (df["Prob"] <= prob_range[1])]

    if search_ip:
        df = df[df["IP"].str.contains(search_ip)]

    return df

# ================================
# 🔥 REAL-TIME DATA GENERATION
# ================================
if run:

    for _ in range(10):  # 🔥 multiple points per refresh

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

    # 🔥 limit memory
    if len(st.session_state.data) > 500:
        st.session_state.data = st.session_state.data[-500:]

# ================================
# DATA
# ================================
df = pd.DataFrame(st.session_state.data)
f = apply_filters(df)

# ================================
# 🔥 TOP PIE
# ================================
if not f.empty:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚨 Attack vs Normal")
        st.bar_chart(f["Final"].value_counts())

    with col2:
        st.subheader("⚡ Severity")
        st.bar_chart(f["Severity"].value_counts())

# ================================
# KPI
# ================================
c1, c2, c3 = st.columns(3)
c1.metric("🚨 Attacks", int((f["Final"]==1).sum()) if not f.empty else 0)
c2.metric("✅ Normal", int((f["Final"]==0).sum()) if not f.empty else 0)
c3.metric("🌍 Countries", f["Country"].nunique() if not f.empty else 0)

# ================================
# TABLE
# ================================
st.subheader("📋 Logs")
st.dataframe(f.tail(20), use_container_width=True)

# ================================
# MAP
# ================================
st.subheader("🌍 Global Map")
if not f.empty:
    st.map(f[["lat","lon"]])

# ================================
# CHARTS
# ================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Attack Types")
    st.bar_chart(f["Type"].value_counts())

with col2:
    st.subheader("Attack vs Normal")
    st.bar_chart(f["Final"].value_counts())

with col3:
    st.subheader("Severity")
    st.bar_chart(f["Severity"].value_counts())

# ================================
# TIME TREND
# ================================
st.subheader("⏱ Time Trend")
if not f.empty:
    trend = f.set_index("Time").resample("1min")["Final"].sum()
    st.line_chart(trend)

# ================================
# SCATTER
# ================================
st.subheader("🧠 Confidence vs Anomaly")
if not f.empty:
    st.scatter_chart(f[["Prob","Anomaly"]])

# ================================
# TOP IPs
# ================================
st.subheader("📊 Top Risky IPs")
if not f.empty:
    risky = f.groupby("IP")["Final"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(risky)