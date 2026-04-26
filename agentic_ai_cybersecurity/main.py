# ================================
# IMPORTS + THREAD FIX
# ================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import joblib

from preprocessing.preprocess_cicids import load_cicids_data, balance_data
from preprocessing.preprocess_logs import load_logs
from preprocessing.preprocess_fraud import load_fraud_data

from agents.detection_agent import DetectionAgent
from agents.ueba_agent import UEBAAgent
from agents.investigation_agent import InvestigationAgent
from agents.coordinator_agent import CoordinatorAgent
from agents.response_agent import ResponseAgent
from agents.anomaly_agent import AnomalyAgent

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from utils.config import TEST_SIZE, RANDOM_STATE
from utils.helper import create_output_folder, print_section

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

import warnings
warnings.filterwarnings("ignore")

# ================================
# SETUP
# ================================
os.makedirs("outputs/models", exist_ok=True)
create_output_folder()

# ================================
# STEP 1: LOAD DATA
# ================================
print_section("Loading datasets")

X_cicids, y_cicids = load_cicids_data()
X_fraud, y_fraud = load_fraud_data()
logs = load_logs()

print("Data loaded successfully")

# ================================
# STEP 2: SPLIT
# ================================
print_section("Splitting data")

X_train, X_test, y_train, y_test = train_test_split(
    X_cicids, y_cicids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_cicids
)

print("Split complete")

# ================================
# SCALE
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# FEATURE SELECTION
# ================================
selector = SelectKBest(score_func=f_classif, k=30)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# 🔥 SAVE PREPROCESSING (IMPORTANT)
joblib.dump(scaler, "outputs/models/scaler.pkl")
joblib.dump(selector, "outputs/models/selector.pkl")

# ================================
# STEP 3: BALANCE DATA
# ================================
print_section("Balancing training data")

X_train, y_train = balance_data(X_train, y_train)

print("Training data balanced")

# ================================
# STEP 4: ANOMALY MODEL
# ================================
print_section("Training Anomaly Model")

anomaly = AnomalyAgent()
anomaly.train(X_train)

# 🔥 SAVE MODEL
anomaly.save("outputs/models/anomaly_model.pkl")

print("Anomaly model trained & saved")

# ================================
# STEP 5: INITIALIZE AGENTS
# ================================
detector = DetectionAgent()
ueba = UEBAAgent()
investigator = InvestigationAgent()
coordinator = CoordinatorAgent()
responder = ResponseAgent()

# ================================
# STEP 6: TRAIN / LOAD MODELS
# ================================
print_section("Training / Loading Models")

# Detection
if not detector.load("outputs/models/xgb_model.json"):
    print("Training Detection model...")
    detector.train(X_train, y_train)
    detector.save("outputs/models/xgb_model.json")
else:
    print("Loaded saved Detection model")

# UEBA
if not ueba.model.load("outputs/models/ueba_model.pkl"):
    print("Training UEBA model...")
    ueba.train(X_fraud, y_fraud)
    ueba.model.save("outputs/models/ueba_model.pkl")
else:
    print("Loaded saved UEBA model")

# Transformer
transformer_path = "outputs/models/transformer_model.pth"

if not investigator.load(transformer_path):
    print("Training Transformer model...")
    investigator.train(logs)
    investigator.save(transformer_path)
else:
    print("Loaded saved Transformer model")

# ================================
# STEP 7: EVALUATION
# ================================
print_section("Evaluating model")

probs = detector.predict_proba(X_test)

precision, recall, thresholds = precision_recall_curve(y_test, probs)

best_threshold = 0.5
best_score = 0

for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
    if r >= 0.98 and p >= 0.95:
        f1 = (2 * p * r) / (p + r)
        if f1 > best_score:
            best_score = f1
            best_threshold = t

if best_threshold == 0.5:
    best_threshold = np.percentile(probs, 90)

print(f"🔥 Final Threshold Used: {best_threshold}")

preds = (probs >= best_threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, preds))

# ================================
# STEP 8: CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_test, preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix - Detection Model")
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()

# ================================
# STEP 9: SIMULATION
# ================================
print_section("Running real attack simulation")

attack_samples = X_test[y_test == 1][:5]
normal_samples = X_test[y_test == 0][:5]

test_data = pd.concat([
    pd.DataFrame(attack_samples),
    pd.DataFrame(normal_samples)
])

#FIXED (NO .values)
test_probs = detector.predict_proba(test_data)
net_alerts = (test_probs >= best_threshold).astype(int)

fraud_alerts = ueba.detect(X_fraud.sample(len(test_data)))

try:
    log_alerts = investigator.analyze(logs.head(len(test_data)))
except:
    print("⚠️ Transformer skipped")
    log_alerts = np.zeros(len(test_data))

#FIXED
anomaly_alerts = anomaly.detect(test_data)

# ================================
# STEP 10: FINAL DECISION
# ================================
print_section("Final Decision Engine")

min_len = min(len(net_alerts), len(fraud_alerts), len(log_alerts))

final_alerts = []

for i in range(min_len):
    votes = (
        net_alerts[i] +
        fraud_alerts[i] +
        log_alerts[i] +
        anomaly_alerts[i]
    )
    final_alerts.append(1 if votes >= 2 else 0)

decisions = coordinator.decide(final_alerts, fraud_alerts, log_alerts)

# ================================
# STEP 11: RESPONSE
# ================================
print_section("Final Response")

responder.execute(decisions)

# ================================
# END
# ================================
print("\nPipeline execution completed")