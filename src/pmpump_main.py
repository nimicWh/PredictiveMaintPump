import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

from src.feature_engineering import create_features

# -----------------------------
# CONFIG
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.3
ISO_CONTAMINATION = 0.01
SHAP_WINDOW = 200
PLC_WINDOW = 10

logging.basicConfig(level=logging.INFO)

# -----------------------------
# LOAD DATA
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(current_dir, "data/dataset_ai4.csv")
results_csv = os.path.join(current_dir, "outputs/results.csv")
shap_csv = os.path.join(current_dir, "outputs/shap_values.csv")

logging.info("Loading dataset...")
df = pd.read_csv(input_csv)

df = df.drop(columns=["UDI", "Product ID", "Type"])
df.columns = [
    "AirTemp", "ProcessTemp", "RotSpeed", "Torque", "ToolWear",
    "MachineFailure", "TWF", "HDF", "PWF", "OSF", "RNF"
]

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
sensor_cols = ["AirTemp", "ProcessTemp", "RotSpeed", "Torque", "ToolWear"]
fft_cols = ["RotSpeed", "Torque"]  # example
X = create_features(df, sensor_cols, fft_cols=fft_cols, window=50)
y = df["MachineFailure"]

# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
train_idx = X_train.index
test_idx = X_test.index

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# ISOLATION FOREST
# -----------------------------
logging.info("Training Isolation Forest...")
iso_model = IsolationForest(
    contamination=ISO_CONTAMINATION,
    random_state=RANDOM_STATE
)
iso_model.fit(X_train_scaled[y_train == 0])
iso_pred_test = np.where(iso_model.predict(X_test_scaled) == -1, 1, 0)
df["ISO_Prediction"] = 0
df.loc[test_idx, "ISO_Prediction"] = iso_pred_test

# -----------------------------
# RANDOM FOREST
# -----------------------------
logging.info("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_probs = rf_model.predict_proba(X_test_scaled)[:,1]
rf_pred = (rf_probs >= 0.5).astype(int)
df["RF_Prediction"] = 0
df.loc[test_idx, "RF_Prediction"] = rf_pred

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_model(name, y_true, y_pred, probs=None):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{name} Results")
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    if probs is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, probs):.3f}")
    return cm

evaluate_model("Random Forest", y_test, rf_pred, rf_probs)

# -----------------------------
# THRESHOLD OPTIMIZATION
# -----------------------------
thresholds = np.linspace(0.01, 0.99, 200)
f1_scores = [f1_score(y_test, (rf_probs >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
rf_pred_opt = (rf_probs >= best_threshold).astype(int)
evaluate_model("Random Forest (Optimized)", y_test, rf_pred_opt, rf_probs)
df.loc[test_idx, "RF_Prediction_Opt"] = rf_pred_opt

# -----------------------------
# SHAP EXPLAINABILITY
# -----------------------------
logging.info("Calculating SHAP values...")
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
explainer = shap.TreeExplainer(rf_model, model_output="raw")
shap_values = explainer.shap_values(X_test_df)
shap_df = pd.DataFrame(shap_values[1], columns=X.columns)
shap_df["Index"] = test_idx.values
shap_df = shap_df.sort_values("Index")

# Rolling SHAP
rolling_shap = shap_df[X.columns].abs().rolling(SHAP_WINDOW).mean()
# (Plotting code here if desired)

shap_df.to_csv(shap_csv, index=False)
logging.info(f"SHAP values saved to {shap_csv}")

# -----------------------------
# RUL CALCULATION
# -----------------------------
rul = np.full(len(df), np.nan)
next_failure = None
for i in reversed(range(len(df))):
    if df.loc[i, "MachineFailure"] == 1:
        next_failure = i
    if next_failure is not None:
        rul[i] = next_failure - i
df["RUL"] = rul

# -----------------------------
# PLC ALARM SIMULATION
# -----------------------------
recent_anomalies = df["ISO_Prediction"].iloc[-PLC_WINDOW:]
plc_alarm = 1 if np.any(recent_anomalies == 1) else 0
status = "⚠ WARNING" if plc_alarm else "HEALTHY"
print(f"\nPLC Alarm Bit: {plc_alarm} ({status})")

# -----------------------------
# SAVE RESULTS
# -----------------------------
df.to_csv(results_csv, index=False)
logging.info(f"Results saved to {results_csv}")
logging.info("Pipeline execution completed successfully.")
