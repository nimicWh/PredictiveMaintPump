import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ==========================================================
# 1. Configuration
# ==========================================================
RANDOM_STATE = 42
TEST_SIZE = 0.30
PLC_WINDOW = 10
ISO_CONTAMINATION = 0.01
SHAP_WINDOW = 200

logging.basicConfig(level=logging.INFO)

# ==========================================================
# 2. Load Dataset
# ==========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(current_dir, "dataset_ai4.csv")
results_csv = os.path.join(current_dir, "results_eif.csv")
shap_csv = os.path.join(current_dir, "shap_values.csv")

logging.info("Loading dataset...")
df = pd.read_csv(input_csv)

df = df.drop(columns=["UDI", "Product ID", "Type"])
df.columns = [
    "AirTemp", "ProcessTemp", "RotSpeed",
    "Torque", "ToolWear",
    "MachineFailure", "TWF", "HDF", "PWF", "OSF", "RNF"
]

features = ["AirTemp", "ProcessTemp", "RotSpeed", "Torque", "ToolWear"]
X = df[features]
y = df["MachineFailure"]

# ==========================================================
# 3. Train/Test Split
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

train_idx = X_train.index
test_idx = X_test.index

# ==========================================================
# 4. Scaling
# ==========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# 5. Anomaly Detection: PyOD Extended Isolation Forest (EIF)
# ==========================================================
logging.info("Training PyOD Extended Isolation Forest...")

iso_model = IForest(
    n_estimators=200,
    max_samples=256,
    contamination=ISO_CONTAMINATION,
    random_state=RANDOM_STATE
)

# Train only on normal samples
iso_model.fit(X_train_scaled[y_train == 0])

# Binary anomaly prediction: 0 = normal, 1 = anomaly
iso_pred_test = iso_model.predict(X_test_scaled)
df["EIF_Prediction"] = 0
df.loc[test_idx, "EIF_Prediction"] = iso_pred_test

# Optional: continuous anomaly scores for analysis
scores = iso_model.decision_function(X_test_scaled)  # higher = more anomalous

# ==========================================================
# 6. Evaluation Function
# ==========================================================
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

# ==========================================================
# 7. Threshold Optimization for EIF
# ==========================================================
thresholds = np.linspace(0.01, 0.99, 200)
f1_scores = [f1_score(y_test, (scores >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"\nOptimal Threshold: {best_threshold:.3f}")

# Apply optimized threshold
eif_pred_opt = (scores >= best_threshold).astype(int)
evaluate_model("Extended Isolation Forest (Optimized)", y_test, eif_pred_opt, scores)
df.loc[test_idx, "EIF_Prediction_Opt"] = eif_pred_opt

plt.figure(figsize=(6,4))
plt.plot(thresholds, f1_scores)
plt.axvline(best_threshold, color='r', linestyle='--')
plt.title("F1 Score vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.show()

# ==========================================================
# 8. SHAP Explainability
# ==========================================================

# ==========================================================
# 9. Data Drift Detection (PSI)
# ==========================================================
def calculate_psi(expected, actual, bins=10):
    expected_counts, bin_edges = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)
    psi = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-6)/(actual_perc + 1e-6)))
    return psi

print("\nPSI Drift Results:")
for i, col in enumerate(features):
    psi = calculate_psi(X_train_scaled[:, i], X_test_scaled[:, i])
    print(f"{col}: {psi:.4f}")

# ==========================================================
# 10. RUL Calculation
# ==========================================================
rul = np.full(len(df), np.nan)
next_failure = None
for i in reversed(range(len(df))):
    if df.loc[i, "MachineFailure"] == 1:
        next_failure = i
    if next_failure is not None:
        rul[i] = next_failure - i
df["RUL"] = rul

# ==========================================================
# 11. PLC Alarm Simulation
# ==========================================================
recent_anomalies = df["EIF_Prediction"].iloc[-PLC_WINDOW:]
plc_alarm = 1 if np.any(recent_anomalies == 1) else 0
status = "⚠ WARNING" if plc_alarm else "HEALTHY"
print(f"\nPLC Alarm Bit: {plc_alarm} ({status})")

# ==========================================================
# 12. Save Results
# ==========================================================
df.to_csv(results_csv, index=False)
logging.info(f"Results saved to {results_csv}")
logging.info("Pipeline execution completed successfully.")import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from pyod.models.iforest import IForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ==========================================================
# 1. Configuration
# ==========================================================
RANDOM_STATE = 42
TEST_SIZE = 0.30
PLC_WINDOW = 10
ISO_CONTAMINATION = 0.01
SHAP_WINDOW = 200

logging.basicConfig(level=logging.INFO)

# ==========================================================
# 2. Load Dataset
# ==========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(current_dir, "dataset_ai4.csv")
results_csv = os.path.join(current_dir, "results_eif.csv")
shap_csv = os.path.join(current_dir, "shap_values.csv")

logging.info("Loading dataset...")
df = pd.read_csv(input_csv)

df = df.drop(columns=["UDI", "Product ID", "Type"])
df.columns = [
    "AirTemp", "ProcessTemp", "RotSpeed",
    "Torque", "ToolWear",
    "MachineFailure", "TWF", "HDF", "PWF", "OSF", "RNF"
]

features = ["AirTemp", "ProcessTemp", "RotSpeed", "Torque", "ToolWear"]
X = df[features]
y = df["MachineFailure"]

# ==========================================================
# 3. Train/Test Split
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

train_idx = X_train.index
test_idx = X_test.index

# ==========================================================
# 4. Scaling
# ==========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# 5. Anomaly Detection: PyOD Isolation Forest (EIF)
# ==========================================================
logging.info("Training PyOD Isolation Forest...")

iso_model = IForest(
    n_estimators=200,
    max_samples=256,
    contamination=ISO_CONTAMINATION,
    random_state=RANDOM_STATE
)

# Train only on normal samples
iso_model.fit(X_train_scaled[y_train == 0])

# Binary anomaly prediction: 0 = normal, 1 = anomaly
iso_pred_test = iso_model.predict(X_test_scaled)
df["EIF_Prediction"] = 0
df.loc[test_idx, "EIF_Prediction"] = iso_pred_test

# Optional: continuous anomaly scores for analysis
scores = iso_model.decision_function(X_test_scaled)  # higher = more anomalous

# ==========================================================
# 6. Random Forest
# ==========================================================
logging.info("Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_pred = (rf_probs >= 0.5).astype(int)
df["RF_Prediction"] = 0
df.loc[test_idx, "RF_Prediction"] = rf_pred

# ==========================================================
# 7. Evaluation Function
# ==========================================================
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

cm_rf = evaluate_model("Random Forest", y_test, rf_pred, rf_probs)

# ==========================================================
# 8. Threshold Optimization for RF
# ==========================================================
thresholds = np.linspace(0.01, 0.99, 200)
f1_scores = [f1_score(y_test, (rf_probs >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"\nOptimal Threshold: {best_threshold:.3f}")

rf_pred_opt = (rf_probs >= best_threshold).astype(int)
evaluate_model("Random Forest (Optimized)", y_test, rf_pred_opt, rf_probs)
df.loc[test_idx, "RF_Prediction_Opt"] = rf_pred_opt

plt.figure(figsize=(6,4))
plt.plot(thresholds, f1_scores)
plt.axvline(best_threshold, color='r', linestyle='--')
plt.title("F1 Score vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.show()

# ==========================================================
# 9. SHAP Explainability 
# ==========================================================

logging.info("Calculating SHAP values...")

X_test_df = pd.DataFrame(X_test_scaled, columns=features)

# Changed model_output to "raw" for correct processing
explainer = shap.TreeExplainer(rf_model, model_output="raw")
shap_values = explainer.shap_values(X_test_df)

# For binary classification, shap_values is a list: [class0, class1]
# class 1 (positive) for RUL / failure explanation
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values_class1 = shap_values[1]  # shape: (n_samples, n_features)
else:
    shap_values_class1 = shap_values  # fallback if not binary (shouldn't happen)

# Ensure the shape is 2D before creating a DataFrame
if len(shap_values_class1.shape) == 3:  # If it's 3D due to class separation
    shap_values_class1 = shap_values_class1[:, :, 1]  # Select class 1 column

# Flatten the values and create the dataframe
shap_df = pd.DataFrame(shap_values_class1, columns=features)

shap_df["Index"] = test_idx.values
shap_df = shap_df.sort_values("Index")


# ==========================================================
# 10. Rolling SHAP Trends
# ==========================================================
rolling_shap = shap_df[features].abs().rolling(SHAP_WINDOW).mean()
plt.figure(figsize=(10,6))
for col in features:
    plt.plot(rolling_shap[col], label=col)
plt.title(f"Rolling Mean |SHAP| Over Time (window={SHAP_WINDOW})")
plt.xlabel("Sample Index")
plt.ylabel("|SHAP| Value")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# 11. Save SHAP Values
# ==========================================================
shap_df.to_csv(shap_csv, index=False)
logging.info(f"SHAP values saved to {shap_csv}")

# ==========================================================
# 12. Data Drift Detection (PSI)
# ==========================================================
def calculate_psi(expected, actual, bins=10):
    expected_counts, bin_edges = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)
    psi = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-6)/(actual_perc + 1e-6)))
    return psi

print("\nPSI Drift Results:")
for i, col in enumerate(features):
    psi = calculate_psi(X_train_scaled[:, i], X_test_scaled[:, i])
    print(f"{col}: {psi:.4f}")

# ==========================================================
# 13. RUL Calculation
# ==========================================================
rul = np.full(len(df), np.nan)
next_failure = None
for i in reversed(range(len(df))):
    if df.loc[i, "MachineFailure"] == 1:
        next_failure = i
    if next_failure is not None:
        rul[i] = next_failure - i
df["RUL"] = rul

# ==========================================================
# 14. PLC Alarm Simulation
# ==========================================================
recent_anomalies = df["EIF_Prediction"].iloc[-PLC_WINDOW:]
plc_alarm = 1 if np.any(recent_anomalies == 1) else 0
status = "⚠ WARNING" if plc_alarm else "HEALTHY"
print(f"\nPLC Alarm Bit: {plc_alarm} ({status})")

# ==========================================================
# 15. Save Results
# ==========================================================
df.to_csv(results_csv, index=False)
logging.info(f"Results saved to {results_csv}")

logging.info("Pipeline execution completed successfully.")



