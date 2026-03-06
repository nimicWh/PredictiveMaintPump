import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import rrcf
from joblib import Parallel, delayed

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

NUM_TREES = 50       # Number of trees in forest
TREE_SIZE = 128      # Subsample size per tree
SCORING_TREES = 20   # Number of trees used for scoring

logging.basicConfig(level=logging.INFO)

# ==========================================================
# 2. Load Dataset
# ==========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(current_dir, "dataset_ai4.csv")
results_csv = os.path.join(current_dir, "results_rrcf_safe.csv")

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

# ==============================================================================
# 5. Random Cut Forest (RRCF) - Parallel + Subsampling, Robust Random Cut Forest
# ==============================================================================
logging.info("Training Random Cut Forest (RRCF) with parallel trees…")

normal_points = X_train_scaled[y_train == 0]

def build_tree(tree_idx):
    tree = rrcf.RCTree()
    # Subsample points
    idxs = np.random.choice(len(normal_points), size=min(TREE_SIZE, len(normal_points)), replace=False)
    for i in idxs:
        row = tuple(normal_points[i].tolist())  # convert to tuple
        tree.insert_point(row, index=i)
    return tree

# Build forest in parallel
forest = Parallel(n_jobs=-1)(delayed(build_tree)(i) for i in range(NUM_TREES))
logging.info("Forest training completed.")

# ==========================================================
# 6. Compute anomaly scores safely
# ==========================================================
logging.info("Computing anomaly scores for test set safely…")
subset_trees = np.random.choice(forest, size=min(SCORING_TREES, len(forest)), replace=False)

# Compute anomaly scores for each test sample
def safe_avg_codisp(point):
    row_tuple = tuple(point.tolist())
    codisp_list = []
    for tree in subset_trees:
        if row_tuple in tree.leaves:
            codisp_list.append(tree.codisp(row_tuple))
    return np.mean(codisp_list) if codisp_list else 0  # default 0 if unseen in all trees

scores = np.array([safe_avg_codisp(pt) for pt in X_test_scaled])

# Convert scores to binary labels based on contamination
threshold = np.quantile(scores, 1 - ISO_CONTAMINATION)
rcf_pred_test = (scores >= threshold).astype(int)

df["RCF_Prediction"] = 0
df.loc[test_idx, "RCF_Prediction"] = rcf_pred_test

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

# Evaluate RRCF results
cm_rcf = evaluate_model("Random Cut Forest", y_test, rcf_pred_test)

# ==========================================================
# 8. Threshold Optimization for RRCF (if needed)
# ==========================================================
# NA

# ==========================================================
# 9. SHAP Explainability (Not Applicable to RRCF)
# ==========================================================
# NA
# ==========================================================
# 10. Data Drift Detection (PSI)
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
# 11. RUL Calculation
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
# 12. PLC Alarm Simulation
# ==========================================================
recent_anomalies = df["RCF_Prediction"].iloc[-PLC_WINDOW:]
plc_alarm = 1 if np.any(recent_anomalies == 1) else 0
status = "⚠ WARNING" if plc_alarm else "HEALTHY"
print(f"\nPLC Alarm Bit: {plc_alarm} ({status})")

# ==========================================================
# 13. Save Results
# ==========================================================
df.to_csv(results_csv, index=False)
logging.info(f"Results saved to {results_csv}")

logging.info("Pipeline execution completed successfully.")
