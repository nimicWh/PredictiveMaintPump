import os
import logging
import numpy as np
import pandas as pd
import rrcf
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ==========================================================
# 1. Configuration
# ==========================================================
RANDOM_STATE = 42
TEST_SIZE = 0.30
PLC_WINDOW = 10
ISO_CONTAMINATION = 0.01

NUM_TREES = 50        # Number of trees in forest
TREE_SIZE = 128       # Subsample size per tree
SCORING_TREES = 20    # Number of trees used for scoring

logging.basicConfig(level=logging.INFO)

# ==========================================================
# 2. Load Dataset
# ==========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(current_dir, "dataset_ai4.csv")
results_csv = os.path.join(current_dir, "results_rrcf_only.csv")

logging.info("Loading dataset...")
df = pd.read_csv(input_csv)
df = df.drop(columns=["UDI", "Product ID", "Type"])
df.columns = [
    "AirTemp", "ProcessTemp", "RotSpeed", "Torque", "ToolWear",
    "MachineFailure", "TWF", "HDF", "PWF", "OSF", "RNF"
]

features = ["AirTemp", "ProcessTemp", "RotSpeed", "Torque", "ToolWear"]
X = df[features]
y = df["MachineFailure"]

# ==========================================================
# 3. Train/Test Split
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
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
# 5. Build RRCF Forest
# ==========================================================
logging.info("Training RRCF forest in parallel...")

normal_points = X_train_scaled[y_train == 0]

def build_tree(_):
    tree = rrcf.RCTree()
    idxs = np.random.choice(len(normal_points), size=min(TREE_SIZE, len(normal_points)), replace=False)
    for i in idxs:
        tree.insert_point(tuple(normal_points[i].tolist()), index=i)
    return tree

forest = Parallel(n_jobs=-1)(delayed(build_tree)(i) for i in range(NUM_TREES))
logging.info("Forest training completed.")

# ==========================================================
# 6. Compute anomaly scores for test set
# ==========================================================
logging.info("Computing anomaly scores...")

subset_trees = np.random.choice(forest, size=min(SCORING_TREES, len(forest)), replace=False)

def safe_avg_codisp(point):
    row_tuple = tuple(point.tolist())
    codisp_list = [tree.codisp(row_tuple) for tree in subset_trees if row_tuple in tree.leaves]
    return np.mean(codisp_list) if codisp_list else 0

scores = np.array([safe_avg_codisp(pt) for pt in X_test_scaled])

threshold = np.quantile(scores, 1 - ISO_CONTAMINATION)
rrcf_pred_test = (scores >= threshold).astype(int)

df["RRCF_Prediction"] = 0
df.loc[test_idx, "RRCF_Prediction"] = rrcf_pred_test

# ==========================================================
# 7. Evaluation
# ==========================================================
def evaluate_model(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{name} Results\nConfusion Matrix:\n{cm}\nPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    return cm

evaluate_model("RRCF", y_test, rrcf_pred_test)

# ==========================================================
# 8. Data Drift Detection (PSI)
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
# 9. RUL Calculation
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
# 10. PLC Alarm Simulation
# ==========================================================
recent_anomalies = df["RRCF_Prediction"].iloc[-PLC_WINDOW:]
plc_alarm = 1 if np.any(recent_anomalies == 1) else 0
status = "⚠ WARNING" if plc_alarm else "HEALTHY"
print(f"\nPLC Alarm Bit: {plc_alarm} ({status})")

# ==========================================================
# 11. Save Results
# ==========================================================
df.to_csv(results_csv, index=False)
logging.info(f"Results saved to {results_csv}")
logging.info("RRCF-only pipeline execution completed successfully.")
