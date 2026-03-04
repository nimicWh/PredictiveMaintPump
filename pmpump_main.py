import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# --------------------------
# 1. Paths
# --------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(current_dir, "dataset_ai4.csv")
output_csv = os.path.join(current_dir, "results.csv")

# --------------------------
# 2. Load dataset
# --------------------------
df = pd.read_csv(input_csv)

# Drop unnecessary columns
df = df.drop(columns=["UDI", "Product ID", "Type"])
df.columns = [
    "AirTemp", "ProcessTemp", "RotSpeed",
    "Torque", "ToolWear",
    "MachineFailure", "TWF", "HDF", "PWF", "OSF", "RNF"
]

features = ["AirTemp", "ProcessTemp", "RotSpeed", "Torque", "ToolWear"]
X = df[features]
y = df["MachineFailure"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# 3. Isolation Forest (unsupervised)
# --------------------------
print("\nTraining Isolation Forest (unsupervised)...")
X_train_iso = X_scaled[y == 0]  # train only on healthy data
iso_model = IsolationForest(contamination=0.02, random_state=42)
iso_model.fit(X_train_iso)

iso_pred = iso_model.predict(X_scaled)
iso_pred = np.where(iso_pred == -1, 1, 0)  # -1 = anomaly -> 1

# --------------------------
# 4. Random Forest (supervised)
# --------------------------
print("Training Random Forest (supervised)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)
rf_pred = rf_model.predict(X_scaled)

# --------------------------
# 5. Evaluate Models
# --------------------------
def evaluate_model(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{name} Results")
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
    return cm, precision, recall

cm_iso, precision_iso, recall_iso = evaluate_model("Isolation Forest", y, iso_pred)
cm_rf, precision_rf, recall_rf = evaluate_model("Random Forest", y, rf_pred)

# Plot RF confusion matrix
plt.figure(figsize=(8,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.show()

# --------------------------
# 6. Lead-Time Before Failure
# --------------------------
print("\nCalculating Lead-Time Before Failure...")
lead_times = []

failure_indices = df.index[df["MachineFailure"] == 1]

for idx in failure_indices:
    prior_anomalies = df.loc[:idx-1].index[iso_pred[:idx] == 1]
    if len(prior_anomalies) > 0:
        lead_time = idx - prior_anomalies[-1]
        lead_times.append(lead_time)

if lead_times:
    avg_lead = np.mean(lead_times)
    print(f"Average Lead-Time Before Failure (samples): {avg_lead:.2f}")
else:
    print("No early anomalies detected before failures.")

# --------------------------
# 7. MTBF Calculation
# --------------------------
min_gap = 50  # minimum gap to separate distinct failure events
events = []
prev = -min_gap*2
for idx in failure_indices:
    if idx - prev > min_gap:
        events.append(idx)
    prev = idx

if len(events) > 1:
    intervals = [events[i+1] - events[i] for i in range(len(events)-1)]
    mtbf = np.mean(intervals)
    print(f"Mean Time Between Failures (MTBF): {mtbf:.2f} samples")
else:
    mtbf = None
    print("Not enough failures to calculate MTBF.")

# --------------------------
# 8. RUL Estimation
# --------------------------
rul = []
for i in df.index:
    future_failures = [f for f in failure_indices if f >= i]
    if future_failures:
        rul.append(future_failures[0] - i)
    else:
        rul.append(None)

df["RUL"] = rul

# --------------------------
# 9. PLC Alarm Simulation
# --------------------------
plc_alarm = 1 if np.any(iso_pred == 1) else 0
status = "⚠ WARNING" if plc_alarm else "HEALTHY"
print(f"\nPLC Alarm Bit: {plc_alarm} ({status})")

# --------------------------
# 10. Save Results
# --------------------------
df["ISO_Prediction"] = iso_pred
df["RF_Prediction"] = rf_pred
df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")
