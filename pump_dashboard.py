import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------
# 1. Paths
# --------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
results_csv = os.path.join(current_dir, "results.csv")

if not os.path.exists(results_csv):
    st.error("❌ Please run main.py first to generate results.csv")
    st.stop()

df = pd.read_csv(results_csv)

st.title("Industrial Pump Predictive Maintenance Dashboard")

# --------------------------
# 2. Sidebar Controls
# --------------------------
st.sidebar.header("Display Settings")
window = st.sidebar.slider("Aggregation Window (samples)", min_value=1, max_value=500, value=50, step=10)
start_idx = st.sidebar.number_input("Zoom Start Index", min_value=0, max_value=len(df)-1, value=0, step=1)
end_idx = st.sidebar.number_input("Zoom End Index", min_value=0, max_value=len(df), value=len(df), step=1)

model_choice = st.sidebar.radio("Select Model to Display", ["Isolation Forest", "Random Forest"])
pred_column = "ISO_Prediction" if model_choice == "Isolation Forest" else "RF_Prediction"

critical_threshold = st.sidebar.number_input("Critical Lead-Time Threshold (samples)", min_value=1, max_value=500, value=20)

if end_idx <= start_idx:
    st.sidebar.error("End Index must be greater than Start Index")
    st.stop()

# --------------------------
# 3. PLC Alarm Status
# --------------------------
plc_alarm = df["ISO_Prediction"].sum() > 0
if plc_alarm:
    st.error("⚠ Warning: Potential Pump Degradation Detected (PLC Alarm Bit = 1)")
else:
    st.success("Pump Operating Normally (PLC Alarm Bit = 0)")

# --------------------------
# 4. Aggregate and Zoom Data
# --------------------------
df_agg = df[[pred_column, "MachineFailure"]].rolling(window=window, min_periods=1).max()
df_zoom = df_agg.iloc[start_idx:end_idx]

# --------------------------
# 5. Interactive Step Plot with Lead-Time Markers
# --------------------------
st.subheader(f"{model_choice} Predictions vs Actual Failures (Interactive Lead-Time)")

failure_indices = df.index[df["MachineFailure"] == 1]
lead_time_list = []

fig = go.Figure()

# Anomalies
fig.add_trace(go.Scatter(
    x=df_zoom.index,
    y=df_zoom[pred_column],
    mode='lines',
    name=f"{model_choice} Anomaly",
    line=dict(color='blue'),
    fill='tozeroy',
    hoverinfo='x+y'
))

# Actual failures
fig.add_trace(go.Scatter(
    x=df_zoom.index,
    y=df_zoom["MachineFailure"],
    mode='lines+markers',
    name='Actual Failure',
    line=dict(color='red'),
    hoverinfo='x+y'
))

# Lead-time markers
for idx in failure_indices:
    prior_anomalies = df.index[(df.index < idx) & (df["ISO_Prediction"] == 1)]
    if len(prior_anomalies) > 0:
        first_anomaly = prior_anomalies[-1]
        lead_time = idx - first_anomaly
        lead_time_list.append({
            "Failure Sample Index": idx,
            "First ISO Anomaly Index": first_anomaly,
            "Lead-Time (samples)": lead_time
        })
        if start_idx <= first_anomaly < end_idx:
            color = 'green' if lead_time <= critical_threshold else 'orange'
            fig.add_trace(go.Scatter(
                x=[first_anomaly],
                y=[1.05],
                mode='markers+text',
                marker=dict(size=10, color=color),
                text=[f"Lead {lead_time}"],
                textposition='top center',
                name='Lead-Time Marker',
                hovertemplate=f"Failure at {idx}<br>First anomaly: {first_anomaly}<br>Lead-Time: {lead_time} samples"
            ))
    else:
        lead_time_list.append({
            "Failure Sample Index": idx,
            "First ISO Anomaly Index": None,
            "Lead-Time (samples)": None
        })

fig.update_layout(
    xaxis_title="Sample Index",
    yaxis_title="Status (0=Normal, 1=Anomaly/Failure)",
    title=f"{model_choice} Anomaly Detection with Lead-Time Markers",
    hovermode="closest",
    height=400
)

st.plotly_chart(fig)

# --------------------------
# 6. Lead-Time Summary Table
# --------------------------
st.subheader("Lead-Time Summary for Failures")
lead_time_df = pd.DataFrame(lead_time_list)
st.dataframe(lead_time_df)

# --------------------------
# 7. Remaining Useful Life (RUL) Plot
# --------------------------
st.subheader("Remaining Useful Life (RUL)")
rul_df = df[['RUL']].dropna().iloc[start_idx:end_idx]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=rul_df.index,
    y=rul_df['RUL'],
    mode='lines',
    line=dict(color='green'),
    name='RUL (samples)',
    hoverinfo='x+y'
))
fig2.update_layout(
    xaxis_title="Sample Index",
    yaxis_title="RUL (samples)",
    title="Remaining Useful Life Estimation",
    height=300
)
st.plotly_chart(fig2)

# --------------------------
# 8. Metrics Summary
# --------------------------
st.subheader("Pump Metrics")

failure_rate = df["MachineFailure"].mean()
anomaly_count = df["ISO_Prediction"].sum()
st.metric("Failure Rate", f"{failure_rate*100:.2f}%")
st.metric("Total Anomalies Detected", anomaly_count)

# ISO Precision & Recall
precision_iso = (df['ISO_Prediction'] & df['MachineFailure']).sum() / df['ISO_Prediction'].sum() if df['ISO_Prediction'].sum() > 0 else 0
recall_iso = (df['ISO_Prediction'] & df['MachineFailure']).sum() / df['MachineFailure'].sum() if df['MachineFailure'].sum() > 0 else 0

st.write(f"**Isolation Forest Precision:** {precision_iso:.3f}")
st.write(f"**Isolation Forest Recall:** {recall_iso:.3f}")

# MTBF
min_gap = 50
events = []
prev = -min_gap*2
for idx in failure_indices:
    if idx - prev > min_gap:
        events.append(idx)
    prev = idx

if len(events) > 1:
    intervals = [events[i+1]-events[i] for i in range(len(events)-1)]
    mtbf = sum(intervals)/len(intervals)
    st.write(f"**Mean Time Between Failures (MTBF):** {mtbf:.2f} samples")
else:
    st.write("MTBF: Not enough failures to calculate.")

# --------------------------
# 9. Info
# --------------------------
st.markdown("""
- **Lead-Time Markers (green/orange):** first ISO anomaly before failure; orange = critical early warning.  
- **Aggregation Window:** rolling max to reduce noise in binary signals.  
- **Zoom:** inspect specific segments of pump data.  
- **RUL:** estimated Remaining Useful Life per sample.  
- **PLC Alarm:** triggered if any ISO anomaly detected.
""")