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
shap_csv = os.path.join(current_dir, "shap_values.csv")  # SHAP log-odds values saved from main pipeline

if not os.path.exists(results_csv) or not os.path.exists(shap_csv):
    st.error("❌ Please run main.py first to generate results.csv and shap_values.csv")
    st.stop()

df = pd.read_csv(results_csv)
shap_df_full = pd.read_csv(shap_csv)  # columns: features + "Index"

st.title("Industrial Pump Predictive Maintenance Dashboard")

# --------------------------
# 2. Sidebar Controls
# --------------------------
st.sidebar.header("Display Settings")
window = st.sidebar.slider("Aggregation Window (samples)", 1, 500, 50, 10)
start_idx = st.sidebar.number_input("Zoom Start Index", 0, len(df)-1, 0, 1)
end_idx = st.sidebar.number_input("Zoom End Index", 0, len(df), len(df), 1)

model_choice = st.sidebar.radio("Select Model to Display", ["Isolation Forest", "Random Forest"])
pred_column = "ISO_Prediction" if model_choice == "Isolation Forest" else "RF_Prediction"

critical_threshold = st.sidebar.number_input("Critical Lead-Time Threshold (samples)", 1, 500, 20)

if end_idx <= start_idx:
    st.sidebar.error("End Index must be greater than Start Index")
    st.stop()

# --------------------------
# 3. PLC Alarm Status
# --------------------------
plc_alarm = df["ISO_Prediction"].iloc[start_idx:end_idx].sum() > 0
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
# 8. Rolling SHAP Trends
# --------------------------
st.subheader("Feature Importance Trends (Rolling |SHAP| log-odds)")

# Filter shap_df for zoom window
shap_zoom = shap_df_full[(shap_df_full["Index"] >= start_idx) & (shap_df_full["Index"] < end_idx)].copy()
shap_features = [c for c in shap_zoom.columns if c != "Index"]

# Compute rolling mean |SHAP|
shap_rolling = shap_zoom[shap_features].abs().rolling(window).mean()
shap_rolling["Index"] = shap_zoom["Index"].values

fig3 = go.Figure()
for f in shap_features:
    fig3.add_trace(go.Scatter(
        x=shap_rolling["Index"],
        y=shap_rolling[f],
        mode='lines',
        name=f
    ))
fig3.update_layout(
    xaxis_title="Sample Index",
    yaxis_title="Rolling |SHAP| (log-odds)",
    height=400,
    title="Rolling Feature Contribution Trends"
)
st.plotly_chart(fig3)

# --------------------------
# 9. Metrics Summary
# --------------------------
st.subheader("Pump Metrics")

failure_rate = df["MachineFailure"].mean()
anomaly_count = df["ISO_Prediction"].sum()
st.metric("Failure Rate", f"{failure_rate*100:.2f}%")
st.metric("Total Anomalies Detected", anomaly_count)

precision_iso = ((df['ISO_Prediction'] & df['MachineFailure']).sum() /
                 max(df['ISO_Prediction'].sum(),1))
recall_iso = ((df['ISO_Prediction'] & df['MachineFailure']).sum() /
              max(df['MachineFailure'].sum(),1))

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
# 10. Info
# --------------------------
st.markdown("""
- **Lead-Time Markers (green/orange):** first ISO anomaly before failure; orange = critical early warning.  
- **Rolling SHAP Trends:** feature importance over time (log-odds contributions).  
- **RUL:** estimated Remaining Useful Life per sample.  
- **PLC Alarm:** triggered if any ISO anomaly detected.  
- **Zoom & Aggregation:** focus on specific segments and smooth anomaly signals.  
""")
