import pandas as pd, plotly.graph_objects as go, streamlit as st
from ui.common import csv, base_layout, COLORS

st.title("Forecasts and model comparison")
st.caption("Rolling-origin evaluation: each week every model was retrained on earlier weeks only, then forecast 1 or 2 weeks ahead.")

c1, c2 = st.columns(2)
dis = c1.segmented_control("Disease", ["Dengue", "Leptospirosis", "ILI"], default=st.session_state.get("disease", "Dengue"), key="fc_dis") or "Dengue"
h = c2.segmented_control("Horizon (weeks ahead)", [1, 2], default=1, key="fc_h") or 1

P = csv("d6_rolling_predictions.csv").query("dis == @dis and h == @h")
models = P.model.unique().tolist()
chosen = st.multiselect("Models to show", models, default=[m for m in ["Naive (persistence)", "Random forest", "XGBoost (Poisson)"] if m in models])
obs = P[P.model == models[0]].sort_values("target_mw")
fig = go.Figure()
fig.add_scatter(x=obs.target_mw, y=obs.y, name="Observed", mode="lines+markers", line=dict(color="#111", width=2.5))
palette = ["#8A94A3", COLORS.get(dis, "#2F6DB5"), "#C9A227", "#6B4FA0", "#2E8B57"]
for i, mn in enumerate(chosen):
    s = P[P.model == mn].sort_values("target_mw")
    fig.add_scatter(x=s.target_mw, y=s.pred, name=mn, mode="lines", line=dict(width=2, dash="dot" if "Naive" in mn else "solid", color=palette[i % 5]))
fig.add_hline(y=float(obs.thr.iloc[0]), line=dict(color="#C9A227", dash="dash"), annotation_text="Epidemic threshold")
fig.update_layout(xaxis_title="Target morbidity week", yaxis_title="Cases, District VI")
st.plotly_chart(base_layout(fig, 400), width="stretch")

st.subheader("All six districts")
R = csv("table_forecast_metrics.csv").query("Disease == @dis and Horizon == @h").drop(columns=["Disease", "Horizon"])
st.dataframe(R.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "Sensitivity": "{:.2f}", "Precision": "{:.2f}", "F1": "{:.2f}"})
             .highlight_min(subset=["MAE", "RMSE"], color="#d6ead3").highlight_max(subset=["F1"], color="#d6ead3"),
             hide_index=True, width="stretch")
st.caption("MAE/RMSE: forecast error in cases per district-week. Sensitivity, precision and F1 score the outbreak flag (forecast above threshold).")

a, b = st.columns(2)
with a:
    st.subheader("District VI only, incl. SARIMAX")
    S = csv("table_d6_metrics.csv").query("Disease == @dis and Horizon == @h").drop(columns=["Disease", "Horizon"]).sort_values("MAE")
    st.dataframe(S.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}), hide_index=True, width="stretch")
with b:
    st.subheader("Alert timing (h = 1)")
    LS = csv("table_leadtime_summary.csv").query("Disease == @dis").drop(columns=["Disease"])
    LS.columns = ["Method", "Onset week (median)", "Median delay (wk)", "Early or on time", "Districts"]
    st.dataframe(LS, hide_index=True, width="stretch")
    st.caption("Negative delay = alert before outbreak onset.")

st.subheader("Barangay risk models (next-week case, District VI, MW 21–40)")
BR = csv("table_brgy_metrics.csv")
st.dataframe(BR.style.format({"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Brier": "{:.3f}", "Top-10 capture": "{:.1%}", "Prevalence": "{:.1%}"}),
             hide_index=True, width="stretch")
st.info("Naive persistence is hard to beat on weekly counts. Machine learning adds value in ranking barangays "
        "and in dengue outbreak flagging. No model foresaw the first leptospirosis surge, because nothing like it was in the training weeks.",
        icon=":material/lightbulb:")
