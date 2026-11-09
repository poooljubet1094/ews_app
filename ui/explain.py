import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from ui.common import csv, base_layout, COLORS

dis = st.session_state.get("disease", "Dengue")
st.title("Explainability")
st.caption("SHAP values show how much each input moves a prediction. Mean absolute SHAP ranks inputs by overall influence.")

SH = csv("table_shap.csv")
a, b = st.columns(2)
for col, lvl, unit in [(a, "district", "log-count"), (b, "barangay", "log-odds")]:
    g = SH[(SH.Disease == dis) & (SH.level == lvl)].nlargest(10, "mean_abs_shap").iloc[::-1]
    fig = go.Figure(go.Bar(x=g.mean_abs_shap, y=g.Feature, orientation="h",
                           marker_color=[COLORS[dis] if "rain" in f.lower() or "humid" in f.lower() or "temp" in f.lower() else "#9AA5B4" for f in g.Feature],
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title=f"{lvl.capitalize()} model", xaxis_title=f"Mean |SHAP| ({unit})")
    col.plotly_chart(base_layout(fig, 380).update_layout(hovermode="closest"), width="stretch")
st.caption("Coloured bars are weather inputs; grey bars are case-history inputs.")

if dis == "Leptospirosis":
    st.markdown("**Reading:** rainfall over the previous two to four weeks dominates leptospirosis risk, matching the 1–2 week "
                "incubation after flood exposure. Recent case counts matter much less.")
else:
    st.markdown("**Reading:** dengue risk follows case momentum and place: where cases have been and how many there are now. "
                "Weather enters through 4-week rainfall and humidity lagged about three weeks, the mosquito-breeding pathway.")

st.subheader("Rainfall lead time")
dw = csv("district_weekly.csv"); clim = csv("weekly_climate.csv").set_index("mw")
fig = go.Figure()
for d in ["Dengue", "Leptospirosis", "ILI"]:
    y = dw[dw.disease == d].groupby("mw").y.sum()
    r = [np.corrcoef(np.log1p(y.values), clim.rain.shift(L).loc[y.index].values)[0, 1] for L in range(5)]
    fig.add_scatter(x=list(range(5)), y=r, name=d, mode="lines+markers", line=dict(color=COLORS[d], width=2.5))
fig.update_layout(xaxis_title="Rainfall lag (weeks)", yaxis_title="Correlation with log weekly cases (city)", yaxis_range=[0, 1])
st.plotly_chart(base_layout(fig, 340), width="stretch")
st.caption("Correlations over one season are inflated by shared seasonality; they are descriptive, not causal.")
