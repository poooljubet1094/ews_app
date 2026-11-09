import pandas as pd, plotly.graph_objects as go, streamlit as st
from ui.common import csv, meta, risk_map, score, shap_bar, pct, COLORS, base_layout

dis = st.session_state.get("disease", "Dengue")
last = meta()["LAST_WEEK"][dis]
risk = csv("barangay_risk.csv").query("disease == @dis").copy()

st.title(f"Barangay risk for MW {last + 1}")
st.caption(f"Predicted probability that each District VI barangay reports at least one {dis.lower()} case next week. "
           "Outlined barangays form the ten-barangay watch list.")

col_map, col_tab = st.columns([3, 2])
with col_map:
    st.plotly_chart(risk_map(risk, "risk", "Risk", scale="Blues" if dis == "Dengue" else "Reds"), width="stretch")
with col_tab:
    areas = ["All areas"] + sorted(a for a in risk.area.dropna().unique() if a)
    area = st.selectbox("Area", areas)
    view = risk if area == "All areas" else risk[risk.area == area]
    view = view.sort_values("rank")[["rank", "brgy", "area", "risk", "cases_last4", "cases_ytd", "driver"]].copy()
    view["risk"] = view.risk * 100
    st.dataframe(view, hide_index=True, width="stretch", height=470, column_config={
        "rank": st.column_config.NumberColumn("Rank", width="small"),
        "brgy": "Barangay", "area": "Area",
        "risk": st.column_config.ProgressColumn("Risk", format="%.1f%%", min_value=0, max_value=max(view.risk.max(), 1)),
        "cases_last4": st.column_config.NumberColumn("Cases, 4 wk"),
        "cases_ytd": st.column_config.NumberColumn("Cases, 2026"),
        "driver": "Main driver"})

st.divider()
st.subheader("Barangay profile")
pick = st.selectbox("Barangay", risk.sort_values("rank").brgy.tolist())
row = risk[risk.brgy == pick].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk next week", pct(row.risk)); c2.metric("Rank in District VI", f"{int(row['rank'])} of {len(risk)}")
c3.metric("Cases, last 4 weeks", int(row.cases_last4)); c4.metric("Cases, 2026", int(row.cases_ytd))

hist = csv("barangay_weekly_d6.csv").query("disease == @dis and brgy == @pick").sort_values("mw")
L = csv(f"latest_features_{dis}.csv")
_, contrib = score(dis, L[L.brgy == pick])
a, b = st.columns(2)
with a:
    fig = go.Figure(go.Bar(x=hist.mw, y=hist.cases, marker_color=COLORS[dis], hovertemplate="MW %{x}: %{y}<extra></extra>"))
    fig.update_layout(title="Weekly cases", xaxis_title="Morbidity week", yaxis_title="Cases")
    st.plotly_chart(base_layout(fig, 320), width="stretch")
with b:
    st.plotly_chart(shap_bar(contrib[0]).update_layout(title="Why this score (SHAP)"), width="stretch")
st.caption("Barangay names are shown for CESU planning. Do not publish rankings: labelling hotspots can stigmatise residents.")
