import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from ui.common import csv, meta, score, risk_map, pct, base_layout, ALERT

dis = st.session_state.get("disease", "Dengue")
last = meta()["LAST_WEEK"][dis]
clim = csv("weekly_climate.csv").set_index("mw")
L = csv(f"latest_features_{dis}.csv")

st.title("Rainfall scenario")
st.caption(f"Change the weather for the four weeks up to MW {last} and see how next-week {dis.lower()} risk would shift. "
           "Case counts stay as observed; only the climate inputs change.")

obs = {"r37": float(clim.loc[last - 3, "rain"]), "r38": float(clim.loc[last - 2, "rain"]), "r39": float(clim.loc[last - 1, "rain"]),
       "r40": float(clim.loc[last, "rain"]), "rh": float(clim.loc[last - 1, "rh"]), "tm": float(clim.loc[last - 1, "tmean"]), "hd": int(clim.loc[last, "heavy_days"])}
PRESETS = {
    "Observed": obs,
    "Monsoon surge (like MW 29–32)": {"r37": 48.5, "r38": 73.5, "r39": 175.1, "r40": 174.3, "rh": 85.9, "tm": 27.2, "hd": 5},
    "Typhoon week": {"r37": 40.0, "r38": 60.0, "r39": 120.0, "r40": 300.0, "rh": 90.0, "tm": 26.5, "hd": 5},
    "Dry spell": {"r37": 5.0, "r38": 5.0, "r39": 3.0, "r40": 2.0, "rh": 60.0, "tm": 30.5, "hd": 0},
}
def apply(name):
    for k, v in PRESETS[name].items(): st.session_state[f"wi_{k}"] = v
for k, v in obs.items(): st.session_state.setdefault(f"wi_{k}", v)

cols = st.columns(len(PRESETS))
for c, name in zip(cols, PRESETS):
    c.button(name, on_click=apply, args=(name,), width="stretch")

s1, s2, s3, s4 = st.columns(4)
r37 = s1.slider(f"Rain, MW {last - 3} (mm)", 0.0, 400.0, key="wi_r37", step=5.0)
r38 = s2.slider(f"Rain, MW {last - 2} (mm)", 0.0, 400.0, key="wi_r38", step=5.0)
r39 = s3.slider(f"Rain, MW {last - 1} (mm)", 0.0, 400.0, key="wi_r39", step=5.0)
r40 = s4.slider(f"Rain, MW {last} (mm)", 0.0, 400.0, key="wi_r40", step=5.0)
t1, t2, t3 = st.columns(3)
rh = t1.slider(f"Humidity, MW {last - 1} (%)", 50.0, 95.0, key="wi_rh", step=0.5)
tm = t2.slider(f"Mean temperature, MW {last - 1} (°C)", 24.0, 33.0, key="wi_tm", step=0.1)
hd = t3.slider(f"Heavy-rain days (≥20 mm), MW {last}", 0, 7, key="wi_hd")

X = L.copy()
X["rain_l1"], X["rain_l2"], X["rain_l3"] = r40, r39, r38
X["rain_4wk_l1"] = r37 + r38 + r39 + r40
X["rh_l2"], X["tmean_l2"] = rh, tm
p_new, contrib = score(dis, X)
p_base = L.risk.values
X["risk_new"], X["risk"], X["change"] = p_new, p_base, p_new - p_base
area = csv("barangay_risk.csv").query("disease == @dis").set_index("brgy").area
X["area"] = X.brgy.map(area)

if hd >= 3:
    st.error(f"**Rainfall trigger met** ({hd} heavy-rain days). Issue a district leptospirosis alert for the next two weeks, whatever the model scores.", icon=":material/rainy:")
max4 = csv("weekly_climate.csv").set_index("mw").rain.rolling(4).sum().max()
if r37 + r38 + r39 + r40 > max4:
    st.info(f"4-week rainfall of {r37 + r38 + r39 + r40:.0f} mm is above anything in the training data ({max4:.0f} mm). "
            "Tree models cannot extrapolate, so risk is capped at the level learned for the wettest weeks.", icon=":material/info:")

k1, k2, k3 = st.columns(3)
k1.metric("Mean barangay risk", pct(p_new.mean()), f"{(p_new.mean() - p_base.mean()) * 100:+.1f} pts vs observed", delta_color="inverse")
k2.metric("Barangays above 20%", int((p_new > 0.2).sum()), f"{int((p_new > 0.2).sum() - (p_base > 0.2).sum()):+d}", delta_color="inverse")
k3.metric("Expected barangays with a case", f"{p_new.sum():.1f}", f"{p_new.sum() - p_base.sum():+.1f}", delta_color="inverse")

a, b = st.columns([3, 2])
with a:
    st.plotly_chart(risk_map(X.assign(driver=""), "risk_new", "Scenario risk", scale="Blues" if dis == "Dengue" else "Reds",
                             rng=(0, max(float(p_new.max()), float(p_base.max()))), height=480), width="stretch")
with b:
    st.markdown("**Largest increases**")
    t = X.sort_values("change", ascending=False).head(12)[["brgy", "risk", "risk_new", "change"]].copy()
    for c in ["risk", "risk_new", "change"]: t[c] = t[c] * 100
    st.dataframe(t, hide_index=True, width="stretch", height=440, column_config={
        "brgy": "Barangay", "risk": st.column_config.NumberColumn("Observed", format="%.1f%%"),
        "risk_new": st.column_config.NumberColumn("Scenario", format="%.1f%%"),
        "change": st.column_config.NumberColumn("Change (pts)", format="%.1f")})

F = meta()["BFEATS"]; clim_idx = [F.index(f) for f in ["rain_l1", "rain_l2", "rain_l3", "rain_4wk_l1", "rh_l2", "tmean_l2"]]
share = np.abs(contrib[:, clim_idx]).sum(1) / np.abs(contrib[:, :-1]).sum(1)
st.caption(f"Under this scenario, weather accounts for {share.mean():.0%} of the model's total evidence on average (SHAP). "
           "The model was trained on one season, so treat large shifts as direction, not precise probabilities.")
