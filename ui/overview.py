import streamlit as st
from ui.common import csv, meta, epi_chart, pct, week_label

dis = st.session_state.get("disease", "Dengue")
m = meta()
last = m["LAST_WEEK"][dis]
dw = csv("district_weekly.csv")
d6 = dw[(dw.disease == dis) & (dw.District == "VI")].set_index("mw")
risk = csv("barangay_risk.csv").query("disease == @dis")
clim = csv("weekly_climate.csv").set_index("mw")
thr = float(d6.threshold.iloc[0])

st.title(f"{dis} early warning, District VI")
y_now, y_prev = int(d6.y.loc[last]), int(d6.y.loc[last - 1])
above = y_now > thr
top = risk.nsmallest(1, "rank").iloc[0]
watch = risk.nsmallest(10, "rank")
share = watch.cases_last4.sum() / max(risk.cases_last4.sum(), 1)
heavy = int(clim.loc[last, "heavy_days"])

if above:
    st.error(f"**Above epidemic threshold.** {y_now} cases in {week_label(last)} against a threshold of {thr:.1f}.", icon=":material/warning:")
else:
    st.success(f"**Below epidemic threshold.** {y_now} cases in {week_label(last)} against a threshold of {thr:.1f}.", icon=":material/check_circle:")
if heavy >= 3:
    st.warning(f"**Rainfall trigger met:** {heavy} heavy-rain days in {week_label(last)}. Expect leptospirosis to rise in 1–2 weeks.", icon=":material/rainy:")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Cases, MW {last}", y_now, f"{y_now - y_prev:+d} vs MW {last - 1}", delta_color="inverse")
peak_mw = int(d6.y.idxmax())
c2.metric(f"Season peak (MW {peak_mw})", int(d6.y.max()))
c3.metric(f"Highest risk, MW {last + 1}: {top.brgy}", pct(top.risk))
c4.metric(f"Watch list share of cases, last 4 wk ({int(watch.cases_last4.sum())} of {int(risk.cases_last4.sum())})", f"{share:.0%}")

st.subheader("Weekly cases and rainfall")
st.caption("Bars turn red above the epidemic threshold (mean + 2 SD of MW 1–26, minimum 3 cases).")
st.plotly_chart(epi_chart(dis), width="stretch")

left, right = st.columns([3, 2])
with left:
    st.subheader("All districts, last 8 weeks")
    piv = dw[(dw.disease == dis) & (dw.mw > last - 8)].pivot(index="District", columns="mw", values="y")
    thr_all = dw[dw.disease == dis].groupby("District").threshold.first()
    piv.columns = [f"MW {c}" for c in piv.columns]
    piv["Threshold"] = thr_all.round(1)
    st.dataframe(piv, width="stretch")
with right:
    st.subheader("What to do this week")
    if dis == "Leptospirosis":
        st.markdown("- Watch rainfall: a week with **3 or more days of ≥20 mm** has preceded every surge this year.\n"
                    "- Pre-position doxycycline for flood-exposed residents in the watch-list barangays.\n"
                    "- Remind health centers to ask about wading or flood exposure in febrile patients.")
    else:
        st.markdown("- Prioritise search-and-destroy and fogging in the **10 watch-list barangays**.\n"
                    "- Check that health centers have NS1 kits and IV fluids for the next two weeks.\n"
                    "- Re-check rankings after each weekly PIDSR upload.")
    st.caption("Recommendations are decision support, not orders. Validate with field reports.")
