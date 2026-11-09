import streamlit as st
from ui.common import csv

st.title("Method and ethics")
st.markdown("""
**Study.** An explainable AI early warning system for notifiable diseases in District VI, City of Manila
(Paco part, Pandacan, Sampaloc part, San Miguel, Santa Ana). Professional Science Master in Data Science,
Technological Institute of the Philippines.

**Data**
- CESU Manila PIDSR case line list, 2026, Morbidity Weeks 1–41, 15 notifiable diseases. Personal identifiers are removed
  before analysis; this app holds only barangay-week and district-week counts.
- Daily weather for central Manila (14.59°N, 121.00°E) from the Open-Meteo historical archive (ERA5 reanalysis), summed or averaged by morbidity week.
- Barangay boundaries from the PSGC-coded *philippines-json-maps* dataset.

**Models**
- District counts: naive persistence, 4-week moving average, negative binomial GLM, random forest, XGBoost (Poisson); SARIMAX(1,1,0) with climate for District VI.
- Barangay risk: XGBoost classifier for "at least one case next week", trained on all Manila barangays, evaluated on District VI.
- Evaluation: rolling origin from MW 20; MAE, RMSE, outbreak sensitivity/precision/F1, ROC-AUC, PR-AUC, Brier, top-10 capture, alert lead time.
- Explanations: SHAP (TreeSHAP via XGBoost's built-in contributions).
- Epidemic threshold: mean + 2 SD of weekly counts in MW 1–26 per district, minimum 3.
- Rainfall trigger: a week with 3 or more days of ≥20 mm rain raises a two-week leptospirosis alert.
""")
st.subheader("Records by disease")
t = csv("table_disease_summary.csv").rename(columns={"disease_all": "Disease", "cases": "Manila", "d6": "District VI", "deaths": "Deaths", "med_age": "Median age", "male": "% male"})
st.dataframe(t.style.format({"% male": "{:.0f}", "Median age": "{:.0f}"}), hide_index=True, width="stretch")

st.subheader("Responsible use")
st.markdown("""
- **Privacy (RA 10173).** Never upload the raw line list to a public server. Rebuild the model offline with
  `python -m ews.build`, which writes only aggregated files.
- **Stigma.** Barangay rankings are for internal CESU planning. Do not publish them; frame action as resource allocation, not blame.
  Set `APP_PASSWORD` when deploying.
- **Reporting lag.** Median onset-to-encoding delay is 9–12 days; the latest weeks are always incomplete.
- **Automation bias.** Scores support epidemiological judgement and field validation; they do not replace them.

**Limits.** One season of data, one reanalysis grid cell, no population denominators, no flood-depth data, and 12 diseases too sparse to model.
""")
