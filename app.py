"""District VI Outbreak Early Warning — Streamlit web app.

Run locally:   streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="District VI Early Warning", page_icon="🩺", layout="wide")

from ui.common import password_gate  # noqa: E402

if not password_gate():
    st.stop()

with st.sidebar:
    st.markdown("### District VI Early Warning")
    st.caption("CESU Manila · PIDSR 2026, MW 1–40")
    st.radio("Disease", ["Dengue", "Leptospirosis"], key="disease", horizontal=True)

pages = [
    st.Page("ui/overview.py", title="Overview", icon=":material/monitoring:", default=True),
    st.Page("ui/riskmap.py", title="Barangay risk", icon=":material/map:"),
    st.Page("ui/whatif.py", title="Rainfall scenario", icon=":material/rainy:"),
    st.Page("ui/forecasts.py", title="Forecasts and models", icon=":material/insights:"),
    st.Page("ui/explain.py", title="Explainability", icon=":material/psychology:"),
    st.Page("ui/about.py", title="Method and ethics", icon=":material/info:"),
]
st.navigation(pages).run()

with st.sidebar:
    st.divider()
    st.caption("Paul Jovit Calibuso · Glenda Gonzales · Khimmy Joy Daquioag\n\nPSM in Data Science, Technological Institute of the Philippines")
