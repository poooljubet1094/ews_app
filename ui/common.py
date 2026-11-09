"""Shared loaders, scoring and chart helpers for the web app."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st, xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ART = Path(__file__).resolve().parent.parent / "artifacts"
COLORS = {"Dengue": "#2F6DB5", "Leptospirosis": "#C2492E", "ILI": "#5F8F3E"}
ALERT = "#B3261E"
MUTED = "#8A94A3"
W0 = pd.Timestamp("2026-01-04")          # Sunday that starts Morbidity Week 1, 2026


@st.cache_data
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ART / name)


@st.cache_data
def js(name: str):
    with open(ART / name) as f:
        return json.load(f)


@st.cache_resource
def model(disease: str) -> xgb.Booster:
    b = xgb.Booster(); b.load_model(str(ART / "models" / f"brgy_{disease}.json")); return b


def meta():
    return js("meta.json")


def feats():
    return meta()["BFEATS"]


def label(f):
    return meta()["FEAT_LABEL"].get(f, f)


def score(disease: str, X: pd.DataFrame):
    """Return (probability, SHAP contributions incl. bias column) from the saved XGBoost model."""
    dm = xgb.DMatrix(X[feats()].fillna(0).to_numpy(dtype=float))
    b = model(disease)
    return b.predict(dm), b.predict(dm, pred_contribs=True)


def week_label(mw: int) -> str:
    start = W0 + pd.Timedelta(days=7 * (int(mw) - 1))
    return f"MW {int(mw)} ({start:%b %d})"


def pct(x):
    return f"{x * 100:.1f}%"


def base_layout(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.08, x=0),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=12), hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    return fig


def epi_chart(disease: str, district: str = "VI", height=360):
    dw = csv("district_weekly.csv")
    d = dw[(dw.disease == disease) & (dw.District == district)].sort_values("mw")
    clim = csv("weekly_climate.csv").set_index("mw")
    thr = float(d.threshold.iloc[0])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=d.mw, y=d.y, name="Cases", marker_color=[ALERT if v > thr else COLORS[disease] for v in d.y],
                hovertemplate="%{y} cases<extra></extra>")
    fig.add_scatter(x=d.mw, y=clim.loc[d.mw, "rain"].values, name="Rainfall (mm/week)", mode="lines",
                    line=dict(color=MUTED, width=2), secondary_y=True, hovertemplate="%{y:.0f} mm<extra></extra>")
    fig.add_hline(y=thr, line=dict(color="#C9A227", dash="dash", width=1.5),
                  annotation_text=f"Epidemic threshold {thr:.1f}", annotation_position="top left")
    fig.update_yaxes(title_text="Cases per week", secondary_y=False)
    fig.update_yaxes(title_text="Rainfall (mm/week)", secondary_y=True, showgrid=False, rangemode="tozero")
    fig.update_yaxes(rangemode="tozero", secondary_y=False)
    fig.update_xaxes(title_text="Morbidity week, 2026")
    return base_layout(fig, height)


def _rings(geom):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [ring for poly in polys for ring in poly[:1]]          # outer rings only


def risk_map(df: pd.DataFrame, value: str, title: str, scale="Reds", height=520, rng=None, outline_top=10):
    """Choropleth of District VI barangays drawn as filled polygons.

    Self-contained: needs no map tiles or CDN downloads, so it renders on any network."""
    from plotly.colors import sample_colorscale
    geo = js("d6_barangays.geojson")
    lo, hi = rng or (0, float(df[value].max()) or 1.0)
    is_risk = value.startswith("risk")
    d = df.set_index("brgy")
    top = set(df.nlargest(outline_top, value).brgy) if outline_top else set()
    fig = go.Figure()
    for f in sorted(geo["features"], key=lambda f: f["properties"]["name"] in top):   # watch list drawn last (on top)
        nm = f["properties"]["name"]
        if nm in d.index:
            v = float(d.at[nm, value]); t = (v - lo) / (hi - lo) if hi > lo else 0
            fill = sample_colorscale(scale, [min(max(t, 0), 1)])[0]
            area = d.at[nm, "area"] if "area" in d.columns and pd.notna(d.at[nm, "area"]) else ""
            drv = d.at[nm, "driver"] if "driver" in d.columns and pd.notna(d.at[nm, "driver"]) and d.at[nm, "driver"] else ""
            tip = f"<b>{nm}</b>  {area}<br>{title}: " + (f"{v:.1%}" if is_risk else f"{v:g}") + (f"<br>Main driver: {drv}" if drv else "")
        else:
            fill, tip = "rgba(200,200,200,0.4)", f"<b>{nm}</b><br>not scored"
        for ring in _rings(f["geometry"]):
            xs, ys = zip(*ring)
            fig.add_scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=fill, hoveron="fills", name=tip, hoverinfo="name",
                            line=dict(color="#111" if nm in top else "rgba(70,70,70,0.65)", width=2.4 if nm in top else 0.6),
                            showlegend=False)
    fig.add_scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                    marker=dict(colorscale=scale, cmin=lo, cmax=hi, color=[lo], showscale=True,
                                colorbar=dict(title=title, tickformat=".0%" if is_risk else None, thickness=12, len=0.8)))
    lat0 = np.mean([c[1] for f in geo["features"] for r in _rings(f["geometry"]) for c in r])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1 / np.cos(np.radians(lat0)))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", dragmode="pan", hoverlabel=dict(align="left"))
    return fig


def shap_bar(contrib_row: np.ndarray, top=8, height=320):
    F = feats()
    s = pd.Series(contrib_row[:-1], index=[label(f) for f in F])
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:top][::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[ALERT if v > 0 else "#4E7FB8" for v in s.values],
                           hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#888", width=1))
    fig.update_layout(xaxis_title="Contribution to risk (log-odds; red raises, blue lowers)")
    return base_layout(fig, height).update_layout(hovermode="closest")


def password_gate():
    """Optional: set APP_PASSWORD as an environment variable or Streamlit secret to restrict access."""
    pw = os.environ.get("APP_PASSWORD")
    try:
        pw = pw or st.secrets.get("APP_PASSWORD")
    except Exception:
        pass
    if not pw or st.session_state.get("auth_ok"):
        return True
    st.title("District VI Early Warning")
    st.caption("Restricted to CESU Manila staff and the project team.")
    entry = st.text_input("Access code", type="password", key="pw_entry")
    if st.button("Enter", type="primary"):
        if entry == pw:
            st.session_state.auth_ok = True; st.rerun()
        else:
            st.error("That access code is not correct.")
    return False
