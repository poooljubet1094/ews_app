"""
XAI early-warning pipeline for notifiable diseases, District VI, City of Manila (PIDSR 2026 MW1-40).
Steps: clean -> weekly panels -> climate features -> rolling-origin forecasting (baselines vs ML)
       -> outbreak classification -> SHAP -> barangay risk scores for dashboard.
All outputs are aggregated counts; no patient-level identifiers are written.
"""
import json, re, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, roc_auc_score,
                             average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score)
import xgboost as xgb, shap
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")
np.random.seed(42)
import os
CASES_PKL = os.environ.get("EWS_CASES", "cases.pkl")
WEATHER_PATH = os.environ.get("EWS_WEATHER", "weather.json")
GEO_PATH = os.environ.get("EWS_GEO", "bgysubmuns-municity-1380600000.0.01.json")
OUT = os.environ.get("EWS_WORK", "work"); os.makedirs(OUT, exist_ok=True); os.makedirs(f"{OUT}/fig", exist_ok=True)
ART = os.environ.get("EWS_ARTIFACTS", "artifacts"); os.makedirs(f"{ART}/models", exist_ok=True)

DIS = {"DENGUE": "Dengue", "LEPTOSPIROSIS": "Leptospirosis", "INFLFUENZA-LIKE ILLNESS": "ILI"}
LAST_WEEK = {"Dengue": 40, "Leptospirosis": 40, "ILI": 37}   # right-censoring from reporting lag
W0 = pd.Timestamp("2026-01-04")                              # start (Sunday) of MW1

# ---------------------------------------------------------------- 1. clean
c = pd.read_pickle(CASES_PKL)
c = c[c.Sex.isin(["Male", "Female", "FEmale"])].copy()
c["disease_all"] = c.disease
c["District"] = c.District.astype(str).str.strip()
c["brgy"] = c["(Current Address) Barangay"].astype(str).str.strip()
c["mw"] = pd.to_numeric(c["Morbidity Week"], errors="coerce")
c["outcome"] = c.outcome.astype(str).str.upper()
c["cls"] = c["case classification"].astype(str).str.upper().str.extract(r"(SUSPECT|PROBABLE|CONFIRMED)")[0]
c["lag"] = pd.to_numeric(c["timelapse dateonsettodateencode"], errors="coerce")
c["age"] = pd.to_numeric(c["Age in Years"], errors="coerce")

# descriptive table (all 15 diseases)
desc = (c.groupby("disease_all").agg(cases=("Sex", "size"),
        d6=("District", lambda s: (s == "VI").sum()),
        deaths=("outcome", lambda s: s.str.contains("DIED").sum()),
        med_age=("age", "median"), male=("Sex", lambda s: (s == "Male").mean() * 100))
        .sort_values("cases", ascending=False))
desc.to_csv(f"{OUT}/table_disease_summary.csv")
lag_stats = c[c.disease.isin(DIS)].groupby("disease").lag.describe()[["count", "50%", "mean", "75%"]]
lag_stats.to_csv(f"{OUT}/table_reporting_lag.csv")

m = c[c.disease.isin(DIS) & c.mw.between(1, 41)].copy()
m["dis"] = m.disease.map(DIS)
d6_brgys = sorted(m.loc[m.District == "VI", "brgy"].unique().tolist() +
                  c.loc[c.District == "VI", "brgy"].unique().tolist())
d6_brgys = sorted(set(d6_brgys) - {"nan", "None", ""})
brgy_district = (c[c.brgy.notna()].groupby("brgy").District
                 .agg(lambda s: s.value_counts().index[0])).to_dict()
d6_area = (c[c.District == "VI"].groupby("brgy")["(Current Address) City / Municipality"]
           .agg(lambda s: s.value_counts().index[0])).to_dict()

# ---------------------------------------------------------------- 2. climate -> weekly
w = json.load(open(WEATHER_PATH))
wd = pd.DataFrame({k: v for k, v in w.items() if isinstance(v, list)})
wd["date"] = pd.date_range(w["time_start"], periods=len(wd))
wd["mw"] = ((wd.date - W0).dt.days // 7) + 1
wk = wd.groupby("mw").agg(rain=("precipitation_sum", "sum"), rain_max=("precipitation_sum", "max"),
                          tmean=("temperature_2m_mean", "mean"), tmax=("temperature_2m_max", "mean"),
                          rh=("relative_humidity_2m_mean", "mean"),
                          heavy_days=("precipitation_sum", lambda s: (s >= 20).sum()))
wk = wk.loc[-3:44]
CLIM = pd.DataFrame(index=wk.index)
for L in [1, 2, 3, 4]:
    CLIM[f"rain_l{L}"] = wk.rain.shift(L)
    CLIM[f"rh_l{L}"] = wk.rh.shift(L)
CLIM["rain_4wk_l1"] = wk.rain.shift(1).rolling(4).sum()
CLIM["rain_2wk_l1"] = wk.rain.shift(1).rolling(2).sum()
CLIM["heavy_days_l2"] = wk.heavy_days.shift(2)
CLIM["tmean_l2"] = wk.tmean.shift(2); CLIM["tmax_l2"] = wk.tmax.shift(2)
CLIM["tmean_l4"] = wk.tmean.shift(4)
wk.to_csv(f"{OUT}/weekly_climate.csv")

# ---------------------------------------------------------------- 3. panels
WEEKS = np.arange(1, 41)
DISTRICTS = ["I", "II", "III", "IV", "V", "VI"]

def district_panel(dis):
    s = m[m.dis == dis].groupby(["District", "mw"]).size()
    idx = pd.MultiIndex.from_product([DISTRICTS, np.arange(1, 42)], names=["District", "mw"])
    return s.reindex(idx, fill_value=0).rename("y").reset_index()

def add_lags(df, key, ycol="y", h=1):
    """features known at end of week t, target = y at week t+h"""
    df = df.sort_values([key, "mw"]).copy()
    g = df.groupby(key)[ycol]
    for L in [0, 1, 2, 3]:
        df[f"y_l{L}"] = g.shift(L)           # y_l0 = cases in week t (origin)
    df["y_ma4"] = g.transform(lambda s: s.rolling(4).mean())
    df["y_growth"] = np.log1p(df.y_l0) - np.log1p(df.y_l1)
    df["y_hist_mean"] = g.transform(lambda s: s.expanding().mean())
    df["target"] = g.shift(-h)
    df["target_mw"] = df.mw + h
    # climate aligned to target week, using only info available at origin:
    # CLIM row k holds weather lagged relative to week k; joining row t+1 to origin t means
    # rain_l1 = rainfall in the origin week itself (observed), rain_l2 = week t-1, ... -> strictly causal
    cl2 = CLIM.copy(); cl2.index = cl2.index - 1
    df = df.merge(cl2, left_on="mw", right_index=True, how="left")
    return df

# threshold: per-district mean + 2SD over baseline MW1-26 (pre-rainy-season), min 3 cases
def thresholds(panel):
    b = panel[panel.mw.between(1, 26)].groupby("District").y
    return (b.mean() + 2 * b.std()).clip(lower=3).to_dict()

# ---------------------------------------------------------------- 4. forecasting, rolling origin
def fit_predict(name, tr, te, feats):
    Xtr, ytr, Xte = tr[feats].values, tr.target.values, te[feats].values
    if name == "Naive (persistence)":
        return te.y_l0.values
    if name == "Moving average (4 wk)":
        return te.y_ma4.values
    if name == "Negative binomial GLM":
        Xa = sm.add_constant(np.column_stack([np.log1p(tr.y_l0), np.log1p(tr.y_l1), tr.rain_4wk_l1 / 100,
                                              tr.rh_l2 / 10, tr.tmean_l2, pd.get_dummies(tr.District).values[:, 1:]]),
                             has_constant="add")
        Xb = sm.add_constant(np.column_stack([np.log1p(te.y_l0), np.log1p(te.y_l1), te.rain_4wk_l1 / 100,
                                              te.rh_l2 / 10, te.tmean_l2,
                                              pd.get_dummies(te.District).reindex(columns=DISTRICTS, fill_value=0).values[:, 1:]]),
                             has_constant="add")
        try:
            r = sm.GLM(ytr, Xa.astype(float), family=sm.families.NegativeBinomial(alpha=0.3)).fit()
            return np.clip(r.predict(Xb.astype(float)), 0, 5 * max(ytr.max(), 1))
        except Exception:
            return te.y_l0.values
    if name == "Random forest":
        r = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
        r.fit(Xtr, ytr); return r.predict(Xte)
    if name == "XGBoost (Poisson)":
        r = xgb.XGBRegressor(objective="count:poisson", n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, max_delta_step=0.7)
        r.fit(Xtr, ytr); return r.predict(Xte)

MODELS = ["Naive (persistence)", "Moving average (4 wk)", "Negative binomial GLM", "Random forest", "XGBoost (Poisson)"]
FEATS = ["y_l0", "y_l1", "y_l2", "y_l3", "y_ma4", "y_growth", "y_hist_mean",
         "rain_l1", "rain_l2", "rain_l3", "rain_l4", "rain_2wk_l1", "rain_4wk_l1", "heavy_days_l2",
         "rh_l1", "rh_l2", "rh_l3", "tmean_l2", "tmax_l2", "tmean_l4", "dist_code"]
FEAT_LABEL = {"y_l0": "Cases this week", "y_l1": "Cases 1 wk ago", "y_l2": "Cases 2 wk ago", "y_l3": "Cases 3 wk ago",
              "y_ma4": "4-wk mean cases", "y_growth": "Case growth (log)", "y_hist_mean": "Historical mean cases",
              "rain_l1": "Rainfall lag 1 wk", "rain_l2": "Rainfall lag 2 wk", "rain_l3": "Rainfall lag 3 wk",
              "rain_l4": "Rainfall lag 4 wk", "rain_2wk_l1": "2-wk cumulative rain", "rain_4wk_l1": "4-wk cumulative rain",
              "heavy_days_l2": "Heavy-rain days (lag 2)", "rh_l1": "Humidity lag 1", "rh_l2": "Humidity lag 2",
              "rh_l3": "Humidity lag 3", "tmean_l2": "Mean temp lag 2", "tmax_l2": "Max temp lag 2",
              "tmean_l4": "Mean temp lag 4", "dist_code": "District"}

results, preds_all, shap_store = [], [], {}
ORIGINS = range(20, 40)
for dis in ["Dengue", "Leptospirosis", "ILI"]:
    pnl = district_panel(dis)
    thr = thresholds(pnl)
    for h in [1, 2]:
        df = add_lags(pnl, "District", h=h)
        df["dist_code"] = df.District.map({k: i for i, k in enumerate(DISTRICTS)})
        df = df[df.target_mw <= LAST_WEEK[dis]].dropna(subset=["target", "y_l3", "rain_4wk_l1"])
        rows = []
        for t in ORIGINS:
            if t + h > LAST_WEEK[dis]: break
            tr = df[df.target_mw <= t]; te = df[df.mw == t]
            if len(te) == 0: continue
            for mn in MODELS:
                p = fit_predict(mn, tr, te, FEATS)
                rows.append(pd.DataFrame({"dis": dis, "h": h, "model": mn, "origin": t, "District": te.District.values,
                                          "target_mw": te.target_mw.values, "y": te.target.values, "pred": p}))
        P = pd.concat(rows)
        P["thr"] = P.District.map(thr)
        P["out_true"] = (P.y > P.thr).astype(int); P["out_pred"] = (P.pred > P.thr).astype(int)
        preds_all.append(P)
        for mn, g in P.groupby("model"):
            tp = ((g.out_true == 1) & (g.out_pred == 1)).sum()
            results.append({"Disease": dis, "Horizon": h, "Model": mn,
                            "MAE": mean_absolute_error(g.y, g.pred),
                            "RMSE": mean_squared_error(g.y, g.pred) ** 0.5,
                            "Sensitivity": recall_score(g.out_true, g.out_pred, zero_division=0),
                            "Precision": precision_score(g.out_true, g.out_pred, zero_division=0),
                            "F1": f1_score(g.out_true, g.out_pred, zero_division=0),
                            "Outbreak weeks": int(g.out_true.sum()), "n": len(g)})
        # explanatory fit on all data for SHAP (h=1 only)
        if h == 1:
            r = xgb.XGBRegressor(objective="count:poisson", n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, max_delta_step=0.7)
            r.fit(df[FEATS].values, df.target.values)
            ex = shap.TreeExplainer(r); sv = ex.shap_values(df[FEATS].values)
            shap_store[dis] = (sv, df[FEATS].copy(), df.copy())
    json.dump({k: float(v) for k, v in thr.items()}, open(f"{OUT}/thr_{dis}.json", "w"))

R = pd.DataFrame(results); R.to_csv(f"{OUT}/table_forecast_metrics.csv", index=False)
PA = pd.concat(preds_all); PA.to_csv(f"{OUT}/rolling_predictions.csv", index=False)

# ---------------------------------------------------------------- 5. SARIMAX on District VI totals
sar_rows = []
for dis in ["Dengue", "Leptospirosis", "ILI"]:
    y = district_panel(dis).query("District=='VI'").set_index("mw").y.loc[1:LAST_WEEK[dis]].astype(float)
    yl = np.log1p(y)
    for h in [1, 2]:
        # exogenous weather lagged by h weeks so forecasts use only observed weather
        ex = pd.DataFrame({"rain4": wk.rain.shift(h).rolling(4).sum() / 100, "rh": wk.rh.shift(h) / 10}).loc[y.index].bfill()
        fc = []
        for t in ORIGINS:
            if t + h > LAST_WEEK[dis]: break
            try:
                mod = SARIMAX(yl.loc[:t].values, exog=ex.loc[:t].values, order=(1, 1, 0), trend="n",
                              enforce_stationarity=False).fit(disp=False)
                f = mod.get_forecast(h, exog=ex.loc[t + 1:t + h].values).predicted_mean[-1]
                fc.append((t + h, float(np.expm1(f)), float(y.loc[t + h])))
            except Exception as e:
                print("SARIMAX fallback", dis, h, t, e)
                fc.append((t + h, float(y.loc[t]), float(y.loc[t + h])))
        F = pd.DataFrame(fc, columns=["mw", "pred", "y"])
        # compare with other models restricted to District VI
        sar_rows.append({"Disease": dis, "Horizon": h, "Model": "SARIMAX(1,1,0)+climate",
                         "MAE": mean_absolute_error(F.y, F.pred.clip(0)),
                         "RMSE": mean_squared_error(F.y, F.pred.clip(0)) ** 0.5})
        for mn in MODELS:
            g = PA[(PA.dis == dis) & (PA.h == h) & (PA.model == mn) & (PA.District == "VI")]
            sar_rows.append({"Disease": dis, "Horizon": h, "Model": mn,
                             "MAE": mean_absolute_error(g.y, g.pred), "RMSE": mean_squared_error(g.y, g.pred) ** 0.5})
S = pd.DataFrame(sar_rows); S.to_csv(f"{OUT}/table_d6_metrics.csv", index=False)

# ---------------------------------------------------------------- 6. barangay-level next-week risk (all Manila train, D-VI eval)
def brgy_panel(dis):
    sub = m[(m.dis == dis) & m.brgy.str.match(r"^Barangay \d+$", na=False)]
    allb = sorted(set(sub.brgy) | set(b for b in d6_brgys if re.match(r"^Barangay \d+$", b)))
    s = sub.groupby(["brgy", "mw"]).size()
    idx = pd.MultiIndex.from_product([allb, np.arange(1, 42)], names=["brgy", "mw"])
    p = s.reindex(idx, fill_value=0).rename("y").reset_index()
    p["District"] = p.brgy.map(brgy_district)
    dist = district_panel(dis).rename(columns={"y": "dist_y"})
    p = p.merge(dist, on=["District", "mw"], how="left")
    city = m[m.dis == dis].groupby("mw").size().rename("city_y")
    p = p.merge(city, left_on="mw", right_index=True, how="left").fillna({"city_y": 0, "dist_y": 0})
    return p

BFEATS = ["y_l0", "y_l1", "y_l2", "y_l3", "y_ma4", "y_hist_mean", "dist_l0", "dist_l1", "city_l0", "city_growth",
          "rain_l1", "rain_l2", "rain_l3", "rain_4wk_l1", "rh_l2", "tmean_l2"]
FEAT_LABEL.update({"dist_l0": "District cases this week", "dist_l1": "District cases 1 wk ago",
                   "city_l0": "City cases this week", "city_growth": "City case growth (log)"})
brgy_rows, brgy_scores, bshap = [], {}, {}
for dis in ["Dengue", "Leptospirosis"]:
    p = brgy_panel(dis).sort_values(["brgy", "mw"])
    g = p.groupby("brgy")
    for L in [0, 1, 2, 3]: p[f"y_l{L}"] = g.y.shift(L)
    p["y_ma4"] = g.y.transform(lambda s: s.rolling(4).mean())
    p["y_hist_mean"] = g.y.transform(lambda s: s.expanding().mean())
    p["dist_l0"] = p.dist_y; p["dist_l1"] = g.dist_y.shift(1)
    p["city_l0"] = p.city_y; p["city_growth"] = np.log1p(p.city_y) - np.log1p(g.city_y.shift(1))
    cl2 = CLIM.copy(); cl2.index = cl2.index - 1
    p = p.merge(cl2, left_on="mw", right_index=True, how="left")
    p["target"] = (g.y.shift(-1) > 0).astype(float); p.loc[g.y.shift(-1).isna(), "target"] = np.nan
    p["target_mw"] = p.mw + 1
    full = p.copy()
    p = p[p.target_mw <= LAST_WEEK[dis]].dropna(subset=["target", "y_l3", "rain_4wk_l1"])
    rows = []
    for t in range(20, LAST_WEEK[dis]):
        tr = p[p.target_mw <= t]; te = p[(p.mw == t) & (p.District == "VI")]
        if tr.target.nunique() < 2 or len(te) == 0: continue
        out = {"Naive (had case this week)": (te.y_l0 > 0).astype(float).values * 0.5 + te.y_hist_mean.values * 0.5}
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(np.log1p(tr[BFEATS].clip(lower=0).values), tr.target)
        out["Logistic regression"] = lr.predict_proba(np.log1p(te[BFEATS].clip(lower=0).values))[:, 1]
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1,
                                    class_weight="balanced_subsample").fit(tr[BFEATS].values, tr.target)
        out["Random forest"] = rf.predict_proba(te[BFEATS].values)[:, 1]
        xg = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, eval_metric="logloss").fit(tr[BFEATS].values, tr.target)
        out["XGBoost"] = xg.predict_proba(te[BFEATS].values)[:, 1]
        for mn, pr in out.items():
            rows.append(pd.DataFrame({"model": mn, "origin": t, "brgy": te.brgy.values, "y": te.target.values, "p": pr}))
    B = pd.concat(rows)
    for mn, gg in B.groupby("model"):
        # top-10 hit rate: share of next-week case barangays captured in the 10 highest-risk barangays each week
        hits = []
        for t, gt in gg.groupby("origin"):
            top = gt.nlargest(10, "p"); hits.append(top.y.sum() / max(gt.y.sum(), 1))
        brgy_rows.append({"Disease": dis, "Model": mn, "ROC-AUC": roc_auc_score(gg.y, gg.p),
                          "PR-AUC": average_precision_score(gg.y, gg.p),
                          "Brier": brier_score_loss(gg.y, gg.p.clip(0, 1)), "Top-10 capture": np.mean(hits),
                          "Prevalence": gg.y.mean(), "n": len(gg)})
    B.to_csv(f"{OUT}/brgy_predictions_{dis}.csv", index=False)
    # final model on all labelled data -> score latest origin (MW40) for dashboard
    xg = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, random_state=42, eval_metric="logloss").fit(p[BFEATS].values, p.target)
    last = full[(full.mw == LAST_WEEK[dis]) & (full.District == "VI")].copy()
    last["risk"] = xg.predict_proba(last[BFEATS].fillna(0).values)[:, 1]
    xg.save_model(f"{ART}/models/brgy_{dis}.json")                       # used by the web app's what-if simulator
    last[["brgy", "District", "mw"] + BFEATS + ["risk"]].to_csv(f"{ART}/latest_features_{dis}.csv", index=False)
    ex = shap.TreeExplainer(xg); sv = ex.shap_values(last[BFEATS].fillna(0).values)
    last["top_driver"] = [FEAT_LABEL[BFEATS[i]] for i in np.argmax(np.abs(sv), axis=1)]
    bshap[dis] = (ex.shap_values(p[BFEATS].values), p[BFEATS].copy())
    hist = full[full.District == "VI"].pivot_table(index="brgy", columns="mw", values="y", aggfunc="sum").fillna(0)
    brgy_scores[dis] = (last, hist)
BR = pd.DataFrame(brgy_rows); BR.to_csv(f"{OUT}/table_brgy_metrics.csv", index=False)

# ---------------------------------------------------------------- 7. figures
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C = {"Dengue": "#2F6DB5", "Leptospirosis": "#C2492E", "ILI": "#5F8F3E", "rain": "#7A8899"}

# Fig 1: epi curves District VI + rainfall
fig, ax = plt.subplots(figsize=(3.5, 2.3)); ax2 = ax.twinx()
ax2.bar(wk.loc[1:41].index, wk.loc[1:41].rain, color=C["rain"], alpha=0.35, width=0.8, label="Rainfall (mm/wk)")
for dis in ["Dengue", "Leptospirosis", "ILI"]:
    y = district_panel(dis).query("District=='VI'").set_index("mw").y.loc[1:LAST_WEEK[dis]]
    ax.plot(y.index, y.values, color=C[dis], lw=1.4, label=dis)
ax.set_xlabel("Morbidity week (2026)"); ax.set_ylabel("Cases per week, District VI"); ax2.set_ylabel("Rainfall (mm/week)")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig/fig1_epicurve.png", dpi=300); plt.close()

# Fig 2: rain-lag cross-correlation (city level)
fig, ax = plt.subplots(figsize=(3.5, 2.0))
for dis in ["Dengue", "Leptospirosis", "ILI"]:
    y = m[m.dis == dis].groupby("mw").size().reindex(range(1, LAST_WEEK[dis] + 1), fill_value=0)
    cc = [np.corrcoef(np.log1p(y.values), wk.rain.shift(L).loc[y.index].values)[0, 1] for L in range(0, 9)]
    ax.plot(range(0, 9), cc, marker="o", ms=3, color=C[dis], label=dis)
ax.axhline(0, color="#999", lw=0.6); ax.set_xlabel("Rainfall lag (weeks)"); ax.set_ylabel("Pearson r with log cases")
ax.legend(frameon=False, fontsize=6.5); fig.tight_layout(); fig.savefig(f"{OUT}/fig/fig2_lagcorr.png", dpi=300); plt.close()

# Fig 3: rolling forecasts District VI (h=1) best ML vs baselines
fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
for ax, dis in zip(axes, ["Dengue", "Leptospirosis"]):
    g = PA[(PA.dis == dis) & (PA.h == 1) & (PA.District == "VI")]
    obs = g[g.model == MODELS[0]].set_index("target_mw").y
    ax.plot(obs.index, obs.values, color="black", lw=1.4, label="Observed")
    for mn, st in [("Naive (persistence)", ":"), ("Negative binomial GLM", "--"), ("XGBoost (Poisson)", "-")]:
        s = g[g.model == mn].set_index("target_mw").pred
        ax.plot(s.index, s.values, ls=st, lw=1.1, color=C[dis] if mn.startswith("XG") else "#777", label=mn)
    thr = json.load(open(f"{OUT}/thr_{dis}.json"))["VI"]
    ax.axhline(thr, color="#C9A227", lw=0.9, ls="-.", label="Epidemic threshold")
    ax.set_title(f"{dis}, District VI, one-week-ahead", fontsize=8); ax.set_ylabel("Cases")
axes[0].legend(frameon=False, fontsize=6); axes[1].set_xlabel("Target morbidity week")
fig.tight_layout(); fig.savefig(f"{OUT}/fig/fig3_forecasts.png", dpi=300); plt.close()

# Fig 4: SHAP bar (district models)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, dis in zip(axes, ["Dengue", "Leptospirosis"]):
    sv, X, _ = shap_store[dis]
    imp = pd.Series(np.abs(sv).mean(0), index=[FEAT_LABEL[f] for f in FEATS]).sort_values().tail(10)
    ax.barh(imp.index, imp.values, color=C[dis]); ax.set_title(f"{dis} (district model)", fontsize=8)
    ax.set_xlabel("Mean absolute SHAP value (log-count scale)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig/fig4_shap_bar.png", dpi=300); plt.close()

# Fig 5: SHAP dependence for lepto rain_4wk & dengue rain lag
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
for ax, (dis, f) in zip(axes, [("Leptospirosis", "rain_4wk_l1"), ("Dengue", "rh_l2")]):
    sv, X, _ = shap_store[dis]; j = FEATS.index(f)
    ax.scatter(X[f], sv[:, j], s=5, color=C[dis], alpha=0.6); ax.axhline(0, color="#999", lw=0.5)
    ax.set_xlabel(FEAT_LABEL[f] + (" (mm)" if "rain" in f else " (%)"), fontsize=7); ax.set_ylabel("SHAP value", fontsize=7)
    ax.set_title(dis, fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig/fig5_shap_dep.png", dpi=300); plt.close()

# ---------------------------------------------------------------- 8. dashboard data
geo = json.load(open(GEO_PATH))
feats = []
for f in geo["features"]:
    nm = f["properties"]["adm4_en"]
    if nm in d6_brgys:
        feats.append({"type": "Feature", "properties": {"name": nm, "area": d6_area.get(nm, "")}, "geometry": f["geometry"]})
# round coordinates to shrink
def rnd(o):
    if isinstance(o, list): return [rnd(x) for x in o]
    return round(o, 5) if isinstance(o, float) else o
for f in feats: f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
dash = {"geo": {"type": "FeatureCollection", "features": feats}, "diseases": {}, "climate": wk.loc[1:41].round(1).reset_index().to_dict("list")}
for dis in ["Dengue", "Leptospirosis"]:
    last, hist = brgy_scores[dis]
    dash["diseases"][dis] = {
        "forecast_week": LAST_WEEK[dis] + 1,
        "brgy": {r.brgy: {"risk": round(float(r.risk), 3), "driver": r.top_driver,
                          "recent4": int(hist.loc[r.brgy, LAST_WEEK[dis] - 3:LAST_WEEK[dis]].sum()) if r.brgy in hist.index else 0,
                          "total": int(hist.loc[r.brgy, 1:LAST_WEEK[dis]].sum()) if r.brgy in hist.index else 0}
                 for r in last.itertuples()},
        "d6_series": district_panel(dis).query("District=='VI'").set_index("mw").y.loc[1:LAST_WEEK[dis]].tolist(),
        "threshold": json.load(open(f"{OUT}/thr_{dis}.json"))["VI"],
        "metrics": BR[BR.Disease == dis].round(3).to_dict("records")}
json.dump(dash, open(f"{OUT}/dashboard_data.json", "w"))
print("matched geometries:", len(feats), "of", len(d6_brgys))
pd.set_option("display.width", 200)
print(desc); print(lag_stats); print(R.round(2).to_string()); print(S.round(2).to_string()); print(BR.round(3).to_string())

# ---------------------------------------------------------------- 9. SHAP importance tables
rows = []
for dis, (sv, X, _) in shap_store.items():
    for f, v in zip(FEATS, np.abs(sv).mean(0)): rows.append({"Disease": dis, "level": "district", "Feature": FEAT_LABEL[f], "mean_abs_shap": v})
for dis, (sv, X) in bshap.items():
    for f, v in zip(BFEATS, np.abs(sv).mean(0)): rows.append({"Disease": dis, "level": "barangay", "Feature": FEAT_LABEL[f], "mean_abs_shap": v})
SH = pd.DataFrame(rows); SH.to_csv(f"{OUT}/table_shap.csv", index=False)
for (d, l), g in SH.groupby(["Disease", "level"]):
    print(d, l, g.nlargest(6, "mean_abs_shap")[["Feature", "mean_abs_shap"]].round(3).values.tolist())
# Fig 6: barangay-level SHAP bar for leptospirosis & dengue
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
for ax, dis in zip(axes, ["Dengue", "Leptospirosis"]):
    g = SH[(SH.Disease == dis) & (SH.level == "barangay")].nlargest(10, "mean_abs_shap").iloc[::-1]
    ax.barh(g.Feature, g.mean_abs_shap, color=C[dis]); ax.set_title(f"{dis} (barangay risk model)", fontsize=8)
    ax.set_xlabel("Mean absolute SHAP value (log-odds)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig/fig6_shap_brgy.png", dpi=300); plt.close()

# ---------------------------------------------------------------- 10. early-warning lead time
def sustained_onset(weeks_flag):
    """first week w with flag at w and w+1 (sustained exceedance)"""
    f = weeks_flag.sort_index()
    for w_ in f.index:
        if f.get(w_, 0) == 1 and f.get(w_ + 1, 0) == 1: return w_
    return np.nan
trig = (wk.heavy_days >= 3)            # a-priori rule: >=3 days with >=20 mm rain in one week
lt = []
for (dis, h, mn), g in PA[PA.target_mw >= 27].groupby(["dis", "h", "model"]):
    for dist, gd in g.groupby("District"):
        onset = sustained_onset(gd.set_index("target_mw").out_true)
        if onset != onset: continue
        prd = gd[gd.out_pred == 1].target_mw
        first_alert = prd[prd >= onset - 3].min() if len(prd) else np.nan
        lt.append({"Disease": dis, "h": h, "Model": mn, "District": dist, "onset_mw": onset,
                   "first_alert_target_mw": first_alert,
                   "alert_delay_wk": first_alert - onset if first_alert == first_alert else np.nan})
LT = pd.DataFrame(lt)
# rainfall trigger: alert issued at end of week t applies to weeks t+1..t+2
trig_weeks = [int(x) for x in wk.index[trig]]
for dis in ["Dengue", "Leptospirosis", "ILI"]:
    for dist in DISTRICTS:
        sub = LT[(LT.Disease == dis) & (LT.District == dist)]
        if len(sub) == 0: continue
        onset = sub.onset_mw.iloc[0]
        cover = [t + 1 for t in trig_weeks if t >= 20]
        fa = min([c_ for c_ in cover if c_ >= onset - 3], default=np.nan)
        LT = pd.concat([LT, pd.DataFrame([{"Disease": dis, "h": 1, "Model": "Rainfall trigger (>=3 heavy-rain days)",
                        "District": dist, "onset_mw": onset, "first_alert_target_mw": fa, "alert_delay_wk": fa - onset}])])
LT.to_csv(f"{OUT}/table_leadtime.csv", index=False)
LS = LT[LT.h == 1].groupby(["Disease", "Model"]).agg(onset=("onset_mw", "median"), median_delay=("alert_delay_wk", "median"),
        early_or_on_time=("alert_delay_wk", lambda s: int((s <= 0).sum())), districts=("District", "count")).reset_index()
LS.to_csv(f"{OUT}/table_leadtime_summary.csv", index=False)
print("trigger weeks", trig_weeks); print(LS.to_string())


# ---------------------------------------------------------------- 11. export de-identified artifacts for the web app
import shutil
for f in ["table_forecast_metrics.csv", "table_d6_metrics.csv", "table_brgy_metrics.csv", "table_shap.csv",
          "table_leadtime_summary.csv", "table_disease_summary.csv", "table_reporting_lag.csv", "weekly_climate.csv"]:
    shutil.copy(f"{OUT}/{f}", f"{ART}/{f}")
PA[PA.District == "VI"].to_csv(f"{ART}/d6_rolling_predictions.csv", index=False)
dw = pd.concat([district_panel(dd).assign(disease=dd) for dd in ["Dengue", "Leptospirosis", "ILI"]])
dw = dw[dw.mw <= dw.disease.map(LAST_WEEK)]
dw["threshold"] = [json.load(open(f"{OUT}/thr_{r.disease}.json"))[r.District] for r in dw.itertuples()]
dw.to_csv(f"{ART}/district_weekly.csv", index=False)
rows, hist_rows = [], []
area = {f["properties"]["name"]: f["properties"]["area"] for f in feats}
for dd, (last, hist) in brgy_scores.items():
    last = last.sort_values("risk", ascending=False).reset_index(drop=True)
    for k, r in last.iterrows():
        rows.append({"disease": dd, "brgy": r.brgy, "area": area.get(r.brgy, d6_area.get(r.brgy, "")), "rank": k + 1,
                     "risk": round(float(r.risk), 4), "driver": r.top_driver, "forecast_mw": LAST_WEEK[dd] + 1,
                     "cases_last4": int(hist.loc[r.brgy, LAST_WEEK[dd] - 3:LAST_WEEK[dd]].sum()) if r.brgy in hist.index else 0,
                     "cases_ytd": int(hist.loc[r.brgy, 1:LAST_WEEK[dd]].sum()) if r.brgy in hist.index else 0})
    hh = hist.loc[:, 1:LAST_WEEK[dd]].stack().rename("cases").reset_index(); hh["disease"] = dd; hist_rows.append(hh)
pd.DataFrame(rows).to_csv(f"{ART}/barangay_risk.csv", index=False)
pd.concat(hist_rows).astype({"cases": int}).to_csv(f"{ART}/barangay_weekly_d6.csv", index=False)
json.dump({"type": "FeatureCollection", "features": feats}, open(f"{ART}/d6_barangays.geojson", "w"), separators=(",", ":"))
json.dump({"BFEATS": BFEATS, "FEAT_LABEL": FEAT_LABEL, "LAST_WEEK": LAST_WEEK, "W0": str(W0.date()),
           "thresholds_VI": {dd: json.load(open(f"{OUT}/thr_{dd}.json"))["VI"] for dd in ["Dengue", "Leptospirosis", "ILI"]}},
          open(f"{ART}/meta.json", "w"), indent=1)
print("artifacts written to", ART)
