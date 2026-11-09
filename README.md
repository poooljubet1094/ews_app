# District VI Outbreak Early Warning (web app)

Explainable AI early warning system for notifiable diseases in District VI, City of Manila.
PSM in Data Science final project, Technological Institute of the Philippines.
Paul Jovit Calibuso · Glenda Gonzales · Khimmy Joy Daquioag

## What the app does

| Page | Purpose |
|---|---|
| Overview | District VI epidemic curve vs threshold and rainfall, alert status, rainfall trigger, weekly action notes |
| Barangay risk | Map and ranked watch list of next-week risk; per-barangay case history and SHAP explanation |
| Rainfall scenario | What-if simulator: change 4 weeks of rainfall, humidity and temperature and re-score every barangay live |
| Forecasts and models | Rolling-origin forecasts vs observed; model comparison tables; alert timing; barangay model metrics |
| Explainability | Global SHAP importance (district and barangay models); rainfall lag correlations |
| Method and ethics | Data, models, privacy (RA 10173), stigma, limits |

Diseases: dengue and leptospirosis (ILI on the forecasts page).

## Project layout

```
app.py                  Streamlit entry point (navigation, sidebar, optional password)
ui/                     one file per page + common.py (loaders, scoring, charts)
artifacts/              de-identified outputs the app reads (aggregated counts, metrics, SHAP, XGBoost models)
ews/                    offline pipeline: extract.py, pipeline.py, build.py
data/                   weather.json (Open-Meteo) and Manila barangay GeoJSON used by the pipeline
tests/test_app.py       smoke test: every page renders for both diseases
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                   # opens http://localhost:8501
pytest -q                                              # optional smoke test (pip install pytest)
```

## Deploy to the web

The app needs only `requirements.txt` and the `artifacts/` folder. No database, no GPU.

### Option A: Streamlit Community Cloud (free, easiest)
1. Push this folder to a GitHub repository (private is fine).
2. Go to share.streamlit.io, choose **Create app**, pick the repo, branch and `app.py`.
3. Under **Advanced settings → Secrets**, add `APP_PASSWORD = "your-code"` to restrict access (recommended).
4. Deploy. You get a URL like `https://<name>.streamlit.app`.

### Option B: Render (Docker)
1. Push to GitHub, then in Render choose **New → Blueprint** and select the repo (uses `render.yaml`),
   or **New → Web Service → Docker**.
2. Set the `APP_PASSWORD` environment variable.

### Option C: any Docker host (Azure App Service, Google Cloud Run, Railway, a VPS)
```bash
docker build -t d6-ews .
docker run -p 8501:8501 -e APP_PASSWORD=your-code d6-ews
```
Cloud Run and App Service set `$PORT` automatically; the image honours it.

Heroku-style hosts can use the included `Procfile`.

## Rebuild the models with new surveillance data (offline only)

The raw PIDSR workbook contains names and birthdates. Never upload it to the server.
On a CESU or team computer:

```bash
pip install -r requirements-build.txt
# refresh data/weather.json first if new weeks were added (Open-Meteo archive, 14.59N 121.00E)
python -m ews.build --workbook path/to/ADS-DATASET.xlsx
```

This reads only non-identifying fields, retrains every model, and rewrites `artifacts/`.
Commit the new `artifacts/` folder and redeploy. Takes about 3 minutes.
Note: `ews/pipeline.py` currently fixes the analysis window (MW 1–40, ILI to MW 37) in `LAST_WEEK`; update it for a new season.

## Responsible use
- Barangay rankings are for internal CESU planning. Keep the app behind `APP_PASSWORD`; do not publish rankings.
- Scores support, not replace, epidemiological judgement and field validation.
- One season of data: treat scenario outputs as direction, not exact probabilities.

## Data sources
- CESU Manila PIDSR line list, 2026 (de-identified, aggregated).
- Open-Meteo historical weather API (ERA5 reanalysis).
- Barangay boundaries: github.com/faeldon/philippines-json-maps.
