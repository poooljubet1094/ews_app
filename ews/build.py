"""Rebuild every artifact the web app reads from the raw inputs.

    python -m ews.build --workbook ADS-DATASET.xlsx --weather data/weather.json --geo data/manila_barangays.geojson

Run this offline (never on the public server): the workbook contains personal data.
Only aggregated, de-identified files are written to ./artifacts.
"""
import argparse, os, runpy, tempfile
from .extract import extract

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--weather", default="data/weather.json")
    ap.add_argument("--geo", default="data/manila_barangays.geojson")
    ap.add_argument("--artifacts", default="artifacts")
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="ews_")
    pkl = os.path.join(work, "cases.pkl")
    extract(a.workbook).to_pickle(pkl)
    os.environ.update(EWS_CASES=pkl, EWS_WEATHER=a.weather, EWS_GEO=a.geo, EWS_WORK=work, EWS_ARTIFACTS=a.artifacts)
    runpy.run_path(os.path.join(os.path.dirname(__file__), "pipeline.py"), run_name="__main__")
    os.remove(pkl)                                   # do not leave case-level data behind
    print("done; temporary work folder:", work)

if __name__ == "__main__":
    main()
