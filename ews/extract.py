"""Read the CESU PIDSR workbook and keep only non-identifying fields (RA 10173)."""
import openpyxl, pandas as pd

KEEP = ['Sex', 'Age in Years', '(Current Address) City / Municipality', '(Current Address) Barangay', 'District',
        'Date on set of illness', 'Date of Consultation', 'Date admitted', 'case classification', 'outcome', 'Year',
        'Morbidity Week', 'timelapse dateonsettodateencode', 'pat address brgycode 10digit PSGC', 'Patient Admitted']

def extract(workbook_path: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)  # cached values; cells hold formulas
    frames = []
    for ws in wb.worksheets:
        rows = ws.iter_rows(values_only=True)
        hdr = [str(h).strip() if h is not None else '' for h in next(rows)]
        idx = {}
        for k in KEEP:
            for i, h in enumerate(hdr):
                if h.startswith(k) and k not in idx: idx[k] = i
        data = []
        for r in rows:
            if r is None or r[6] in (None, ''): continue        # column G (Sex) blank -> empty template row
            if str(r[0]).startswith('UNIQUE'): continue         # repeated header row
            data.append({k: (r[i] if i < len(r) else None) for k, i in idx.items()})
        df = pd.DataFrame(data); df['disease'] = ws.title.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
