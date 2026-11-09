"""Smoke test: every page renders without an exception.   pytest -q"""
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGES = ["ui/overview.py", "ui/riskmap.py", "ui/whatif.py", "ui/forecasts.py", "ui/explain.py", "ui/about.py"]

@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("disease", ["Dengue", "Leptospirosis"])
def test_page(page, disease):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.session_state["disease"] = disease
    at.run()
    assert not at.exception, at.exception
