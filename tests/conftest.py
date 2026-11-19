import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from biziq import data_layer  # noqa: E402


@pytest.fixture(scope="session")
def sample():
    """Cleaned sample datasets straight from the CSVs (no database involved)."""
    out = {}
    for name in data_layer.SCHEMAS:
        raw = data_layer.read_upload(data_layer.SAMPLE_DIR / f"{name}.csv")
        df, rep = data_layer.clean_and_validate(raw, name)
        assert rep.ok, (name, rep.errors)
        out[name] = df
    return out


@pytest.fixture(scope="session")
def pr(sample):
    from biziq import pipeline
    return pipeline.run_all(sample, as_of=pd.Timestamp("2026-10-31"), today=pd.Timestamp("2026-11-01"))


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Point the database and output folders at a temp dir so tests never touch real state."""
    from biziq import automation
    monkeypatch.setattr(data_layer, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(automation, "OUTBOX", tmp_path / "outbox")
    monkeypatch.setattr(automation, "DRAFTS", tmp_path / "drafts")
    monkeypatch.setattr(automation, "SUMMARIES", tmp_path / "summaries")
    return tmp_path
