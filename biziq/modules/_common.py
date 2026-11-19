"""Small helpers shared by the platform modules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import inr, safe_div  # noqa: F401  (re-exported for the modules)


def pct1(x: float) -> float | None:
    """Fraction -> percentage rounded to 1 decimal; None when not a number."""
    return None if x is None or pd.isna(x) else round(float(x) * 100, 1)


def r1(x, d: int = 1):
    return None if x is None or pd.isna(x) else round(float(x), d)


def open_mask(paid: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """True where an invoice is still unpaid on `as_of`."""
    return paid.isna() | (paid > as_of)


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return (b - a).days / 30.44


def clean_facts(d: dict) -> dict:
    """Drop None values so the facts the LLM sees contain only real numbers."""
    return {k: v for k, v in d.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}


def argext(s: pd.Series, kind: str = "max"):
    """Label of the largest (or smallest) value, or None when there is nothing to compare (empty or all NaN)."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    return s.idxmax() if kind == "max" else s.idxmin()
