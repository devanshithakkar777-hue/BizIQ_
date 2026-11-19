"""Shared result types and formatting helpers used by every BizIQ module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

RED, AMBER, GREEN = "red", "amber", "green"
LEVEL_ORDER = {RED: 0, AMBER: 1, GREEN: 2}


@dataclass
class Flag:
    level: str          # red | amber | green
    title: str
    detail: str = ""
    module: str = ""

    def as_dict(self) -> dict:
        return dict(level=self.level, module=self.module, title=self.title, detail=self.detail)


@dataclass
class ModuleResult:
    """What every analytics module returns.

    metrics  - headline numbers (floats / ints / short strings)
    tables   - named DataFrames for display
    flags    - red / amber / green findings
    facts    - computed numbers the decision layer may quote. The LLM only ever
               sees this dict, never raw data, and every number it writes is
               checked against it.
    score    - optional 0-100 score
    """
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    notes: list[str] = field(default_factory=list)

    def add_flag(self, level: str, title: str, detail: str = "") -> None:
        self.flags.append(Flag(level, title, detail, self.name))

    def worst_level(self) -> str:
        if not self.flags:
            return GREEN
        return min((f.level for f in self.flags), key=lambda x: LEVEL_ORDER[x])


def inr(x: float, compact: bool = True) -> str:
    """Format a number as Indian rupees (lakh / crore aware)."""
    if x is None or pd.isna(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    v = abs(float(x))
    if compact:
        if v >= 1e7:
            return f"{sign}Rs {v / 1e7:.2f} Cr"
        if v >= 1e5:
            return f"{sign}Rs {v / 1e5:.2f} L"
    return f"{sign}Rs {v:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def safe_div(a, b, default=float("nan")):
    try:
        if b is None or pd.isna(b) or b == 0:
            return default
        return a / b
    except Exception:
        return default


def score_between(value: float, bad: float, good: float) -> float:
    """Linear 0-100 score: `bad` -> 0, `good` -> 100 (works for either direction)."""
    if value is None or pd.isna(value):
        return float("nan")
    if good == bad:
        return 100.0
    s = (value - bad) / (good - bad) * 100
    return float(max(0.0, min(100.0, s)))
