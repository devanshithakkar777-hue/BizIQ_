"""Module registry and the BizIQ module contract.

Every analytics module that plugs into the platform is described by a ModuleSpec and registered here.
The pipeline runs registered modules after the built-in ones, skips a module when its data is missing,
and isolates failures so one broken module never takes the dashboard down.

THE CONTRACT (what a module must do)
  1. needs      - names of datasets (see data_layer.SCHEMAS) the module requires. Missing -> skipped.
  2. run        - run(data, ctx) -> ModuleResult. Pure function of its inputs: no network, no clock reads
                  (use ctx.as_of / ctx.today), no writes.
  3. facts      - every number the decision layer may quote goes in ModuleResult.facts. Nothing else is quoted.
  4. flags      - red / amber / green findings with a plain-language title.
  5. recommend  - recommend(facts, flags) -> list[str]; templates over computed facts only.
  6. sensitive  - True for modules about people (salary, performance, hiring). They require the `view_people`
                  permission and must never make automated decisions about an individual.
  7. tests      - covered by tests/test_contract.py, which runs every registered module against sample data
                  and checks this contract automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .core import LEVEL_ORDER, ModuleResult

AUDIENCES = ("B2B", "B2C", "Both")


@dataclass
class Context:
    """Read-only run context handed to every plugin module."""
    as_of: pd.Timestamp
    today: pd.Timestamp
    modules: dict[str, ModuleResult] = field(default_factory=dict)   # results of modules that ran earlier
    settings: dict = field(default_factory=dict)


@dataclass
class ModuleSpec:
    name: str
    group: str                      # e.g. "Finance and compliance"
    blurb: str                      # one line shown at the top of the page
    needs: list[str]
    run: Callable[[dict, Context], ModuleResult]
    recommend: Callable[[dict, list], list[str]]
    audience: str = "Both"          # B2B | B2C | Both
    phase: str = "2"
    sensitive: bool = False
    depends_on: list[str] = field(default_factory=list)   # modules whose results this one reads (optional)

    def __post_init__(self):
        if self.audience not in AUDIENCES:
            raise ValueError(f"{self.name}: audience must be one of {AUDIENCES}")
        if not self.name.endswith("IQ"):
            raise ValueError(f"{self.name}: module names end in 'IQ'")


REGISTRY: dict[str, ModuleSpec] = {}


def register(spec: ModuleSpec) -> ModuleSpec:
    if spec.name in REGISTRY:
        raise ValueError(f"Module {spec.name} is already registered")
    REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> ModuleSpec | None:
    _load()
    return REGISTRY.get(name)


def all_specs() -> list[ModuleSpec]:
    _load()
    return list(REGISTRY.values())


def _load() -> None:
    """Import the plugin modules once so they register themselves."""
    if not REGISTRY:
        from . import modules  # noqa: F401  (side effect: registers every module)


def check_result(spec: ModuleSpec, res: ModuleResult) -> list[str]:
    """Contract check used by the test-suite and the pipeline. Returns a list of problems (empty = ok)."""
    problems = []
    if not isinstance(res, ModuleResult):
        return [f"{spec.name}: run() must return a ModuleResult"]
    if res.name != spec.name:
        problems.append(f"{spec.name}: result name is '{res.name}'")
    if not res.facts:
        problems.append(f"{spec.name}: no facts, so the decision layer has nothing it may quote")
    for f in res.flags:
        if f.level not in LEVEL_ORDER:
            problems.append(f"{spec.name}: invalid flag level '{f.level}'")
        if not f.title:
            problems.append(f"{spec.name}: flag with empty title")
    for k, v in res.facts.items():
        if isinstance(v, float) and pd.isna(v):
            problems.append(f"{spec.name}: fact '{k}' is NaN (use None or omit it)")
    for k, t in res.tables.items():
        if not isinstance(t, pd.DataFrame):
            problems.append(f"{spec.name}: table '{k}' is not a DataFrame")
    return problems


def generic_summary(res: ModuleResult, n_facts: int = 6) -> str:
    """Short plain-language summary built only from a result's own flags and facts."""
    parts = []
    flags = sorted(res.flags, key=lambda f: LEVEL_ORDER[f.level])
    if flags:
        parts.append(flags[0].title + ".")
    shown = list(res.facts.items())[:n_facts]
    if shown:
        parts.append("Key figures: " + "; ".join(f"{k.replace('_', ' ')} {v}" for k, v in shown) + ".")
    return " ".join(parts)
