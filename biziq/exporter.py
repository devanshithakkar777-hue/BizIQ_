"""Take everything out: one zip with a business's data as plain CSV files. Nothing here is locked in."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime

import pandas as pd

from . import billbook, data_layer, db

README = """BizIQ export for {org}
Exported {when}

datasets/   the tables behind your analysis (one CSV per dataset), exactly as BizIQ holds them
billbook/   your bill book: items, customers, bills, bill lines, payments, stock movements and expenses
manifest.json   how many rows each file has

These are ordinary CSV files that open in Excel, Google Sheets or any accounting tool. You do not need BizIQ to read them.
Text cells that start with = + - or @ have a leading ' so a spreadsheet shows them as text instead of running them as a formula.
"""


def _safe(df: pd.DataFrame) -> pd.DataFrame:
    """Stop spreadsheet formula injection: text that starts with = + - @ is shown as text, not executed."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or str(out[c].dtype).startswith("str"):
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") else v)
    return out


def build_zip(org_name: str, datasets: list[str] | None = None, include_billbook: bool = True) -> bytes:
    buf = io.BytesIO()
    manifest: dict[str, int] = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        data = data_layer.load_all()
        for name, df in data.items():
            if datasets is not None and name not in datasets:
                continue
            z.writestr(f"datasets/{name}.csv", _safe(df).to_csv(index=False))
            manifest[f"datasets/{name}.csv"] = len(df)
        if include_billbook:
            for name, df in billbook.export_tables().items():
                z.writestr(f"billbook/{name}.csv", _safe(df).to_csv(index=False))
                manifest[f"billbook/{name}.csv"] = len(df)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("README.txt", README.format(org=org_name, when=datetime.now().strftime("%d %b %Y %H:%M")))
    return buf.getvalue()
