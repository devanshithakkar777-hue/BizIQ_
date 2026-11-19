"""Reading and parsing real-world business files.

Handles what real exports do: title rows above the headings, "Total" rows at the bottom, Windows-1252/UTF-16 text,
semicolon or tab separators, Excel files with several sheets, dd/mm/yyyy dates, Excel date serials, Tally's YYYYMMDD dates,
Indian number formats, "(1,000)" negatives, and "-" or "N/A" for blank. Every judgement call is reported, never silent.
"""
from __future__ import annotations

import csv
import io
import os
import re
from datetime import date

import numpy as np
import pandas as pd

from . import mapping

MAX_UPLOAD_MB = float(os.environ.get("BIZIQ_MAX_UPLOAD_MB", "200"))
DATE_ORDER = os.environ.get("BIZIQ_DATE_ORDER", "dmy").lower()          # how to read 03/04/2026 when nothing says otherwise
PLACEHOLDERS = {"", "-", "--", "---", "na", "n/a", "nan", "nil", "null", "none", "#n/a", "n.a.", "not available"}
TOTAL_ROW = re.compile(r"^\s*(grand\s+|sub[\s-]*)?totals?\b|^\s*closing balance|^\s*opening balance|^\s*net total|^\s*report generated|^\s*printed on", re.I)


# ------------------------------------------------------------------ reading
def _bytes(file_or_path) -> bytes:
    if hasattr(file_or_path, "getvalue"):
        raw = file_or_path.getvalue()
    elif hasattr(file_or_path, "read"):
        raw = file_or_path.read()
    else:
        with open(file_or_path, "rb") as f:
            raw = f.read()
    raw = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(raw) > MAX_UPLOAD_MB * 1e6:
        raise ValueError(f"This file is {len(raw) / 1e6:.0f} MB; the limit is {MAX_UPLOAD_MB:.0f} MB. Split it by period and load the parts "
                         "with 'add new rows'.")
    return raw


def decode_text(raw: bytes) -> tuple[str, str]:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), "UTF-16"
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8"), "UTF-8"
    try:
        return raw.decode("utf-8"), "UTF-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "Windows-1252"


def sniff_sep(text: str) -> str:
    head = "\n".join(text.splitlines()[:30])
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {s: head.count(s) for s in ",;\t|"}
        return max(counts, key=counts.get) if max(counts.values()) else ","


def _score_header(row, dataset_targets, dataset) -> int:
    hits = 0
    for v in row:
        if isinstance(v, str) and v.strip():
            if any(mapping.score(v, dataset, t)[0] >= mapping.SUGGEST for t in dataset_targets):
                hits += 1
    return hits


def find_header_row(raw: pd.DataFrame, dataset: str | None, targets: list[str] | None) -> int:
    top = raw.head(40)
    if dataset and targets:
        scores = [_score_header(r, targets, dataset) for r in top.itertuples(index=False)]
        best = max(scores) if scores else 0
        if best >= 2:
            return scores.index(best)
    widths = top.notna().sum(axis=1)
    if widths.empty or widths.max() == 0:
        return 0
    return int(next(i for i, w in enumerate(widths) if w >= 0.6 * widths.max()))


class RowNumbers:
    """Row numbers kept in DataFrame.attrs. Immutable, so pandas' deep copies of attrs cost nothing, and equality is a plain bool
    (pandas compares attrs when concatenating; a bare numpy array there raises)."""
    __slots__ = ("arr",)

    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def __deepcopy__(self, memo):
        return self

    def __eq__(self, other):
        return isinstance(other, RowNumbers) and np.array_equal(self.arr, other.arr)

    __hash__ = None

    def __len__(self):
        return len(self.arr)


def _frame_from_raw(raw: pd.DataFrame, dataset, targets, notes: list[str]) -> pd.DataFrame:
    raw = raw.dropna(how="all")                 # the index keeps each row's original position in the file
    if raw.empty:
        raise ValueError("The file has no data.")
    pos = raw.index.to_numpy()
    raw = raw.reset_index(drop=True)
    h = find_header_row(raw, dataset, targets)
    if h:
        notes.append(f"Skipped {h} title row(s) above the column headings.")
    names, seen = [], {}
    for i, v in enumerate(raw.iloc[h]):
        nm = str(v).strip() if pd.notna(v) and str(v).strip() else f"Unnamed: {i}"
        seen[nm] = seen.get(nm, 0) + 1
        names.append(nm if seen[nm] == 1 else f"{nm}.{seen[nm] - 1}")
    df = raw.iloc[h + 1:].copy()
    df.index = pos[h + 1:]
    df.columns = names
    df = df.dropna(how="all")
    empty_cols = [c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()]
    df = df.drop(columns=empty_cols)
    lead = df.iloc[:, :3]
    first = lead.iloc[:, 0]
    for k in range(1, lead.shape[1]):                                              # first filled cell among the first three columns
        first = first.fillna(lead.iloc[:, k])
    first = first.astype("string").fillna("")
    tot = first.str.match(TOTAL_ROW).fillna(False).astype(bool)
    if tot.any():
        notes.append(f"Left out {int(tot.sum())} total/summary row(s) so they are not counted twice.")
        df = df[~tot]
    df.attrs["header_row"] = int(pos[h])
    df.attrs["source_rows"] = RowNumbers(df.index.to_numpy() + 1)       # 1-based row number in the user's file, per data row
    return df.reset_index(drop=True)


def list_sheets(file_or_path) -> list[str]:
    return pd.ExcelFile(io.BytesIO(_bytes(file_or_path))).sheet_names


def read_table(file_or_path, filename: str | None = None, dataset: str | None = None, targets: list[str] | None = None,
               sheet: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Read CSV/TSV/TXT/Excel into a DataFrame of raw values, plus notes about what was handled."""
    name = (filename or getattr(file_or_path, "name", None) or str(file_or_path)).lower()
    raw_bytes, notes = _bytes(file_or_path), []
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            xl = pd.ExcelFile(io.BytesIO(raw_bytes))
        except ImportError as e:
            raise ValueError("This Excel format needs an extra library. Save the file as .xlsx or CSV and upload it again.") from e
        if sheet is None:
            if len(xl.sheet_names) == 1:
                sheet = xl.sheet_names[0]
            else:                                               # pick the sheet that looks most like the dataset, else the biggest
                best, sheet = -1, xl.sheet_names[0]
                for sn in xl.sheet_names:
                    probe = xl.parse(sn, header=None, nrows=40)
                    sc = (_score_header_best(probe, dataset, targets) * 1000) + len(probe)
                    if sc > best:
                        best, sheet = sc, sn
                notes.append(f"The workbook has {len(xl.sheet_names)} sheets; read '{sheet}'. Pick another sheet if that is wrong.")
        raw = xl.parse(sheet, header=None, dtype=object)
        return _frame_from_raw(raw, dataset, targets, notes), notes
    text, enc = decode_text(raw_bytes)
    if enc == "Windows-1252":
        notes.append("The file is not UTF-8; read as Windows-1252. Check names with accents or the ₹ sign.")
    sep = "\t" if name.endswith(".tsv") else sniff_sep(text)
    if sep not in (","):
        notes.append(f"Separator detected: {'tab' if sep == chr(9) else repr(sep)}.")
    width = max((ln.count(sep) for ln in text.splitlines()), default=0) + 1
    raw = pd.read_csv(io.StringIO(text), sep=sep, header=None, names=list(range(width)), dtype=str, keep_default_na=False,
                      na_values=[""], skip_blank_lines=False, engine="c", on_bad_lines="skip", quotechar='"')
    return _frame_from_raw(raw, dataset, targets, notes), notes


def _score_header_best(probe: pd.DataFrame, dataset, targets) -> int:
    if not (dataset and targets):
        return 0
    return max((_score_header(r, targets, dataset) for r in probe.itertuples(index=False)), default=0)


# ------------------------------------------------------------------ numbers
_CURRENCY = re.compile(r"(₹|rs\.?|inr|usd|\$|€|£)", re.I)
_DRCR = re.compile(r"\s*(?:dr|cr)\.?\s*$", re.I)


def parse_numbers(s: pd.Series) -> tuple[pd.Series, dict]:
    """Returns (numbers, info). info: bad (count), examples (a few unreadable raw values), drcr (count with a Dr/Cr suffix)."""
    info = {"bad": 0, "examples": [], "drcr": 0}
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce"), info
    conv = pd.to_numeric(s, errors="coerce")
    todo = conv.isna() & s.notna()
    if todo.any():
        t = s[todo].astype(str).str.replace(" ", " ", regex=False).str.strip()
        blank = t.str.lower().isin(PLACEHOLDERS)
        t = t[~blank]
        info["drcr"] = int(t.str.contains(_DRCR).sum())
        t = t.str.replace(_DRCR, "", regex=True)
        neg = t.str.match(r"^\(.*\)$") | t.str.endswith("-") | t.str.match(r"^-")
        t = t.str.replace(r"^\((.*)\)$", r"\1", regex=True).str.replace(_CURRENCY, "", regex=True)
        t = t.str.replace(r"[,\s%]", "", regex=True).str.replace(r"^-|-$", "", regex=True)
        vals = pd.to_numeric(t, errors="coerce")
        vals = vals.where(~neg, -vals)
        conv.loc[vals.index] = vals
        badmask = vals.isna()
        info["bad"] = int(badmask.sum())
        info["examples"] = list(s.loc[badmask[badmask].index].astype(str).head(3))
    return conv, info


# ------------------------------------------------------------------ dates
_AMBIG = re.compile(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(?!\d)")
_YMD8 = re.compile(r"^\d{8}$")
_SERIAL = re.compile(r"^\d{5}(\.\d+)?$")


def parse_dates(s: pd.Series, order: str | None = None) -> tuple[pd.Series, dict]:
    """Returns (datetimes, info). info: bad, examples, order ('dmy'|'mdy'|None), ambiguous (count read by assumption),
    conflict (both day-first and month-first patterns present), implausible (outside 1990 .. today + 2 years)."""
    order = (order or DATE_ORDER)
    info = {"bad": 0, "examples": [], "order": None, "ambiguous": 0, "conflict": False, "implausible": 0}
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, info
    t = s.astype("string").str.strip()
    t = t.mask(t.str.lower().isin(PLACEHOLDERS))
    uniq = pd.Series(t.dropna().unique(), dtype="string")
    out = pd.Series(pd.NaT, index=uniq.index, dtype="datetime64[ns]")
    if len(uniq):
        ymd8, serial = uniq.str.match(_YMD8), uniq.str.match(_SERIAL)
        if ymd8.any():
            out[ymd8] = pd.to_datetime(uniq[ymd8], format="%Y%m%d", errors="coerce")
        if serial.any():
            n = pd.to_numeric(uniq[serial], errors="coerce")
            ok = n.between(20000, 80000)
            out[ok[ok].index] = pd.to_datetime(n[ok], unit="D", origin="1899-12-30", errors="coerce")
        rest = ~(ymd8 | serial)
        amb = uniq[rest].str.extract(_AMBIG)
        has = amb[0].notna()
        a, b = pd.to_numeric(amb[0], errors="coerce"), pd.to_numeric(amb[1], errors="coerce")
        first_big, second_big = bool((a > 12).any()), bool((b > 12).any())
        if first_big and second_big:
            info["conflict"] = True
        dayfirst = (order == "dmy") if not (first_big or second_big) else (first_big or (not second_big))
        if info["conflict"]:
            dayfirst = int((a > 12).sum()) >= int((b > 12).sum())
        if has.any():
            info["order"] = "dmy" if dayfirst else "mdy"
            if not (first_big or second_big):
                info["ambiguous"] = int(((a <= 12) & (b <= 12) & has).sum())
        r_idx = uniq[rest].index
        ymd = uniq[rest].str.match(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
        if ymd.any():                                           # year first is unambiguous
            out[ymd[ymd].index] = pd.to_datetime(uniq[rest][ymd].str.replace(r"[/.]", "-", regex=True), errors="coerce", format="mixed")
        other = rest & ~uniq.index.isin(ymd[ymd].index)
        if other.any():
            out[uniq[other].index] = pd.to_datetime(uniq[other], errors="coerce", dayfirst=dayfirst, format="mixed")
    mp = dict(zip(uniq.astype(str), out))
    res = t.astype(object).map(lambda v: mp.get(v, pd.NaT) if v is not pd.NA and v == v else pd.NaT)
    res = pd.to_datetime(res, errors="coerce")
    badmask = t.notna() & res.isna()
    info["bad"], info["examples"] = int(badmask.sum()), list(s[badmask].astype(str).head(3))
    lo, hi = pd.Timestamp("1990-01-01"), pd.Timestamp(date.today()) + pd.Timedelta(days=730)
    info["implausible"] = int(((res < lo) | (res > hi)).sum())
    return res, info
