"""Connectors: one common way to bring data into BizIQ.

A connector's only job is to return a DataFrame for a dataset name. Everything after that (cleaning, strict
validation, storage, audit) is the same for every source, so adding a new source never touches the analytics.

What exists today
  * FileConnector       - CSV, TSV or Excel file (path or uploaded file object).
  * ColumnMapConnector  - wraps another connector and renames the source's column headings to BizIQ's.
  * PRESETS             - starting-point column maps for common exports (Shopify orders, a Tally-style sales register,
                          a generic invoice export). The headings are typical, NOT guaranteed for every account or
                          version: open your export, compare, and adjust the map. Test with your own file.

Live API connectors (Shopify, Zoho Books, Tally, WhatsApp) are in biziq/live/. myBillBook has no public API, so it is
supported through its report downloads (presets below) and its Tally-compatible XML export (biziq/live/tally.py).
Not built: GST portal, marketplaces, other accounting tools.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from . import audit, data_layer


class Connector(ABC):
    name: str = "connector"

    @abstractmethod
    def fetch(self, dataset: str) -> pd.DataFrame:
        """Return the raw table for `dataset` (any column names; mapping and validation happen later)."""


class FileConnector(Connector):
    def __init__(self, file_or_path, filename: str | None = None):
        self.file_or_path, self.filename = file_or_path, filename
        self.name = f"file:{filename or getattr(file_or_path, 'name', str(file_or_path))}"

    def fetch(self, dataset: str) -> pd.DataFrame:
        return data_layer.read_upload(self.file_or_path, self.filename)


class ColumnMapConnector(Connector):
    """Rename source columns to BizIQ column names. Matching ignores case and surrounding spaces."""

    def __init__(self, inner: Connector, mapping: dict[str, str], name: str | None = None, defaults: dict | None = None):
        self.inner, self.mapping, self.defaults = inner, mapping, defaults or {}
        self.name = name or f"mapped:{inner.name}"

    def fetch(self, dataset: str) -> pd.DataFrame:
        df = self.inner.fetch(dataset)
        lookup = {k.strip().lower(): v for k, v in self.mapping.items()}
        df = df.rename(columns={c: lookup[str(c).strip().lower()] for c in df.columns
                                if str(c).strip().lower() in lookup})
        for col, val in self.defaults.items():          # fill a column the export has no equivalent for
            if col not in df.columns:
                df[col] = val
        return df


# Typical export headings -> BizIQ columns. Starting points only (see module docstring).
PRESETS: dict[str, dict] = {
    "shopify_orders": {
        "dataset": "sales",
        "map": {"Name": "order_id", "Created at": "date", "Email": "customer_id", "Lineitem name": "product",
                "Lineitem quantity": "quantity", "Lineitem price": "unit_price", "Shipping Province": "region"},
    },
    "tally_sales_register": {
        "dataset": "sales",
        "map": {"Voucher No.": "order_id", "Date": "date", "Particulars": "customer_id", "Item": "product",
                "Quantity": "quantity", "Rate": "unit_price", "State": "region"},
    },
    # myBillBook has no public API; these map its downloadable reports. The headings are GUESSES from typical exports:
    # open your own file and compare before trusting them.
    "mybillbook_sales_report": {
        "dataset": "sales",
        "note": "Headings are unverified guesses. Check the preview, and fix the file's headings if columns are missing.",
        "map": {"Invoice No": "order_id", "Invoice No.": "order_id", "Invoice Number": "order_id", "Date": "date",
                "Invoice Date": "date", "Party Name": "customer_id", "Customer Name": "customer_id", "Item Name": "product",
                "Item": "product", "Quantity": "quantity", "Qty": "quantity", "Rate": "unit_price", "Price": "unit_price",
                "Sale Price": "unit_price", "State": "region", "Place of Supply": "region", "Category": "category"},
    },
    "mybillbook_invoices_report": {
        "dataset": "invoices",
        "note": "Headings are unverified guesses. A due date column is required; add one if the report has none.",
        "map": {"Invoice No": "invoice_id", "Invoice No.": "invoice_id", "Invoice Number": "invoice_id", "Party Name": "party",
                "Date": "invoice_date", "Invoice Date": "invoice_date", "Due Date": "due_date", "Total Amount": "amount",
                "Invoice Amount": "amount", "Amount": "amount"},
        "defaults": {"type": "receivable"},
    },
    "generic_invoice_export": {
        "dataset": "invoices",
        "map": {"Invoice Number": "invoice_id", "Customer Name": "party", "Invoice Date": "invoice_date",
                "Due Date": "due_date", "Total": "amount", "Payment Date": "paid_date"},
        "defaults": {"type": "receivable"},      # a sales-invoice export; for a bills export change this to "payable"
    },
}


def from_preset(file_or_path, preset: str, filename: str | None = None) -> tuple[ColumnMapConnector, str]:
    p = PRESETS[preset]
    return ColumnMapConnector(FileConnector(file_or_path, filename), p["map"], name=f"preset:{preset}",
                              defaults=p.get("defaults")), p["dataset"]


def ingest_from(connector: Connector, dataset: str, user=None) -> data_layer.ValidationReport:
    """Fetch -> clean and validate -> store. Writes an audit entry when a user is given."""
    df = connector.fetch(dataset)
    rep = data_layer.ingest(df, dataset, source=connector.name)
    if user is not None:
        audit.log(user.org_id, user.email, "data.upload", dataset,
                  f"{connector.name}; rows in {rep.rows_in}, rows kept {rep.rows_out}, ok={rep.ok}")
    return rep
