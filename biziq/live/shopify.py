"""Shopify orders -> BizIQ `sales`, through the GraphQL Admin API (REST is legacy for new apps).

Needs a custom-app Admin API access token with the read_orders scope. Shopify only returns the last 60 days of orders
unless the app has been granted read_all_orders. Set api_version to a version your store supports.
"""
from __future__ import annotations

import re
import time
from datetime import date, timedelta

import pandas as pd

from ..connectors import Connector
from .http import Http, HttpError

SHOP_RE = re.compile(r"^[a-z0-9][a-z0-9-]*\.myshopify\.com$")
DEFAULT_VERSION = "2026-07"

QUERY = """
query Orders($after: String, $q: String) {
  orders(first: 50, after: $after, query: $q, sortKey: CREATED_AT) {
    pageInfo { hasNextPage endCursor }
    nodes {
      name createdAt cancelledAt
      customer { id }
      shippingAddress { provinceCode province }
      lineItems(first: 100) {
        pageInfo { hasNextPage }
        nodes { title quantity discountedUnitPriceSet { shopMoney { amount } } product { productType } }
      }
    }
  }
}"""


class ShopifyConnector(Connector):
    datasets = ["sales"]

    def __init__(self, shop: str, token: str, api_version: str = DEFAULT_VERSION, since_days: int = 365,
                 http: Http | None = None, sleep=time.sleep):
        shop = (shop or "").strip().lower().replace("https://", "").rstrip("/")
        if not SHOP_RE.match(shop):                     # the token is sent to this host, so it must be a real Shopify domain
            raise ValueError("Shop address must look like your-store.myshopify.com")
        if not token:
            raise ValueError("Shopify access token is missing.")
        self.shop, self.token, self.version, self.since_days = shop, token, api_version, int(since_days)
        self.http, self.sleep, self.notes = http or Http(), sleep, []
        self.name = f"live:shopify:{shop}"

    def _gql(self, variables: dict) -> dict:
        url = f"https://{self.shop}/admin/api/{self.version}/graphql.json"
        for attempt in range(6):
            r = self.http.request("POST", url, headers={"X-Shopify-Access-Token": self.token},
                                  json_body={"query": QUERY, "variables": variables}).json()
            errs = r.get("errors") or []
            if errs and any((e.get("extensions") or {}).get("code") == "THROTTLED" for e in errs if isinstance(e, dict)):
                self.sleep(2 * (attempt + 1))
                continue
            if errs:
                msg = "; ".join(str(e.get("message", e)) if isinstance(e, dict) else str(e) for e in errs)[:300]
                raise HttpError(f"Shopify rejected the query: {msg}")
            return r["data"]["orders"]
        raise HttpError("Shopify kept throttling the request; try again later.")

    def test(self) -> str:
        orders = self._gql({"after": None, "q": None})
        return f"Connected to {self.shop}. Orders are readable ({len(orders['nodes'])} on the first page)."

    def fetch(self, dataset: str = "sales") -> pd.DataFrame:
        if dataset != "sales":
            raise ValueError("Shopify provides the sales dataset only.")
        since = (date.today() - timedelta(days=self.since_days)).isoformat()
        rows, after, pages, cut, cancelled = [], None, 0, 0, 0
        while True:
            page = self._gql({"after": after, "q": f"created_at:>={since}"})
            for o in page["nodes"]:
                if o.get("cancelledAt"):
                    cancelled += 1
                    continue
                items = o["lineItems"]
                if items["pageInfo"]["hasNextPage"]:
                    cut += 1
                cust = ((o.get("customer") or {}).get("id") or "guest").rsplit("/", 1)[-1]
                addr = o.get("shippingAddress") or {}
                region = addr.get("province") or addr.get("provinceCode") or "Unknown"
                for li in items["nodes"]:
                    price = ((li.get("discountedUnitPriceSet") or {}).get("shopMoney") or {}).get("amount")
                    rows.append(dict(order_id=o["name"], date=o["createdAt"][:10], customer_id=cust, product=li["title"],
                                     region=region, quantity=li["quantity"], unit_price=float(price or 0),
                                     category=(li.get("product") or {}).get("productType") or None))
            pages += 1
            if not page["pageInfo"]["hasNextPage"] or pages >= 400:
                break
            after = page["pageInfo"]["endCursor"]
        if cancelled:
            self.notes.append(f"{cancelled} cancelled orders were left out.")
        if cut:
            self.notes.append(f"{cut} orders have more than 100 line items; only the first 100 were read.")
        if pages >= 400:
            self.notes.append("Stopped after 400 pages of orders; narrow the date window.")
        return pd.DataFrame(rows)
