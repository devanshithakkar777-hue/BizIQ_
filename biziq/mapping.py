"""Match the column headings in a business's own export to BizIQ's column names.

Real exports say "Invoice No.", "Party Name", "Amount (Rs)", not "invoice_id", "party", "amount". This module suggests a
mapping and says how sure it is. Exact names and well-known aliases are applied automatically; fuzzy matches are only
suggested, for a person to confirm. Aliases are common Indian accounting/ERP/billing-software headings; extend freely.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

def _snake(c) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


AUTO = 0.95          # exact name or known alias: safe to apply without asking
SUGGEST = 0.86       # similar enough to offer for confirmation


def norm(s) -> str:
    s = str(s).lower().replace("&", " and ").replace("₹", " ")
    s = re.sub(r"\(.*?\)", " ", s)                      # drop units: "Amount (Rs)" -> "amount"
    return re.sub(r"[^a-z0-9]", "", s)


_COMMON = {
    "date": ["date", "txn date", "transaction date", "order date", "voucher date", "sale date", "bill date", "invoice date", "posting date", "created at"],
    "order_id": ["order id", "order no", "order number", "order", "invoice no", "invoice number", "bill no", "bill number", "voucher no",
                 "voucher number", "txn id", "transaction id", "sales order", "document no", "doc no", "ref no", "name"],
    "customer_id": ["customer", "customer id", "customer name", "customer code", "party", "party name", "client", "client name", "buyer",
                    "ledger", "ledger name", "particulars", "account name", "bill to"],
    "product": ["product", "product name", "item", "item name", "sku", "item description", "description", "article", "material", "goods"],
    "region": ["region", "state", "place of supply", "zone", "city", "location", "branch", "area", "ship to state"],
    "quantity": ["quantity", "qty", "units", "units sold", "nos", "quantity sold", "billed qty", "no of units", "order quantity", "order qty", "quantity ordered"],
    "unit_price": ["unit price", "price", "rate", "selling price", "sale price", "price per unit", "item rate", "unit rate", "selling rate"],
    "revenue": ["revenue", "amount", "total", "sales amount", "line total", "net amount", "item total", "net sales", "taxable amount", "sales", "sales value"],
    "category": ["category", "product category", "item category", "group", "item group", "product group", "segment"],
}
ALIASES: dict[str, dict[str, list[str]]] = {
    "sales": _COMMON,
    "invoices": {
        "invoice_id": ["invoice id", "invoice no", "invoice number", "bill no", "bill number", "document no", "doc no", "ref no", "voucher no", "voucher number", "invoice #"],
        "party": ["party", "party name", "customer", "customer name", "vendor", "vendor name", "supplier", "supplier name", "ledger", "ledger name", "client", "account name", "name", "particulars"],
        "type": ["type", "invoice type", "voucher type", "document type", "transaction type", "bill type", "kind"],
        "invoice_date": ["invoice date", "bill date", "date", "document date", "voucher date", "issue date", "created date"],
        "due_date": ["due date", "payment due date", "due on", "due", "payment due", "pay by", "maturity date"],
        "amount": ["amount", "invoice amount", "total", "total amount", "bill amount", "gross total", "grand total", "invoice value", "net amount", "total value"],
        "paid_date": ["paid date", "payment date", "date paid", "settled on", "received date", "last payment date", "paid on", "receipt date"],
    },
    "financials": {
        "month": ["month", "period", "date", "month end", "as of", "as at", "reporting month"],
        "revenue": ["revenue", "sales", "turnover", "total revenue", "net sales", "revenue from operations", "income"],
        "cogs": ["cogs", "cost of goods sold", "cost of sales", "direct costs", "cost of materials", "purchases"],
        "operating_expenses": ["operating expenses", "opex", "indirect expenses", "expenses", "operating costs", "overheads"],
        "interest_expense": ["interest expense", "interest", "finance cost", "finance costs", "interest paid", "interest on loans"],
        "tax": ["tax", "income tax", "provision for tax", "tax expense", "taxes"],
        "net_income": ["net income", "net profit", "profit after tax", "pat", "net profit loss", "profit for the period", "net earnings"],
        "cash": ["cash", "cash and bank", "cash bank balances", "cash and cash equivalents", "bank balance", "cash in hand and at bank"],
        "receivables": ["receivables", "accounts receivable", "sundry debtors", "trade receivables", "debtors"],
        "inventory": ["inventory", "stock", "closing stock", "stock in trade", "inventories"],
        "total_assets": ["total assets", "assets total", "total of assets"],
        "current_liabilities": ["current liabilities", "total current liabilities"],
        "total_liabilities": ["total liabilities", "liabilities total", "total of liabilities"],
        "equity": ["equity", "shareholders funds", "shareholders equity", "net worth", "capital", "owners equity", "total equity"],
        "retained_earnings": ["retained earnings", "reserves and surplus", "reserves", "profit and loss account"],
        "operating_cash_flow": ["operating cash flow", "cfo", "cash from operations", "net cash from operating activities", "cash flow from operations"],
        "long_term_debt": ["long term debt", "long term borrowings", "term loans", "non current borrowings", "secured loans"],
        "short_term_debt": ["short term debt", "short term borrowings", "working capital loans", "od cc", "current borrowings", "bank overdraft"],
        "payables": ["payables", "accounts payable", "sundry creditors", "trade payables", "creditors"],
        "depreciation": ["depreciation", "depreciation and amortisation", "depreciation and amortization", "d and a"],
        "fixed_assets": ["fixed assets", "net block", "property plant and equipment", "ppe", "tangible assets"],
        "other_current_assets": ["other current assets", "other assets"],
    },
    "inventory": {
        "product": ["product", "item", "item name", "sku", "product name", "description"],
        "stock_on_hand": ["stock on hand", "stock", "closing stock", "quantity on hand", "qty on hand", "current stock", "available", "balance qty", "on hand", "closing qty"],
        "lead_time_days": ["lead time", "lead time days", "lead days", "delivery lead time"],
        "avg_daily_demand": ["avg daily demand", "daily demand", "average daily sales", "avg daily sales"],
    },
    "supplier_deliveries": {
        "supplier": ["supplier", "supplier name", "vendor", "vendor name", "party"],
        "product": ["product", "item", "item name", "material", "sku"],
        "order_date": ["order date", "po date", "purchase order date", "date ordered"],
        "promised_date": ["promised date", "expected date", "due date", "promised delivery date", "expected delivery"],
        "delivered_date": ["delivered date", "delivery date", "received date", "grn date", "actual delivery date"],
        "qty_ordered": ["qty ordered", "quantity ordered", "ordered qty", "order qty", "po qty"],
        "qty_received": ["qty received", "quantity received", "received qty", "grn qty", "delivered qty"],
        "unit_price": ["unit price", "price", "rate", "purchase rate", "unit rate"],
        "defective_units": ["defective units", "defects", "rejected qty", "rejected quantity", "damaged units"],
        "delivery_id": ["delivery id", "grn no", "grn number", "po no", "po number"],
    },
    "hr": {
        "employee_id": ["employee id", "emp id", "emp code", "employee code", "employee no", "staff id", "employee number"],
        "department": ["department", "dept", "division", "team", "function"],
        "monthly_salary": ["monthly salary", "salary", "gross salary", "monthly ctc", "monthly gross", "gross pay"],
        "join_date": ["join date", "date of joining", "doj", "joining date", "hire date", "start date"],
        "exit_date": ["exit date", "date of leaving", "dol", "last working day", "separation date", "relieving date"],
        "performance_rating": ["performance rating", "rating", "appraisal rating", "performance score"],
        "overtime_hours_month": ["overtime hours month", "overtime hours", "ot hours", "overtime"],
        "satisfaction_score": ["satisfaction score", "engagement score", "satisfaction", "esat"],
    },
    "returns": {
        "return_id": ["return id", "return no", "credit note no", "rma", "rma no", "return number"],
        "order_id": _COMMON["order_id"], "date": ["date", "return date", "credit note date", "refund date"],
        "product": _COMMON["product"],
        "refund_amount": ["refund amount", "refund", "credit note amount", "amount refunded", "amount"],
        "reason": ["reason", "return reason", "remarks"], "channel": ["channel", "sales channel", "source"],
        "is_cod": ["is cod", "cod", "cash on delivery"], "quantity": _COMMON["quantity"],
    },
    "expenses": {
        "expense_id": ["expense id", "voucher no", "voucher number", "ref no", "txn id", "bill no"],
        "date": ["date", "expense date", "voucher date", "txn date", "payment date"],
        "category": ["category", "expense category", "head", "ledger", "expense head", "account", "expense type"],
        "vendor": ["vendor", "vendor name", "payee", "party", "party name", "supplier", "paid to"],
        "amount": ["amount", "total", "expense amount", "debit", "value"],
        "department": ["department", "dept", "cost centre", "cost center"], "payment_mode": ["payment mode", "mode", "mode of payment", "payment method"],
        "approved_by": ["approved by", "approver"], "receipt_attached": ["receipt attached", "has receipt", "bill attached"],
    },
    "campaigns": {
        "date": ["date", "day", "reporting starts", "report date"], "channel": ["channel", "platform", "source", "network"],
        "spend": ["spend", "cost", "amount spent", "amount spent inr", "ad spend", "total spend"],
        "impressions": ["impressions", "impr"], "clicks": ["clicks", "link clicks", "all clicks"],
        "conversions": ["conversions", "results", "purchases", "leads", "orders"],
        "campaign": ["campaign", "campaign name", "ad set name", "ad name"],
        "revenue": ["revenue", "conversion value", "purchase conversion value", "sales", "purchases conversion value"],
    },
    "pipeline": {
        "deal_id": ["deal id", "opportunity id", "lead id", "id", "record id"], "account": ["account", "account name", "company", "customer", "organisation", "organization"],
        "stage": ["stage", "deal stage", "pipeline stage", "status"], "amount": ["amount", "deal value", "value", "expected revenue", "opportunity amount"],
        "created_date": ["created date", "created on", "create date", "date created", "created"],
        "last_activity_date": ["last activity date", "last activity", "last contacted", "last modified", "last touched"],
        "expected_close": ["expected close", "close date", "expected close date", "closing date"],
        "owner": ["owner", "deal owner", "sales rep", "assigned to", "salesperson"], "source": ["source", "lead source"],
    },
    "quotes": {
        "quote_id": ["quote id", "quote no", "quotation no", "quotation number", "estimate no", "quote number"],
        "customer": ["customer", "customer name", "client", "party", "party name"],
        "sent_date": ["sent date", "date", "quote date", "quotation date", "estimate date"],
        "amount": ["amount", "quote amount", "total", "quotation value", "value"], "status": ["status", "quote status", "outcome"],
        "cost": ["cost", "estimated cost"], "lost_reason": ["lost reason", "reason lost", "loss reason"],
        "decided_date": ["decided date", "closed date", "decision date", "accepted date"],
    },
    "contracts": {
        "contract_id": ["contract id", "contract no", "agreement no", "contract number"], "party": ["party", "party name", "counterparty", "customer", "vendor", "client"],
        "end_date": ["end date", "expiry date", "expiry", "valid till", "valid until", "contract end"], "value": ["value", "contract value", "annual value", "amount"],
        "start_date": ["start date", "effective date", "contract start"], "renewal_notice_days": ["renewal notice days", "notice period days", "notice days", "notice period"],
        "auto_renew": ["auto renew", "auto renewal", "autorenew"], "escalation_pct": ["escalation pct", "escalation", "annual escalation", "escalation percent"],
        "type": ["type", "contract type"],
    },
    "tickets": {
        "ticket_id": ["ticket id", "ticket no", "ticket number", "case id", "case number", "id"], "opened_date": ["opened date", "created date", "created", "date opened", "open date", "created at"],
        "category": ["category", "issue type", "ticket type", "topic", "type"], "priority": ["priority", "severity", "urgency"],
        "closed_date": ["closed date", "resolved date", "date closed", "resolution date", "resolved at"], "csat": ["csat", "satisfaction", "rating", "customer rating", "csat score"],
        "channel": ["channel", "source", "contact channel"],
    },
    "gst_books": {
        "invoice_no": ["invoice no", "invoice number", "bill no", "document number", "inv no", "doc no", "supplier invoice no"],
        "party_gstin": ["party gstin", "gstin", "gstin of supplier", "supplier gstin", "gstin/uin", "gstin uin", "customer gstin", "gstin no"],
        "invoice_date": ["invoice date", "date", "document date", "bill date"],
        "taxable_value": ["taxable value", "taxable amount", "taxable", "assessable value", "taxable val"],
        "gst_amount": ["gst amount", "total tax", "tax amount", "total gst", "igst cgst sgst", "tax", "gst"],
        "party": ["party", "party name", "supplier", "supplier name", "trade name", "vendor", "legal name"],
    },
}
ALIASES["gstr2b"] = ALIASES["gst_books"]
ALIASES["social"] = {"date": ["date", "post date", "published", "posted on"], "platform": ["platform", "channel", "network"], "post_type": ["post type", "type", "format", "media type"],
                     "impressions": ["impressions", "reach", "views"], "likes": ["likes", "reactions"], "comments": ["comments"], "shares": ["shares", "reposts"],
                     "clicks": ["clicks", "link clicks"], "theme": ["theme", "topic", "campaign"]}
ALIASES["reviews"] = {"date": ["date", "review date", "created at", "posted"], "rating": ["rating", "stars", "score", "review rating"],
                      "review_text": ["review text", "review", "comment", "text", "feedback", "body"], "product": _COMMON["product"]}
ALIASES["product_costs"] = {"product": _COMMON["product"], "unit_cost": ["unit cost", "cost", "cost price", "purchase price", "cogs per unit", "landing cost"],
                            "channel_fee_pct": ["channel fee pct", "marketplace fee", "commission pct", "channel fee"],
                            "shipping_cost_per_order": ["shipping cost per order", "shipping cost", "delivery cost"], "packaging_cost": ["packaging cost", "packaging"]}
ALIASES["competitor_prices"] = {"date": ["date", "observed on", "checked on"], "product": _COMMON["product"], "competitor": ["competitor", "competitor name", "seller", "brand"],
                                "price": ["price", "competitor price", "selling price", "listed price"]}
ALIASES["credit_limits"] = {"customer": ["customer", "customer name", "party", "party name", "client"], "credit_limit": ["credit limit", "limit", "credit amount"],
                            "payment_terms_days": ["payment terms days", "credit days", "payment terms", "credit period", "terms days"]}
ALIASES["portfolio"] = {"instrument": ["instrument", "security", "scrip", "symbol", "holding", "name", "fund name"], "asset_class": ["asset class", "category", "type", "asset type"],
                        "units": ["units", "quantity", "qty", "holding units"], "buy_price": ["buy price", "average cost", "avg cost", "purchase price", "avg buy price"],
                        "current_price": ["current price", "ltp", "nav", "market price", "cmp"]}


def _alias_norms(dataset: str, target: str) -> set[str]:
    return {norm(a) for a in ALIASES.get(dataset, {}).get(target, [])} | {norm(target)}


def score(source: str, dataset: str, target: str) -> tuple[float, str]:
    n = norm(source)
    if not n:
        return 0.0, ""
    if n == norm(target):
        return 1.0, "exact"
    aliases = _alias_norms(dataset, target)
    if n in aliases:
        return AUTO, "known alias"
    best = max((SequenceMatcher(None, n, a).ratio() for a in aliases), default=0.0)
    words = set(n.split())
    for a in aliases:                      # "order quantity" contains "quantity": worth suggesting, never worth applying unasked
        aw = set(a.split())
        if aw and len(aw) < len(words) + 1 and (aw < words or words < aw) and min(len(aw), len(words)) >= 1 and len(a) > 3:
            best = max(best, SUGGEST + 0.02)
    return (best, "similar") if best >= SUGGEST else (0.0, "")


def suggest(columns: list, dataset: str, targets: list[str]) -> list[dict]:
    """Best one-to-one matches: [{source, target, score, method}], strongest first."""
    cands = []
    for s in columns:
        for t in targets:
            sc, how = score(s, dataset, t)
            if sc:
                cands.append((sc, str(s), t, how))
    cands.sort(key=lambda x: (-x[0], x[1]))
    used_s, used_t, out = set(), set(), []
    for sc, s, t, how in cands:
        if s in used_s or t in used_t:
            continue
        used_s.add(s), used_t.add(t)
        out.append(dict(source=s, target=t, score=round(sc, 2), method=how))
    return out


def auto_map(columns: list, dataset: str, targets: list[str]) -> tuple[dict, list[dict]]:
    """Returns (mapping to apply automatically, weaker suggestions to confirm). Never remaps a column that already has a target's name,
    and never fills a target some other column already provides exactly."""
    present = {norm(c) for c in columns}
    have = {t for t in targets if norm(t) in present}
    mapping, weak = {}, []
    for m in suggest(columns, dataset, targets):
        if m["target"] in have and m["method"] != "exact":
            continue
        if m["method"] == "exact" and _snake(m["source"]) == m["target"]:
            continue                                    # already identical once spaces/dashes become underscores
        (mapping.__setitem__(m["source"], m["target"]) if m["score"] >= AUTO else weak.append(m))
    return mapping, weak


# ---- datasets for the channel, operations, risk and people modules
ALIASES["marketplace_orders"] = {
    "order_id": _COMMON["order_id"], "date": ["date", "order date", "order placed", "purchase date", "created at"],
    "channel": ["channel", "marketplace", "platform", "sales channel", "portal", "seller platform"],
    "product": _COMMON["product"], "gross_amount": ["gross amount", "gross", "order value", "item price", "selling price", "sale amount", "amount", "order amount", "total"],
    "commission": ["commission", "referral fee", "marketplace fee", "commission fee", "platform fee", "category fee"],
    "shipping_fee": ["shipping fee", "shipping charges", "delivery fee", "logistics fee", "fulfilment fee", "weight handling fee"],
    "other_fees": ["other fees", "other charges", "closing fee", "fixed fee", "collection fee", "tcs", "gst on fees"],
    "settlement_amount": ["settlement amount", "settled amount", "payout", "payment received", "net payout", "amount paid", "settlement value"],
    "settled_date": ["settled date", "settlement date", "payout date", "payment date", "paid on"], "returned": ["returned", "is returned", "return flag", "rto", "return"]}
ALIASES["whatsapp_messages"] = {
    "date": ["date", "sent date", "day", "send date"], "campaign": ["campaign", "campaign name", "template", "template name", "broadcast", "broadcast name"],
    "sent": ["sent", "messages sent", "submitted", "total sent"], "delivered": ["delivered", "messages delivered", "delivered count"],
    "read": ["read", "messages read", "seen", "read count"], "replied": ["replied", "replies", "responses", "reply count"],
    "opted_out": ["opted out", "opt outs", "optouts", "unsubscribed", "unsubscribes", "stop replies", "blocked"]}
ALIASES["store_daily"] = {
    "store": ["store", "store name", "outlet", "outlet name", "branch", "shop", "location", "store code"], "date": ["date", "day", "sales date", "business date"],
    "sales": ["sales", "net sales", "revenue", "sales amount", "turnover", "total sales"], "transactions": ["transactions", "bills", "invoices", "bill count", "no of bills", "orders", "txns"],
    "footfall": ["footfall", "visitors", "foot fall", "walk ins", "traffic", "door count"], "staff_hours": ["staff hours", "labour hours", "labor hours", "man hours", "hours worked"],
    "area_sqft": ["area sqft", "area", "sq ft", "store size", "carpet area", "square feet", "size sqft"]}
ALIASES["distributors"] = {
    "distributor": ["distributor", "distributor name", "dealer", "dealer name", "stockist", "super stockist", "channel partner"], "month": ["month", "period", "date", "billing month"],
    "primary_sales": ["primary sales", "sell in", "sellin", "primary", "billed to distributor", "sales to distributor", "dispatch value"],
    "secondary_sales": ["secondary sales", "sell out", "sellout", "secondary", "sales to retailers", "distributor sales", "retail offtake"],
    "closing_stock": ["closing stock", "stock", "stock value", "inventory", "stock on hand"], "outstanding": ["outstanding", "receivable", "amount due", "balance", "overdue amount"],
    "credit_days": ["credit days", "credit period", "payment terms", "terms days", "credit terms"]}
ALIASES["shipments"] = {
    "shipment_id": ["shipment id", "awb", "awb no", "awb number", "tracking id", "tracking number", "consignment no", "docket no", "shipment no"],
    "carrier": ["carrier", "courier", "courier name", "courier partner", "logistics partner", "transporter", "shipping partner"],
    "ship_date": ["ship date", "dispatch date", "shipped on", "pickup date", "manifest date", "date"], "promised_date": ["promised date", "expected delivery", "edd", "estimated delivery", "due date", "sla date"],
    "delivered_date": ["delivered date", "delivery date", "delivered on", "actual delivery"], "status": ["status", "delivery status", "shipment status", "current status"],
    "cost": ["cost", "freight", "shipping cost", "freight charges", "courier charges", "shipping charge"], "zone": ["zone", "delivery zone", "shipping zone", "region"],
    "cod_amount": ["cod amount", "cod", "cod value", "collectable amount", "amount to collect"], "weight_kg": ["weight kg", "weight", "charged weight"]}
ALIASES["production"] = {
    "line": ["line", "production line", "machine", "work centre", "work center", "plant line", "shift line"], "date": ["date", "production date", "day", "shift date"],
    "planned_units": ["planned units", "plan", "target", "planned quantity", "planned qty", "target units"], "produced_units": ["produced units", "actual", "output", "produced", "actual quantity", "actual units", "good and bad units"],
    "defective_units": ["defective units", "rejects", "rejected", "defects", "scrap", "scrap units", "rework"], "downtime_min": ["downtime min", "downtime", "downtime minutes", "stoppage minutes", "stop time", "idle minutes"],
    "planned_minutes": ["planned minutes", "available minutes", "shift minutes", "scheduled minutes"], "product": _COMMON["product"]}
ALIASES["maintenance"] = {
    "asset": ["asset", "machine", "equipment", "machine name", "asset id", "equipment id", "asset name"], "date": ["date", "event date", "breakdown date", "service date", "reported on"],
    "event_type": ["event type", "type", "maintenance type", "work type", "job type", "category"], "downtime_hours": ["downtime hours", "downtime", "hours down", "repair hours", "downtime hrs", "duration hours"],
    "cost": ["cost", "repair cost", "maintenance cost", "spares cost", "amount"], "cause": ["cause", "failure cause", "root cause", "reason", "failure reason"]}
ALIASES["tenders"] = {
    "tender_id": ["tender id", "tender no", "tender number", "rfp no", "bid no", "reference no", "nit no"], "title": ["title", "tender title", "description", "name", "work name", "subject"],
    "deadline": ["deadline", "due date", "bid due date", "submission date", "closing date", "last date", "bid end date"], "value": ["value", "tender value", "estimated value", "estimated cost", "amount", "contract value"],
    "status": ["status", "tender status", "bid status", "outcome", "result"], "our_bid": ["our bid", "bid amount", "quoted amount", "our quote", "l1 price", "our price"],
    "winning_bid": ["winning bid", "awarded value", "l1 value", "winner price", "award amount", "winning price"], "emd_amount": ["emd amount", "emd", "earnest money", "bid security", "security deposit"],
    "submitted_date": ["submitted date", "submission date", "submitted on"]}
ALIASES["esg"] = {
    "month": ["month", "period", "date", "billing month"], "electricity_kwh": ["electricity kwh", "electricity", "kwh", "units consumed", "power consumption", "energy kwh", "electricity units"],
    "fuel_litres": ["fuel litres", "diesel litres", "fuel", "diesel", "fuel consumption", "diesel consumption", "litres"], "water_kl": ["water kl", "water", "water consumption", "water kilolitres", "kl"],
    "waste_kg": ["waste kg", "waste", "waste generated", "total waste"], "recycled_kg": ["recycled kg", "recycled", "waste recycled", "recycled waste"], "revenue": ["revenue", "sales", "turnover"]}
ALIASES["candidates"] = {
    "candidate_id": ["candidate id", "applicant id", "candidate", "application id", "applicant no", "id"], "role": ["role", "position", "job title", "job", "opening", "requisition", "designation"],
    "applied_date": ["applied date", "application date", "applied on", "date applied", "date"], "stage": ["stage", "status", "pipeline stage", "current stage", "application status"],
    "source": ["source", "channel", "sourced from", "application source", "origin"], "department": ["department", "team", "function"], "offer_date": ["offer date", "offered on"],
    "hired_date": ["hired date", "joining date", "date of joining", "joined on", "start date", "hire date"], "offer_accepted": ["offer accepted", "accepted"],
    "last_stage_date": ["last stage date", "last updated", "last activity", "stage date", "last status change"]}

ALIASES["sales"]["discount_pct"] = ["discount pct", "discount %", "discount percent", "discount percentage", "disc %", "disc pct", "discount"]
_DESC = ["description", "narration", "particulars", "details", "remarks", "transaction details"]
_REF = ["reference", "ref no", "reference number", "cheque no", "utr", "txn id", "transaction id", "chq no", "voucher no"]
ALIASES["budgets"] = {"month": ["month", "period", "budget month"], "category": ["category", "expense head", "head", "account", "cost centre", "budget head"],
                      "budget": ["budget", "budget amount", "planned", "planned amount", "allocated", "budgeted"]}
ALIASES["bank_statement"] = {"date": ["date", "txn date", "transaction date", "value date", "posting date"], "description": _DESC,
                             "amount": ["amount", "net amount", "transaction amount"], "reference": _REF}
ALIASES["ledger_entries"] = {"date": ["date", "voucher date", "entry date", "posting date"], "description": _DESC,
                             "amount": ["amount", "net amount", "value"], "reference": _REF}
ALIASES["projects"] = {"project_id": ["project id", "project code", "job no", "job id", "project no"], "client": ["client", "client name", "customer", "customer name"],
                       "project": ["project", "project name", "job name", "title"], "budget": ["budget", "project value", "contract value", "quoted amount", "order value"],
                       "start_date": ["start date", "start", "kickoff", "commencement date"], "due_date": ["due date", "deadline", "end date planned", "planned end", "delivery date"],
                       "completed_date": ["completed date", "completion date", "closed date", "actual end"], "status": ["status", "project status", "stage"],
                       "billed": ["billed", "invoiced", "billed amount", "amount invoiced"], "cost_to_date": ["cost to date", "actual cost", "cost incurred", "spent"],
                       "hours_budget": ["hours budget", "budgeted hours", "planned hours", "estimated hours"], "hours_spent": ["hours spent", "actual hours", "hours logged", "hours used"],
                       "percent_complete": ["percent complete", "% complete", "progress", "progress %", "completion %"]}
ALIASES["subscriptions"] = {"subscription_id": ["subscription id", "sub id", "subscription no", "plan id"], "customer": ["customer", "customer name", "account", "client", "member"],
                            "plan": ["plan", "plan name", "tier", "membership type", "package"], "start_date": ["start date", "subscribed on", "joined", "start", "signup date"],
                            "end_date": ["end date", "cancelled on", "cancellation date", "ended", "churn date", "expiry date"],
                            "mrr": ["mrr", "monthly amount", "monthly fee", "monthly recurring revenue", "monthly price", "fee"]}
ALIASES["appointments"] = {"appointment_id": ["appointment id", "booking id", "booking no", "appointment no"], "date": ["date", "appointment date", "booking date", "visit date"],
                           "customer": ["customer", "client", "patient", "member", "customer name", "client name"], "service": ["service", "treatment", "procedure", "session type", "appointment type"],
                           "status": ["status", "appointment status", "booking status", "outcome"], "staff": ["staff", "provider", "doctor", "stylist", "therapist", "coach", "practitioner"],
                           "price": ["price", "fee", "amount", "charge", "service price"], "duration_min": ["duration", "duration min", "minutes", "duration minutes"],
                           "start_time": ["start time", "time", "slot", "appointment time"]}
ALIASES["recipes"] = {"dish": ["dish", "recipe", "menu item", "item", "dish name", "recipe name"], "ingredient": ["ingredient", "raw material", "component", "ingredient name"],
                      "qty_per_portion": ["qty per portion", "quantity per portion", "qty", "quantity", "portion qty", "usage per portion", "quantity used"],
                      "unit_cost": ["unit cost", "cost per unit", "rate", "cost", "purchase rate", "price per unit"], "unit": ["unit", "uom", "unit of measure"],
                      "selling_price": ["selling price", "menu price", "price", "sale price"], "portions_sold": ["portions sold", "units sold", "qty sold", "quantity sold", "covers", "sold"]}
ALIASES["assets"] = {"asset_id": ["asset id", "asset code", "asset no", "tag", "tag no", "fa no"], "asset_name": ["asset name", "asset", "name", "description", "item"],
                     "category": ["category", "asset class", "asset category", "class", "block"], "purchase_date": ["purchase date", "date of purchase", "acquired on", "capitalisation date", "date put to use", "put to use date"],
                     "cost": ["cost", "purchase cost", "original cost", "gross block", "purchase value", "value"], "useful_life_years": ["useful life", "useful life years", "life years", "life"],
                     "salvage_value": ["salvage value", "residual value", "scrap value"], "status": ["status", "condition", "asset status"]}
ALIASES["survey_responses"] = {"response_id": ["response id", "id", "submission id", "response no"], "date": ["date", "submitted on", "response date", "timestamp", "submitted at"],
                               "score": ["score", "nps", "nps score", "likelihood to recommend", "recommend score", "rating 0-10"],
                               "segment": ["segment", "city", "location", "customer type", "store", "branch"], "channel": ["channel", "survey channel", "source"],
                               "comment": ["comment", "comments", "feedback", "verbatim", "remarks", "reason"], "csat": ["csat", "satisfaction", "satisfaction score", "rating"]}
