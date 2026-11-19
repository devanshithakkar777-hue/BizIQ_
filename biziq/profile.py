"""Business profile and module applicability.

BizIQ is one central platform; each business tells it what it does (a business type plus capability switches) and every module is
then either APPLICABLE or NOT APPLICABLE for that business. A retail shop never sees a recipe-costing page full of nothing, it sees
"not applicable for your business model", with the reason and a button to enable it anyway. The owner can change the profile later.

Rules, kept deliberately simple and visible:
  * a module with no capability requirement is universal;
  * otherwise it applies when the business has ANY of its capabilities switched on;
  * an owner/admin can force-enable a module that does not apply ("enable anyway"), which is saved as an override.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# capability switches: key -> (label, help)
CAPABILITIES: dict[str, tuple[str, str]] = {
    "goods": ("Sells physical products", "Anything you stock, make or buy and resell."),
    "inventory": ("Holds stock", "You keep inventory in a shop, warehouse or kitchen."),
    "online": ("Sells online or on marketplaces", "Own website, Amazon, Flipkart, Meesho, Instagram shop..."),
    "stores": ("Runs physical stores or outlets", "One or more shops, counters or branches."),
    "shipping": ("Ships orders to customers", "Courier or own delivery."),
    "channel": ("Sells through distributors or dealers", "You bill distributors who sell on to retailers."),
    "manufacturing": ("Makes things on production lines", "Factory, workshop or processing unit."),
    "food": ("Prepares food or drink", "Restaurant, cafe, cloud kitchen, bakery, food brand."),
    "b2b": ("Sells to other businesses on credit", "Invoices with payment terms, quotes, contracts."),
    "b2c": ("Sells to individual consumers", "Walk-in or online customers."),
    "returns": ("Handles returns and refunds", "Customers send items back."),
    "ads": ("Spends on advertising", "Meta, Google, influencers or similar."),
    "whatsapp": ("Markets or sells over WhatsApp", "Broadcasts, catalogues or order chats."),
    "projects": ("Delivers client projects billed by time or scope", "Agency, consultancy, freelancer, contractor."),
    "subscriptions": ("Has recurring subscriptions or memberships", "Monthly or annual plans."),
    "appointments": ("Runs on appointments or bookings", "Clinic, salon, coaching, studio."),
    "assets": ("Owns machinery, vehicles or equipment", "Items you depreciate."),
    "tenders": ("Bids for tenders or government supply", "Formal bids with deadlines and deposits."),
    "employees": ("Has employees", "Payroll, hiring, reviews."),
}

BUSINESS_TYPES: dict[str, dict] = {
    "retail": {"label": "Retail shop", "caps": ["goods", "inventory", "stores", "b2c", "returns", "employees", "assets", "whatsapp"]},
    "ecommerce": {"label": "E-commerce / D2C brand", "caps": ["goods", "inventory", "online", "shipping", "b2c", "returns", "ads", "whatsapp", "employees"]},
    "food": {"label": "Restaurant, cafe or food business", "caps": ["food", "inventory", "stores", "b2c", "employees", "assets", "whatsapp"]},
    "manufacturer": {"label": "Manufacturer", "caps": ["goods", "inventory", "manufacturing", "b2b", "shipping", "assets", "employees", "channel", "tenders"]},
    "distributor": {"label": "Distributor or wholesaler", "caps": ["goods", "inventory", "b2b", "shipping", "channel", "employees", "assets"]},
    "services": {"label": "Agency, consultancy or freelancer", "caps": ["projects", "b2b", "employees", "ads"]},
    "saas": {"label": "Software / subscription business", "caps": ["subscriptions", "b2b", "b2c", "ads", "employees"]},
    "appointments": {"label": "Clinic, salon, coaching or studio", "caps": ["appointments", "b2c", "employees", "assets", "whatsapp", "subscriptions"]},
    "contractor": {"label": "Contractor or government supplier", "caps": ["projects", "tenders", "b2b", "assets", "employees", "goods", "inventory"]},
    "other": {"label": "Other / a mix (show me everything)", "caps": list(CAPABILITIES)},
}

# module -> capabilities of which ANY switches it on. Empty = universal. Core (original) modules included.
MODULE_CAPS: dict[str, tuple[str, ...]] = {
    # core
    "WealthIQ": (), "SalesIQ": (), "ClientIQ": (), "CashFlowIQ": (), "TaxIQ": (), "RiskIQ": (), "InvestIQ": (), "ReportIQ": (), "BrandIQ": (),
    "SupplyChainIQ": ("inventory", "manufacturing", "food"), "ProcureIQ": ("goods", "manufacturing", "food"), "PeopleIQ": ("employees",),
    # finance and compliance
    "ProfitIQ": ("goods", "food", "manufacturing"), "ReceivablesIQ": ("b2b", "projects", "tenders", "subscriptions"), "CreditIQ": ("b2b", "channel"),
    "ComplianceIQ": (), "ExpenseIQ": (), "FraudIQ": (),
    # commercial
    "PricingIQ": ("goods", "food", "subscriptions"), "ForecastIQ": (), "ReturnsIQ": ("returns",), "CampaignIQ": ("ads",),
    "LoyaltyIQ": ("b2c",), "PipelineIQ": ("b2b",), "QuoteIQ": ("b2b", "projects", "tenders"), "ContractIQ": ("b2b", "tenders", "projects", "subscriptions"),
    "SupportIQ": (),
    # customers and channels
    "ChurnIQ": (), "PersonalisationIQ": ("goods", "food", "online"), "WhatsAppIQ": ("whatsapp",), "MarketplaceIQ": ("online",), "StoreIQ": ("stores",),
    "DistributorIQ": ("channel",),
    # operations and industry
    "LogisticsIQ": ("shipping",), "ProductionIQ": ("manufacturing", "food"), "MaintenanceIQ": ("manufacturing", "assets"), "TenderIQ": ("tenders",), "ESGIQ": (),
    "VendorRiskIQ": ("goods", "manufacturing", "food", "inventory"),
    # people
    "HiringIQ": ("employees",), "SalaryIQ": ("employees",), "PerformanceReviewIQ": ("employees",),
    # round three
    "BudgetIQ": (), "PayablesIQ": (), "ReconIQ": (), "UnitEconomicsIQ": ("ads", "subscriptions", "online", "b2b", "appointments"), "ScenarioIQ": (),
    "InventoryIQ": ("inventory",), "ComplianceCalendarIQ": (), "ProjectIQ": ("projects",), "SubscriptionIQ": ("subscriptions",),
    "AppointmentIQ": ("appointments",), "RecipeIQ": ("food",), "AssetIQ": ("assets",), "DiscountIQ": ("goods", "food", "subscriptions"),
    "AnomalyIQ": (), "SurveyIQ": (),
}

REASON = {
    "RecipeIQ": "RecipeIQ costs dishes from recipes and ingredient prices, so it is for restaurants, cafes and food brands.",
    "AppointmentIQ": "AppointmentIQ measures bookings, no-shows and slot use, so it is for clinics, salons, coaching and similar.",
    "ProjectIQ": "ProjectIQ tracks hours against quotes on client projects, so it is for agencies, consultancies and contractors.",
    "SubscriptionIQ": "SubscriptionIQ measures recurring revenue and churn, so it is for subscription and membership businesses.",
    "InventoryIQ": "InventoryIQ analyses stock, so it is for businesses that hold inventory.",
}


@dataclass
class Profile:
    business_type: str = "other"
    caps: frozenset = frozenset(CAPABILITIES)
    overrides: frozenset = frozenset()          # modules force-enabled by an owner
    accent: str = "#22d3ee"
    tagline: str = ""
    confirmed: bool = False                      # owner has reviewed the profile

    def to_json(self) -> str:
        return json.dumps(dict(business_type=self.business_type, caps=sorted(self.caps), overrides=sorted(self.overrides),
                               accent=self.accent, tagline=self.tagline, confirmed=self.confirmed))

    @classmethod
    def from_json(cls, raw: str | None) -> "Profile":
        if not raw:
            return cls()
        try:
            d = json.loads(raw)
        except (TypeError, ValueError):
            return cls()
        bt = d.get("business_type") if d.get("business_type") in BUSINESS_TYPES else "other"
        caps = frozenset(c for c in d.get("caps", []) if c in CAPABILITIES)
        return cls(bt, caps, frozenset(d.get("overrides", [])), clean_accent(d.get("accent")), str(d.get("tagline", ""))[:120], bool(d.get("confirmed")))


def defaults_for(business_type: str) -> frozenset:
    return frozenset(BUSINESS_TYPES.get(business_type, BUSINESS_TYPES["other"])["caps"])


def clean_accent(value) -> str:
    """Accent colours go into CSS, so only a plain #rrggbb is accepted."""
    v = str(value or "").strip()
    return v if re.fullmatch(r"#[0-9a-fA-F]{6}", v) else "#22d3ee"


def applies(module: str, profile: Profile) -> bool:
    if module in profile.overrides:
        return True
    need = MODULE_CAPS.get(module)
    if need is None:                              # unknown module: never hide something we have not classified
        return True
    return not need or bool(set(need) & set(profile.caps))


def why_not(module: str, profile: Profile) -> str:
    need = MODULE_CAPS.get(module, ())
    labels = [CAPABILITIES[c][0].lower() for c in need]
    types = [v["label"] for v in BUSINESS_TYPES.values() if set(need) & set(v["caps"]) and v["label"] != BUSINESS_TYPES["other"]["label"]]
    base = REASON.get(module, f"{module} is for businesses that {' or '.join(labels)}.")
    out = f"Not applicable for your business model. {base}"
    if types:
        out += " Typical fits: " + ", ".join(types[:5]) + "."
    return out


def applicable_modules(profile: Profile, names) -> list[str]:
    return [n for n in names if applies(n, profile)]
