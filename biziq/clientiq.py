"""ClientIQ - client retention: RFM segmentation, churn model, CLV, next-best-action,
and follow-up emails drafted for human review."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, inr, safe_div

AT_RISK_P = 0.70  # churn probability above which a customer is treated as "at risk"
SEGMENT_ORDER = ["Champions", "Loyal", "Potential", "New", "Needs Attention", "At Risk", "Hibernating"]


def _q5(s: pd.Series, ascending: bool = True) -> pd.Series:
    """Quintile score 1-5 (5 = best). Rank-based so ties never break the bins."""
    r = s.rank(method="first", ascending=ascending)
    try:
        return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    except ValueError:       # too few customers to form five groups: score by rank position instead
        return pd.Series(np.ceil(r / max(len(r), 1) * 5).clip(1, 5).fillna(3).astype(int), index=s.index)


def rfm_table(sales: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    s = sales[sales["date"] < as_of]
    g = s.groupby("customer_id")
    t = pd.DataFrame({
        "last_order": g["date"].max(), "first_order": g["date"].min(),
        "frequency": g["order_id"].nunique(), "monetary": g["revenue"].sum(),
    })
    t["recency_days"] = (as_of - t["last_order"]).dt.days
    t["tenure_days"] = (as_of - t["first_order"]).dt.days
    t["aov"] = t["monetary"] / t["frequency"]
    return t


def _segment(r: pd.Series) -> str:
    if r["frequency"] == 1 and r["tenure_days"] <= 60:
        return "New"
    if r["R"] >= 4 and r["F"] >= 4:
        return "Champions"
    if r["R"] >= 3 and r["F"] >= 4:
        return "Loyal"
    if r["R"] <= 2 and r["F"] >= 3:
        return "At Risk"
    if r["R"] <= 2:
        return "Hibernating"
    if r["R"] >= 3 and r["F"] <= 2:
        return "Potential"
    return "Needs Attention"


def segment_customers(sales: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    t = rfm_table(sales, as_of)
    t["R"] = _q5(t["recency_days"], ascending=False)  # recent = high
    t["F"] = _q5(t["frequency"])
    t["M"] = _q5(t["monetary"])
    t["segment"] = t.apply(_segment, axis=1)
    return t


def _features_at(sales: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    s = sales[sales["date"] < cutoff]
    g = s.groupby("customer_id")
    f = pd.DataFrame({"frequency": g["order_id"].nunique(), "monetary": g["revenue"].sum(),
                      "last": g["date"].max(), "first": g["date"].min()})
    f["recency"] = (cutoff - f["last"]).dt.days
    f["tenure"] = (cutoff - f["first"]).dt.days
    f["avg_gap"] = f["tenure"] / f["frequency"].clip(lower=1)
    recent = s[s["date"] >= cutoff - pd.Timedelta(days=90)].groupby("customer_id")["order_id"].nunique()
    f["orders_90d"] = recent.reindex(f.index).fillna(0)
    f["recency_ratio"] = f["recency"] / f["avg_gap"].clip(lower=1)
    return f[["recency", "frequency", "monetary", "tenure", "avg_gap", "orders_90d", "recency_ratio"]]


def churn_model(sales: pd.DataFrame, as_of: pd.Timestamp, horizon: int = 90):
    """Train on the past (features at as_of-horizon, label = no purchase in the
    following `horizon` days) and score customers as of now. Returns (probs, auc, method)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cutoff = as_of - pd.Timedelta(days=horizon)
    X = _features_at(sales, cutoff)
    later = sales[(sales["date"] >= cutoff) & (sales["date"] < as_of)]["customer_id"].unique()
    y = (~X.index.isin(later)).astype(int)
    now = _features_at(sales, as_of)

    def heuristic():
        p = 1 - np.exp(-0.5 * now["recency_ratio"].clip(0, 6))
        return pd.Series(p.values, index=now.index)

    if len(X) < 60 or y.sum() < 10 or (len(y) - y.sum()) < 10:
        return heuristic(), float("nan"), "heuristic (not enough history to train a model)"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
    model.fit(X, y)
    probs = pd.Series(model.predict_proba(now)[:, 1], index=now.index)
    return probs, auc, "logistic regression"


def retention_rate(sales: pd.DataFrame) -> float:
    mid = sales["date"].min() + (sales["date"].max() - sales["date"].min()) / 2
    first = set(sales[sales["date"] < mid]["customer_id"])
    second = set(sales[sales["date"] >= mid]["customer_id"])
    return safe_div(len(first & second), len(first), 0.6)


NBA = {
    "Champions": "Reward loyalty: early access to new arrivals and a thank-you note; ask for a review.",
    "Loyal": "Upsell: bundle offer on their favourite category plus a complementary product.",
    "Potential": "Nudge a repeat purchase: personalised offer on a complementary product.",
    "New": "Onboard: welcome message and a second-purchase offer within 30 days.",
    "Needs Attention": "Re-engage: reminder with new arrivals in their favourite category.",
    "At Risk": "Win back: personal follow-up and a time-limited offer on their favourite category.",
    "Hibernating": "Low-cost reactivation: single email with a 'we miss you' offer; stop if no response.",
}


def next_best_action(row: pd.Series, high_value_cut: float) -> str:
    if row["churn_prob"] >= AT_RISK_P and row["monetary"] >= high_value_cut:
        return "High-value customer at risk: personal call from the owner plus a 10-15% win-back offer."
    if row["churn_prob"] >= AT_RISK_P:
        return "Win-back email with a 10% offer on their favourite category."
    return NBA.get(row["segment"], NBA["Needs Attention"])


def draft_followup(row: pd.Series, offer_pct: int = 10) -> dict:
    cat = row.get("fav_category", "your favourites")
    body = (
        f"Subject: We'd love to see you again\n\n"
        f"Hi {{first_name}},\n\n"
        f"It has been {int(row['recency_days'])} days since your last order, and we have missed you. "
        f"As a thank-you for being one of our customers, here is {offer_pct}% off your next "
        f"{cat} purchase, valid for 14 days.\n\n"
        f"Warm regards,\n{{sender_name}}\n\n"
        f"[DRAFT - for human review. Replace the placeholders and confirm the offer before sending.]")
    return {"customer_id": row.name, "segment": row["segment"],
            "churn_prob": round(float(row["churn_prob"]), 2),
            "annual_revenue_at_stake": round(float(row["annual_revenue"])), "draft": body}


def run(sales: pd.DataFrame, gross_margin: float = 0.32) -> ModuleResult:
    res = ModuleResult("ClientIQ")
    as_of = sales["date"].max() + pd.Timedelta(days=1)
    seg = segment_customers(sales, as_of)
    probs, auc, method = churn_model(sales, as_of)
    seg["churn_prob"] = probs.reindex(seg.index).fillna(1.0)

    fav = (sales.groupby(["customer_id", "category"])["revenue"].sum().reset_index()
           .sort_values("revenue", ascending=False).drop_duplicates("customer_id")
           .set_index("customer_id")["category"])
    seg["fav_category"] = fav.reindex(seg.index)

    days_span = max((as_of - sales["date"].min()).days, 1)
    seg["orders_per_year"] = seg["frequency"] / days_span * 365
    seg["annual_revenue"] = seg["aov"] * seg["orders_per_year"]
    ret = float(np.clip(retention_rate(sales), 0.2, 0.95))
    lifetime = float(np.clip(1 / (1 - ret), 1, 6))
    seg["clv"] = seg["annual_revenue"] * gross_margin * lifetime
    seg["risk_value"] = seg["churn_prob"] * seg["annual_revenue"]
    high_cut = seg["monetary"].quantile(0.7)
    seg["next_best_action"] = seg.apply(lambda r: next_best_action(r, high_cut), axis=1)

    # Segment summary
    summ = (seg.groupby("segment").agg(customers=("monetary", "size"), revenue=("monetary", "sum"),
                                       avg_orders=("frequency", "mean"), avg_recency_days=("recency_days", "mean"),
                                       avg_churn_prob=("churn_prob", "mean"), total_clv=("clv", "sum"))
            .reindex(SEGMENT_ORDER).dropna(how="all").reset_index())
    summ["share_of_revenue"] = summ["revenue"] / summ["revenue"].sum()
    res.tables["segments"] = summ

    at_risk = seg[seg["churn_prob"] >= AT_RISK_P].sort_values("risk_value", ascending=False)
    cols = ["segment", "recency_days", "frequency", "monetary", "churn_prob", "annual_revenue",
            "clv", "fav_category", "next_best_action"]
    res.tables["at_risk"] = at_risk[cols].reset_index().head(100)
    res.tables["customers"] = seg.reset_index()
    drafts = [draft_followup(r) for _, r in at_risk.head(15).iterrows()]
    res.tables["drafted_emails"] = pd.DataFrame(drafts)

    total_rev = float(seg["monetary"].sum())
    top10 = seg.sort_values("monetary", ascending=False).head(max(1, len(seg) // 10))
    value_at_stake = float(at_risk["annual_revenue"].sum())
    expected_loss = float(at_risk["risk_value"].sum())
    res.metrics = dict(customers=len(seg), at_risk_customers=len(at_risk),
                       annual_revenue_at_stake=value_at_stake, expected_revenue_loss=expected_loss,
                       retention_rate=ret, avg_clv=float(seg["clv"].mean()),
                       model=method, model_auc=auc)
    res.facts = dict(
        customers=len(seg), at_risk_customers=len(at_risk),
        annual_revenue_at_stake=inr(value_at_stake), expected_revenue_loss=inr(expected_loss),
        retention_rate_pct=round(ret * 100, 1), avg_clv=inr(float(seg["clv"].mean())),
        top10_pct_revenue_share_pct=round(top10["monetary"].sum() / total_rev * 100, 1),
        champions=int((seg["segment"] == "Champions").sum()),
        hibernating=int((seg["segment"] == "Hibernating").sum()),
        at_risk_segment=int((seg["segment"] == "At Risk").sum()),
        churn_model_auc=None if pd.isna(auc) else round(auc, 2),
    )
    share = len(at_risk) / len(seg)
    if share > 0.30:
        res.add_flag(RED, f"{len(at_risk)} customers ({share:.0%}) are at high risk of churn",
                     f"About {inr(value_at_stake)} of annual revenue is attached to them.")
    elif share > 0.15:
        res.add_flag(AMBER, f"{len(at_risk)} customers ({share:.0%}) are at high risk of churn",
                     f"About {inr(value_at_stake)} of annual revenue is attached to them.")
    conc = res.facts["top10_pct_revenue_share_pct"]
    if conc > 50:
        res.add_flag(AMBER, f"Top 10% of customers generate {conc}% of revenue",
                     "Revenue is concentrated in a small group of customers.")
    res.notes.append(
        f"Churn model: {method}" + ("" if pd.isna(auc) else f", test AUC {auc:.2f}") +
        f". 'At risk' means a churn probability of {AT_RISK_P:.0%} or more of no purchase in the next 90 days. "
        f"CLV = annual revenue x {gross_margin:.0%} margin x {lifetime:.1f} expected years "
        f"(retention {ret:.0%}); a simple estimate, not a forecast.")
    return res
