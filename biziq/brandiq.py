"""BrandIQ - social media performance, customer sentiment from reviews, content themes."""
from __future__ import annotations

import re

import pandas as pd

from .core import AMBER, GREEN, RED, ModuleResult, safe_div

POS = {"great", "love", "excellent", "amazing", "lovely", "stylish", "comfortable", "perfectly", "recommend",
       "favourite", "superb", "durable", "fast", "value", "good", "best", "perfect", "happy"}
NEG = {"poor", "late", "unhelpful", "expensive", "wrong", "hassle", "disappointed", "terrible", "damaged",
       "faded", "worst", "bad", "broken", "refund", "returned", "apart", "cheap", "rude", "slow"}
NEGATORS = {"not", "never", "no", "hardly"}
ISSUE_KEYWORDS = {
    "Quality": ["quality", "stitching", "apart", "faded", "material", "broken", "torn", "damaged"],
    "Delivery": ["delivery", "late", "delayed", "shipping", "courier"],
    "Price": ["expensive", "price", "overpriced", "costly"],
    "Size and fit": ["size", "fit", "small", "large", "tight", "loose"],
    "Packaging": ["packaging", "package", "box"],
    "Customer support": ["support", "service", "rude", "unhelpful", "refund", "return"],
}


def sentiment(text: str) -> float:
    toks = re.findall(r"[a-z']+", str(text).lower())
    score = 0
    for i, t in enumerate(toks):
        s = 1 if t in POS else (-1 if t in NEG else 0)
        if s and i > 0 and toks[i - 1] in NEGATORS:
            s = -s
        score += s
    return max(-1.0, min(1.0, score / 2))


def issues_in(text: str) -> list[str]:
    t = str(text).lower()
    return [k for k, words in ISSUE_KEYWORDS.items() if any(w in t for w in words)]


def run(social: pd.DataFrame | None, reviews: pd.DataFrame | None, as_of: pd.Timestamp) -> ModuleResult:
    res = ModuleResult("BrandIQ")

    if social is not None and len(social):
        s = social.copy()
        for c in ("likes", "comments", "shares", "clicks"):
            if c not in s.columns:
                s[c] = 0
            s[c] = s[c].fillna(0)
        s["engagements"] = s["likes"] + s["comments"] + s["shares"]

        def roll(col):
            g = s.groupby(col).agg(posts=("impressions", "size"), impressions=("impressions", "sum"),
                                   engagements=("engagements", "sum"), clicks=("clicks", "sum"))
            g["engagement_rate"] = g["engagements"] / g["impressions"]
            g["ctr"] = g["clicks"] / g["impressions"]
            return g.sort_values("engagement_rate", ascending=False).reset_index()
        by_plat, by_type = roll("platform"), roll("post_type")
        res.tables["by_platform"], res.tables["by_post_type"] = by_plat, by_type
        if "theme" in s.columns:
            by_theme = roll("theme")
            res.tables["by_theme"] = by_theme
        m = s.set_index("date").resample("MS").agg(posts=("impressions", "size"), impressions=("impressions", "sum"),
                                                   engagements=("engagements", "sum"))
        m["engagement_rate"] = m["engagements"] / m["impressions"]
        res.tables["monthly"] = m.reset_index()
        overall = safe_div(s["engagements"].sum(), s["impressions"].sum())
        last90 = s[s["date"] > as_of - pd.Timedelta(days=90)]
        prev90 = s[(s["date"] <= as_of - pd.Timedelta(days=90)) & (s["date"] > as_of - pd.Timedelta(days=180))]
        er_l, er_p = safe_div(last90["engagements"].sum(), last90["impressions"].sum()), \
            safe_div(prev90["engagements"].sum(), prev90["impressions"].sum())
        res.metrics.update(engagement_rate=overall, posts=len(s), impressions=int(s["impressions"].sum()))
        res.facts.update(posts_analysed=len(s), engagement_rate_pct=round(overall * 100, 2),
                         best_platform=by_plat.iloc[0]["platform"],
                         best_platform_engagement_pct=round(float(by_plat.iloc[0]["engagement_rate"]) * 100, 2),
                         weakest_platform=by_plat.iloc[-1]["platform"],
                         best_post_type=by_type.iloc[0]["post_type"],
                         best_post_type_engagement_pct=round(float(by_type.iloc[0]["engagement_rate"]) * 100, 2),
                         weakest_post_type=by_type.iloc[-1]["post_type"])
        if "theme" in s.columns:
            res.facts["best_theme"] = by_theme.iloc[0]["theme"]
            res.facts["weakest_theme"] = by_theme.iloc[-1]["theme"]
        if pd.notna(er_l) and pd.notna(er_p) and er_p > 0:
            chg = (er_l - er_p) / er_p
            res.facts["engagement_change_90d_pct"] = round(chg * 100, 1)
            if chg < -0.15:
                res.add_flag(AMBER, f"Engagement rate fell {abs(chg):.0%} versus the previous 90 days",
                             f"{er_l:.2%} vs {er_p:.2%}.")

    if reviews is not None and len(reviews):
        r = reviews.copy()
        r["sentiment"] = r["review_text"].apply(sentiment)
        r["label"] = pd.cut(r["sentiment"], [-2, -0.01, 0.01, 2], labels=["Negative", "Neutral", "Positive"])
        r["issues"] = r["review_text"].apply(issues_in)
        res.tables["reviews"] = r.sort_values("date", ascending=False)
        dist = r["label"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).reset_index()
        dist.columns = ["sentiment", "reviews"]
        res.tables["sentiment_distribution"] = dist
        promoters, detractors = (r["rating"] >= 4).mean(), (r["rating"] <= 2).mean()
        nps_like = (promoters - detractors) * 100
        avg_rating = float(r["rating"].mean())
        if "product" in r.columns:
            bp = r.groupby("product").agg(reviews=("rating", "size"), avg_rating=("rating", "mean"),
                                          negative_share=("label", lambda x: (x == "Negative").mean())) \
                .sort_values("avg_rating").reset_index()
            res.tables["by_product"] = bp
            worst = bp.iloc[0]
            res.facts.update(lowest_rated_product=worst["product"],
                             lowest_rated_product_rating=round(float(worst["avg_rating"]), 2))
            for _, p in bp[bp["avg_rating"] < 3.5].iterrows():
                res.add_flag(RED if p["avg_rating"] < 3.0 else AMBER,
                             f"{p['product']} averages {p['avg_rating']:.1f} stars",
                             f"{p['negative_share']:.0%} of its reviews are negative.")
        neg = r[r["label"] == "Negative"].explode("issues")
        iss = neg["issues"].dropna().value_counts().reset_index()
        iss.columns = ["issue", "mentions"]
        res.tables["issue_themes"] = iss
        last = r[r["date"] > as_of - pd.Timedelta(days=90)]
        prev = r[(r["date"] <= as_of - pd.Timedelta(days=90)) & (r["date"] > as_of - pd.Timedelta(days=180))]
        res.metrics.update(avg_rating=avg_rating, net_promoter_like=nps_like, reviews=len(r))
        res.facts.update(reviews_analysed=len(r), avg_rating=round(avg_rating, 2),
                         positive_share_pct=round(float((r["label"] == "Positive").mean()) * 100, 1),
                         negative_share_pct=round(float((r["label"] == "Negative").mean()) * 100, 1),
                         net_promoter_like_score=round(nps_like, 0),
                         top_complaint=iss.iloc[0]["issue"] if len(iss) else "none",
                         top_complaint_mentions=int(iss.iloc[0]["mentions"]) if len(iss) else 0)
        if len(last) >= 10 and len(prev) >= 10 and last["rating"].mean() < prev["rating"].mean() - 0.3:
            res.add_flag(AMBER, "Average rating dropped over the last 90 days",
                         f"{last['rating'].mean():.2f} vs {prev['rating'].mean():.2f} before.")
        if (r["label"] == "Negative").mean() > 0.30:
            res.add_flag(RED, f"{(r['label'] == 'Negative').mean():.0%} of reviews are negative",
                         f"Top complaint: {res.facts['top_complaint']}.")

    # Content theme suggestions combine what performs and what customers talk about
    ideas = []
    if "best_theme" in res.facts:
        ideas.append(f"Make more '{res.facts['best_theme']}' content: it has the highest engagement rate.")
    if "best_post_type" in res.facts:
        ideas.append(f"Prefer {res.facts['best_post_type']} posts over {res.facts['weakest_post_type']}.")
    if res.facts.get("top_complaint", "none") != "none":
        ideas.append(f"Address '{res.facts['top_complaint']}' publicly (FAQ, behind-the-scenes, guarantee post).")
    if "lowest_rated_product" in res.facts and res.facts["lowest_rated_product_rating"] < 3.5:
        ideas.append(f"Pause paid promotion of {res.facts['lowest_rated_product']} until its reviews improve.")
    res.tables["content_ideas"] = pd.DataFrame({"suggestion": ideas})
    res.facts["content_suggestions"] = len(ideas)
    if not res.flags:
        res.add_flag(GREEN, "Brand signals look healthy")
    res.notes.append("Sentiment uses a simple word-list method with negation handling; ratings are "
                     "the more reliable signal. Engagement rate = (likes + comments + shares) / impressions.")
    return res
