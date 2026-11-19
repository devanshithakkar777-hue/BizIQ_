"""BizIQ design system: dark glass theme, Plotly template, and the small HTML components the pages share.

No external requests (system font stack, inline SVG). Every piece of data-derived text is HTML-escaped before it
goes into markup, because customer and product names come from uploaded files.
"""
from __future__ import annotations

import html
import math

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ACCENT = "#22d3ee"          # cyan
ACCENT2 = "#a78bfa"         # violet
GOOD, WARN, BAD = "#34d399", "#fbbf24", "#fb7185"
BG, PANEL, LINE, TEXT, MUTED = "#070b14", "rgba(255,255,255,0.045)", "rgba(148,163,184,0.18)", "#e6edf7", "#8b9bb4"
COLORWAY = [ACCENT, ACCENT2, "#f472b6", "#fbbf24", "#34d399", "#60a5fa", "#fb923c", "#94a3b8"]
LEVEL_COLOR = {"red": BAD, "amber": WARN, "green": GOOD}
LEVEL_LABEL = {"red": "Act now", "amber": "Watch", "green": "Healthy"}

CSS = f"""
<style>
:root {{ --acc:{ACCENT}; --acc2:{ACCENT2}; --good:{GOOD}; --warn:{WARN}; --bad:{BAD}; --muted:{MUTED}; --line:{LINE}; --panel:{PANEL}; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: radial-gradient(1200px 600px at 8% -10%, rgba(34,211,238,.10), transparent 60%),
              radial-gradient(1000px 520px at 100% 0%, rgba(167,139,250,.11), transparent 55%), {BG} !important;
  font-family: Inter, "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
h1 {{ background: linear-gradient(90deg,#fff,var(--acc)); -webkit-background-clip:text; background-clip:text; color:transparent !important; font-weight:750; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, rgba(15,23,42,.92), rgba(7,11,20,.96)) !important; border-right:1px solid var(--line); }}
[data-testid="stSidebar"] .stRadio label {{ border-radius:10px; padding:.25rem .5rem; transition:background .15s, transform .15s; }}
[data-testid="stSidebar"] .stRadio label:hover {{ background:rgba(34,211,238,.10); transform:translateX(2px); }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--panel); border:1px solid var(--line) !important; border-radius:16px;
  backdrop-filter: blur(10px); box-shadow: 0 8px 30px rgba(0,0,0,.28); }}
.stTabs [data-baseweb="tab-list"] {{ gap:.25rem; border-bottom:1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ border-radius:10px 10px 0 0; padding:.45rem .9rem; }}
.stTabs [aria-selected="true"] {{ background:rgba(34,211,238,.10); color:var(--acc) !important; }}
.stButton>button, .stDownloadButton>button {{ border-radius:12px; border:1px solid var(--line); background:rgba(255,255,255,.05); transition:all .18s; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color:var(--acc); box-shadow:0 0 0 1px var(--acc), 0 6px 22px rgba(34,211,238,.25); transform:translateY(-1px); }}
.stButton>button[kind="primary"] {{ background:linear-gradient(135deg,var(--acc),{ACCENT2}); color:#04121c; border:0; font-weight:650; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
.biziq-hero {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin:0 0 1.1rem 0; padding-bottom:.9rem; border-bottom:1px solid var(--line); }}
.biziq-hero .eyebrow {{ color:var(--acc); font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; font-weight:650; }}
.biziq-hero h1 {{ margin:.1rem 0 .15rem 0; font-size:2.1rem; line-height:1.1; padding:0; }}
.biziq-hero p {{ color:var(--muted); margin:0; max-width:62ch; }}
.pill {{ display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .7rem; border-radius:999px; font-size:.78rem; font-weight:650;
  border:1px solid currentColor; background:color-mix(in srgb, currentColor 12%, transparent); white-space:nowrap; }}
.pill i {{ width:.5rem; height:.5rem; border-radius:50%; background:currentColor; box-shadow:0 0 10px currentColor; display:inline-block; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.8rem; margin:.4rem 0 1.1rem; }}
.kpi {{ position:relative; overflow:hidden; padding:.95rem 1.05rem; border-radius:16px; background:var(--panel); border:1px solid var(--line);
  backdrop-filter:blur(10px); transition:transform .18s, border-color .18s, box-shadow .18s; }}
.kpi:hover {{ transform:translateY(-3px); border-color:var(--acc); box-shadow:0 10px 30px rgba(34,211,238,.18); }}
.kpi::before {{ content:""; position:absolute; inset:0 0 auto 0; height:2px; background:linear-gradient(90deg,var(--acc),var(--acc2)); opacity:.8; }}
.kpi .l {{ color:var(--muted); font-size:.74rem; letter-spacing:.06em; text-transform:uppercase; }}
.kpi .v {{ font-size:1.6rem; font-weight:720; margin:.2rem 0 .1rem; letter-spacing:-.02em; word-break:break-word; }}
.kpi .d {{ font-size:.8rem; color:var(--muted); }}
.kpi svg {{ position:absolute; right:.6rem; bottom:.5rem; opacity:.9; }}
.feed {{ display:flex; flex-direction:column; gap:.5rem; }}
.feed .row {{ display:flex; gap:.75rem; align-items:flex-start; padding:.65rem .8rem; border-radius:12px; background:rgba(255,255,255,.03); border:1px solid var(--line);
  border-left:3px solid var(--c); transition:background .15s; }}
.feed .row:hover {{ background:rgba(255,255,255,.07); }}
.feed .m {{ color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; }}
.feed .t {{ font-weight:600; }}
.feed .x {{ color:var(--muted); font-size:.85rem; }}
.hint {{ color:var(--muted); font-size:.82rem; }}
</style>
"""


def apply() -> None:
    """Inject the stylesheet and make the BizIQ chart template the default. Call once per run, right after set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)
    pio.templates.default = "biziq"


def brand(accent: str) -> None:
    """Colour the page with a business's own accent (validated #rrggbb: it goes into CSS)."""
    import re
    if re.fullmatch(r"#[0-9a-fA-F]{6}", str(accent or "")):
        st.markdown(f"<style>:root{{--acc:{accent};}}</style>", unsafe_allow_html=True)


def _template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=COLORWAY,
        font=dict(family='Inter, "Segoe UI", system-ui, sans-serif', color=TEXT, size=13),
        title=dict(font=dict(size=15, color=TEXT), x=0.01),
        xaxis=dict(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.2)", linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.2)", linecolor=LINE, tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="#0f172a", bordercolor=ACCENT, font=dict(color=TEXT)),
        hovermode="x unified", margin=dict(l=10, r=10, t=44, b=10),
        colorscale=dict(sequential=[[0, "#0b1d33"], [0.5, "#0e7490"], [1, ACCENT]]),
    )
    return t


pio.templates["biziq"] = _template()
pio.templates.default = "biziq"


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def hero(title: str, blurb: str = "", eyebrow: str = "BizIQ", level: str | None = None) -> None:
    right = pill(level) if level else ""
    st.markdown(f'<div class="biziq-hero"><div><div class="eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1><p>{esc(blurb)}</p></div><div>{right}</div></div>',
                unsafe_allow_html=True)


def pill(level: str, text: str | None = None) -> str:
    c = LEVEL_COLOR.get(level, MUTED)
    return f'<span class="pill" style="color:{c}"><i></i>{esc(text or LEVEL_LABEL.get(level, level))}</span>'


def spark(values, color: str = ACCENT, w: int = 92, h: int = 30) -> str:
    v = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = " ".join(f"{i * w / (len(v) - 1):.1f},{h - 3 - (x - lo) / span * (h - 6):.1f}" for i, x in enumerate(v))
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polyline fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="{pts}"/></svg>'


def kpis(items: list[tuple]) -> None:
    """items: (label, value[, note[, sparkline values]]). Renders a responsive grid of KPI cards."""
    cards = []
    for it in items:
        label, value = it[0], it[1]
        note = it[2] if len(it) > 2 and it[2] else ""
        sp = spark(it[3]) if len(it) > 3 and it[3] is not None else ""
        v = f"{value:,.1f}" if isinstance(value, float) else value
        cards.append(f'<div class="kpi"><div class="l">{esc(str(label).replace("_", " "))}</div><div class="v">{esc(v)}</div>'
                     f'<div class="d">{esc(note)}</div>{sp}</div>')
    st.markdown(f'<div class="kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def feed(flags, limit: int = 12, show_module: bool = True) -> None:
    rows = []
    for f in list(flags)[:limit]:
        c = LEVEL_COLOR.get(f.level, MUTED)
        mod = f'<div class="m">{esc(f.module)} · {esc(LEVEL_LABEL.get(f.level, f.level))}</div>' if show_module else ""
        det = f'<div class="x">{esc(f.detail)}</div>' if f.detail else ""
        rows.append(f'<div class="row" style="--c:{c}"><div>{mod}<div class="t">{esc(f.title)}</div>{det}</div></div>')
    if rows:
        st.markdown(f'<div class="feed">{"".join(rows)}</div>', unsafe_allow_html=True)


def health_score(pr) -> int:
    """0-100: average over the modules that ran of 100 (all clear), 65 (worst finding is amber) or 25 (a red finding). Simple enough to explain in one line."""
    pts = {"green": 100, "amber": 65, "red": 25}
    vals = [pts[m.worst_level()] for m in pr.modules.values()]
    return round(sum(vals) / len(vals)) if vals else 0


def gauge(score: int, title: str = "Business health") -> go.Figure:
    color = GOOD if score >= 75 else WARN if score >= 50 else BAD
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, title=dict(text=title, font=dict(size=14, color=MUTED)),
                                 number=dict(font=dict(size=46, color=TEXT)),
                                 gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED), bar=dict(color=color, thickness=0.32), bgcolor="rgba(255,255,255,.04)",
                                            borderwidth=0, steps=[dict(range=[0, 50], color="rgba(251,113,133,.14)"), dict(range=[50, 75], color="rgba(251,191,36,.12)"),
                                                                  dict(range=[75, 100], color="rgba(52,211,153,.12)")])))
    fig.update_layout(height=250, margin=dict(l=30, r=40, t=50, b=10))
    return fig


def _rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def module_map(pr) -> go.Figure:
    """Interactive treemap: one tile per module, sized by number of findings, coloured by its worst flag."""
    names, parents, vals, cols, text = [], [], [], [], []
    for n, m in pr.modules.items():
        names.append(n)
        parents.append("")
        vals.append(max(1, min(len(m.flags), 3)))
        cols.append(_rgba(LEVEL_COLOR[m.worst_level()], 0.8))
        text.append(f"{len(m.flags)} finding(s)<br>{m.flags[0].title if m.flags else ''}"[:140])
    fig = go.Figure(go.Treemap(labels=names, parents=parents, values=vals, marker=dict(colors=cols, line=dict(width=3, color=BG)),
                               text=text, textinfo="label", hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>", textfont=dict(size=14, color="#05101a"),
                               tiling=dict(pad=3)))
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def style(fig: go.Figure, h: int = 340) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=44, b=10), legend=dict(orientation="h", y=-0.18))
    return fig
