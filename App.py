"""
app.py — Multi-brand sales dashboard (Sadafco · Energizer · Friesland)

Performance model
-----------------
The rule here is: no rerun ever re-reads Excel, and no rerun ever filters a
DataFrame in Python.

  1. Excel -> Parquet happens once, offline, in etl.py.
  2. Parquet is registered as a DuckDB view. DuckDB reads columnar data with
     predicate pushdown, so a filtered aggregate touches only the row groups
     and columns it needs.
  3. Every aggregate is wrapped in @st.cache_data keyed on the filter tuple.
     Flipping a filter back to a prior value is a dict lookup, not a scan.
  4. Charts get explicit `uirevision` so Plotly patches rather than rebuilds.

Run:  streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Sales Intelligence",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
#
# Direction: "instrument panel". Deep slate ground, one cool cyan primary for
# the current period and a warm amber for the comparison period -- the two
# colours you are always reading against each other in a sales review. Each
# brand gets its own accent so you always know which book you're in without
# reading the header. Type is IBM Plex Sans (UI) + JetBrains Mono (figures),
# because numbers that must be scanned column-wise need tabular figures.
# ---------------------------------------------------------------------------
INK = "#0B1220"
PANEL = "#111A2B"
LINE = "#1F2B41"
TEXT = "#E6EDF7"
MUTED = "#8A9BB5"
CUR = "#22D3EE"   # current period
PRV = "#F59E0B"   # prior period
POS = "#34D399"
NEG = "#FB7185"

BRANDS = {
    "Sadafco":   {"accent": "#38BDF8", "file": "sadafco.parquet",   "sub": "E-commerce · KSA depots"},
    "Energizer": {"accent": "#A3E635", "file": "energizer.parquet", "sub": "Amazon · Careem · Talabat"},
    "Friesland": {"accent": "#F472B6", "file": "friesland.parquet", "sub": "MSS · 9 markets"},
}

SEQ = ["#22D3EE", "#38BDF8", "#818CF8", "#A78BFA", "#F472B6", "#FB7185",
       "#F59E0B", "#A3E635", "#34D399", "#2DD4BF"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

.stApp {{ background: {INK}; }}
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; color: {TEXT}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1600px; }}

section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}

.masthead {{
  display:flex; align-items:baseline; gap:.9rem;
  border-bottom:1px solid {LINE}; padding-bottom:.7rem; margin-bottom:1.1rem;
}}
.masthead .rule {{ width:3px; height:34px; border-radius:2px; }}
.masthead h1 {{ font-size:1.5rem; font-weight:700; letter-spacing:-.02em; margin:0; }}
.masthead .sub {{ color:{MUTED}; font-size:.8rem; margin-left:auto;
  font-family:'JetBrains Mono',monospace; }}

/* KPI card */
.kpi {{
  background:{PANEL}; border:1px solid {LINE}; border-radius:10px;
  padding:.85rem .95rem; height:100%;
}}
.kpi .lab {{ font-size:.66rem; letter-spacing:.09em; text-transform:uppercase; color:{MUTED}; }}
.kpi .val {{ font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700;
  font-variant-numeric:tabular-nums; margin:.18rem 0 .1rem; letter-spacing:-.02em; }}
.kpi .dlt {{ font-family:'JetBrains Mono',monospace; font-size:.74rem; font-weight:700; }}
.kpi .foot {{ color:{MUTED}; font-size:.66rem; margin-top:.15rem; }}

/* Segmented control. The wrapper testid is stButtonGroup (NOT
   stSegmentedControl) and the buttons expose aria-checked rather than a
   kind attribute -- verified against the live DOM, because guessing here
   silently falls back to BaseWeb's light default. */
div[data-testid="stButtonGroup"] button {{
  background:{PANEL} !important;
  color:{MUTED} !important;
  border:1px solid {LINE} !important;
  border-radius:7px !important;
  padding:.34rem .9rem !important;
  font-size:.8rem !important;
  font-weight:600 !important;
  transition:color .12s ease, border-color .12s ease;
}}
div[data-testid="stButtonGroup"] button:hover {{
  color:{TEXT} !important; border-color:{MUTED} !important;
}}
div[data-testid="stButtonGroup"] button[aria-checked="true"] {{
  background:{INK} !important;
  color:{TEXT} !important;
  border-color:{CUR} !important;
  box-shadow:inset 0 -2px 0 {CUR};
}}
div[data-testid="stButtonGroup"] button p {{
  color:inherit !important; font-weight:600 !important; margin:0 !important;
}}
div[data-testid="stButtonGroup"] button:focus-visible {{
  outline:2px solid {CUR} !important; outline-offset:2px;
}}

div[data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace; }}
.stDataFrame {{ border:1px solid {LINE}; border-radius:8px; }}
.perf {{ font-family:'JetBrains Mono',monospace; font-size:.66rem; color:{MUTED}; text-align:right; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
@st.cache_resource
def con() -> duckdb.DuckDBPyConnection:
    """One DuckDB connection for the server's lifetime, with a view per brand."""
    c = duckdb.connect(":memory:")
    c.execute("PRAGMA threads=4")
    for name, meta in BRANDS.items():
        p = DATA / meta["file"]
        if p.exists():
            c.execute(f"CREATE OR REPLACE VIEW {name.lower()} AS "
                      f"SELECT * FROM read_parquet('{p.as_posix()}')")
    return c


@st.cache_data(show_spinner=False)
def available() -> list[str]:
    return [b for b, m in BRANDS.items() if (DATA / m["file"]).exists()]


@st.cache_data(show_spinner=False)
def schema(brand: str) -> list[str]:
    return con().execute(f"SELECT * FROM {brand.lower()} LIMIT 0").df().columns.tolist()


@st.cache_data(show_spinner=False)
def distinct(brand: str, col: str) -> list[str]:
    q = f'SELECT DISTINCT "{col}" v FROM {brand.lower()} WHERE "{col}" IS NOT NULL ORDER BY 1'
    return con().execute(q).df()["v"].astype(str).tolist()


@st.cache_data(show_spinner=False)
def months(brand: str) -> list[str]:
    q = f"SELECT DISTINCT MonthKey m FROM {brand.lower()} ORDER BY 1"
    return con().execute(q).df()["m"].astype(str).tolist()


def _where(filters: tuple[tuple[str, tuple[str, ...]], ...], m0: str, m1: str) -> str:
    """Build a SQL predicate. Filters arrive as a hashable tuple so the
    caller's @st.cache_data key stays stable across reruns."""
    parts = [f"MonthKey BETWEEN '{m0}' AND '{m1}'"]
    for col, vals in filters:
        if vals:
            lst = ",".join("'" + v.replace("'", "''") + "'" for v in vals)
            parts.append(f'"{col}" IN ({lst})')
    return " AND ".join(parts)


@st.cache_data(show_spinner=False)
def q(brand: str, select: str, filters, m0: str, m1: str,
      group: str = "", order: str = "", limit: int = 0) -> pd.DataFrame:
    """Generic cached aggregate. Every panel funnels through here so that
    identical requests across tabs share one cache entry."""
    sql = f"SELECT {select} FROM {brand.lower()} WHERE {_where(filters, m0, m1)}"
    if group:
        sql += f" GROUP BY {group}"
    if order:
        sql += f" ORDER BY {order}"
    if limit:
        sql += f" LIMIT {limit}"
    return con().execute(sql).df()


# ---------------------------------------------------------------------------
# Formatting + chart helpers
# ---------------------------------------------------------------------------
def money(v: float) -> str:
    if v is None or pd.isna(v):
        return "—"
    a = abs(v)
    s = "-" if v < 0 else ""
    if a >= 1e9:  return f"{s}{a/1e9:.2f}B"
    if a >= 1e6:  return f"{s}{a/1e6:.2f}M"
    if a >= 1e3:  return f"{s}{a/1e3:.1f}K"
    return f"{s}{a:,.0f}"


def num(v: float) -> str:
    if v is None or pd.isna(v):
        return "—"
    a = abs(v)
    s = "-" if v < 0 else ""
    if a >= 1e6: return f"{s}{a/1e6:.2f}M"
    if a >= 1e3: return f"{s}{a/1e3:.1f}K"
    return f"{s}{a:,.0f}"


def kpi(label: str, value: str, delta: float | None = None,
        foot: str = "", accent: str = CUR) -> str:
    d = ""
    if delta is not None and pd.notna(delta):
        c = POS if delta >= 0 else NEG
        d = f'<div class="dlt" style="color:{c}">{"▲" if delta>=0 else "▼"} {abs(delta)*100:,.1f}%</div>'
    return (f'<div class="kpi" style="border-left:3px solid {accent}">'
            f'<div class="lab">{label}</div>'
            f'<div class="val" style="color:{accent}">{value}</div>{d}'
            f'<div class="foot">{foot}</div></div>')


def style(fig: go.Figure, h: int = 320, legend: bool = True, rev: str = "x") -> go.Figure:
    fig.update_layout(
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", size=11, color=MUTED),
        margin=dict(l=8, r=8, t=28, b=8),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)"),
        uirevision=rev,  # keeps zoom/pan across reruns; avoids full redraw
    )
    # Only touch the title font if a title was actually set. Passing
    # title=dict(font=...) with no `text` makes Plotly render the string
    # "undefined" above the chart.
    if fig.layout.title.text:
        fig.update_layout(title=dict(font=dict(size=12, color=TEXT), x=0, xanchor="left"))
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    return fig


def growth(cur: float, prv: float) -> float | None:
    if prv in (0, None) or pd.isna(prv) or prv == 0:
        return None
    return (cur - prv) / abs(prv)


def make_pareto(d: pd.DataFrame, accent: str,
                title: str = "Pareto — value concentration") -> go.Figure:
    """Bars = value per member (descending), line = cumulative % of total.

    The 80% guide line is the point of the chart: where it crosses the curve
    tells you how few members you actually depend on.
    """
    d = d.sort_values("v", ascending=False).head(40).copy()
    d["cum%"] = d["v"].cumsum() / d["v"].sum() * 100
    lab = d["k"].astype(str).str.slice(0, 22)
    f = go.Figure()
    f.add_bar(x=lab, y=d["v"], marker_color=accent, opacity=.55, name="value",
              hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>")
    f.add_scatter(x=lab, y=d["cum%"], yaxis="y2", mode="lines+markers",
                  line=dict(color=PRV, width=2), marker=dict(size=4),
                  name="cumulative %",
                  hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>")
    f.add_hline(y=80, yref="y2", line=dict(color=MUTED, dash="dot", width=1))
    f.update_layout(title=title,
                    yaxis2=dict(overlaying="y", side="right", range=[0, 105],
                                showgrid=False, ticksuffix="%", color=PRV))
    f.update_xaxes(tickangle=-45, tickfont=dict(size=8))
    return f


# ---------------------------------------------------------------------------
# Per-brand metric configuration
#   value  = revenue column
#   volume = unit column
#   dims   = filterable / groupable dimensions in priority order
# ---------------------------------------------------------------------------
CFG = {
    "Sadafco": dict(
        value="Sales", volume="Qty", vlabel="Gross Sales (SAR)", qlabel="Units",
        dims=["Depot", "Channel", "Type", "Category", "SubGroup", "Customer", "SKU"],
        entity="Customer", product="SubGroupDesc",
    ),
    "Energizer": dict(
        value="Sales", volume="Units", vlabel="Revenue (AED)", qlabel="Units",
        dims=["Retailer", "Brand", "Type", "Title"],
        entity="Retailer", product="Title",
    ),
    "Friesland": dict(
        value="Sales", volume="Cases", vlabel="Sales", qlabel="Cases",
        dims=["Country", "Brand", "Category", "Customer", "SKU"],
        entity="Country", product="SKU",
    ),
}


# ===========================================================================
# Sidebar
# ===========================================================================
avail = available()
if not avail:
    st.error("No data found. Run `python etl.py --raw raw --out data` first.")
    st.stop()

# The brand picker lives in the main column, not the sidebar. A collapsed
# sidebar made the other two dashboards effectively invisible -- the app looked
# like it only had Sadafco.
bsel, _sp = st.columns([2, 3])
with bsel:
    brand = st.segmented_control(
        "Dashboard", avail,
        default=st.session_state.get("brand_pick", avail[0]),
        key="brand_pick", label_visibility="collapsed",
    ) or avail[0]

with st.sidebar:
    st.markdown("### Sales Intelligence")
    st.caption("Brand switcher is at the top of the page.")
    acc = BRANDS[brand]["accent"]
    cfg = CFG[brand]
    cols = schema(brand)
    mlist = months(brand)

    st.markdown(f"<div style='color:{MUTED};font-size:.7rem;letter-spacing:.08em;"
                f"text-transform:uppercase;margin:.6rem 0 .2rem'>Period</div>",
                unsafe_allow_html=True)

    # Slider is the fast path: it emits one rerun on release, not per drag.
    if len(mlist) > 1:
        i0, i1 = st.select_slider(
            "Months", options=list(range(len(mlist))),
            value=(0, len(mlist) - 1),
            format_func=lambda i: mlist[i], label_visibility="collapsed",
        )
    else:
        i0 = i1 = 0
    m0, m1 = mlist[i0], mlist[i1]

    st.markdown(f"<div style='color:{MUTED};font-size:.7rem;letter-spacing:.08em;"
                f"text-transform:uppercase;margin:.9rem 0 .2rem'>Filters</div>",
                unsafe_allow_html=True)

    # Only offer dimensions with a sane cardinality as multiselects; a
    # 46k-member Customer list would freeze the widget, so it gets a
    # search box instead.
    sel: list[tuple[str, tuple[str, ...]]] = []
    for d in cfg["dims"]:
        if d not in cols:
            continue
        opts = distinct(brand, d)
        if len(opts) > 300:
            continue
        picked = st.multiselect(d, opts, default=[], placeholder=f"All {d.lower()}")
        if picked:
            sel.append((d, tuple(picked)))
    filters = tuple(sel)

    search = ""
    big = [d for d in cfg["dims"] if d in cols and len(distinct(brand, d)) > 300]
    if big:
        search = st.text_input(f"Search {big[0].lower()}", placeholder="type to match…").strip()
        if search:
            filters = filters + ((big[0], tuple(
                v for v in distinct(brand, big[0]) if search.lower() in v.lower()
            )[:500]),)

    st.markdown(f"<hr style='border-color:{LINE}'>", unsafe_allow_html=True)
    st.caption(f"{len(mlist)} months loaded · {m0} → {m1}")

t_start = time.perf_counter()
VAL, VOL = cfg["value"], cfg["volume"]

# ===========================================================================
# Masthead
# ===========================================================================
st.markdown(
    f'<div class="masthead"><div class="rule" style="background:{acc}"></div>'
    f'<h1>{brand}</h1>'
    f'<span style="color:{MUTED};font-size:.8rem">{BRANDS[brand]["sub"]}</span>'
    f'<span class="sub">{m0} → {m1}</span></div>',
    unsafe_allow_html=True,
)

# --- headline aggregates -----------------------------------------------------
tot = q(brand, f'SUM("{VAL}") v, SUM("{VOL}") u, COUNT(*) n', filters, m0, m1)
if tot.empty or pd.isna(tot["v"][0]):
    st.warning("No rows match these filters.")
    st.stop()

TV, TU = float(tot["v"][0]), float(tot["u"][0])

ts = q(brand, f'MonthKey mk, SUM("{VAL}") v, SUM("{VOL}") u',
       filters, m0, m1, group="MonthKey", order="MonthKey")
ts["d"] = pd.to_datetime(ts["mk"] + "-01")

# Latest complete month vs. the same month a year earlier -- the comparison a
# sales review actually opens with.
last_mk = ts["mk"].iloc[-1]
last_v = float(ts["v"].iloc[-1])
prev_v = float(ts["v"].iloc[-2]) if len(ts) > 1 else None
mom = growth(last_v, prev_v) if prev_v is not None else None

ly_mk = f"{int(last_mk[:4])-1}-{last_mk[5:]}"
ly_row = ts[ts["mk"] == ly_mk]
yoy = growth(last_v, float(ly_row["v"].iloc[0])) if len(ly_row) else None

avg_v = ts["v"].mean()
best = ts.loc[ts["v"].idxmax()]
asp = (TV / TU) if TU else 0

k = st.columns(5)
k[0].markdown(kpi(cfg["vlabel"], money(TV),
                  foot=f"{len(ts)} months · {int(tot['n'][0]):,} rows", accent=acc),
              unsafe_allow_html=True)
k[1].markdown(kpi(cfg["qlabel"], num(TU), foot="total volume", accent=CUR),
              unsafe_allow_html=True)
k[2].markdown(kpi(f"{last_mk} value", money(last_v), mom,
                  foot="vs prior month", accent=CUR), unsafe_allow_html=True)
k[3].markdown(kpi("YoY (latest mo.)",
                  f"{yoy*100:+.1f}%" if yoy is not None else "n/a", yoy,
                  foot=f"vs {ly_mk}" if yoy is not None else "no prior year",
                  accent=PRV), unsafe_allow_html=True)
k[4].markdown(kpi("Avg price / unit", money(asp),
                  foot=f"peak {best['mk']} · {money(best['v'])}", accent=POS),
              unsafe_allow_html=True)

st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

# Streamlit executes the body of EVERY st.tabs() branch on every rerun, even
# the six you cannot see. On Friesland the customer aggregate alone is ~295 ms,
# so eager tabs would pay for all seven panels on each filter change. A
# segmented control renders exactly one panel per rerun -- the single biggest
# win available here.
VIEWS = ["Trend", "Mix", "Products", "Customers", "Seasonality", "Movers", "Data"]
view = st.segmented_control("View", VIEWS, default="Trend",
                            label_visibility="collapsed", key="view")
if not view:
    view = "Trend"

# ---------------------------------------------------------------------------
# 1. TREND
# ---------------------------------------------------------------------------
if view == "Trend":
    c1, c2 = st.columns([3, 2])

    with c1:
        f = go.Figure()
        f.add_bar(x=ts["d"], y=ts["v"], name=cfg["vlabel"],
                  marker_color=acc, opacity=.72,
                  hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra></extra>")
        if len(ts) >= 3:
            f.add_scatter(x=ts["d"], y=ts["v"].rolling(3, min_periods=1).mean(),
                          name="3-mo moving avg", mode="lines",
                          line=dict(color=PRV, width=2.5))
        f.add_hline(y=avg_v, line=dict(color=MUTED, width=1, dash="dot"),
                    annotation_text=f"avg {money(avg_v)}",
                    annotation_font_color=MUTED, annotation_font_size=9)
        f.update_layout(title=f"Monthly {cfg['vlabel'].split(' (')[0].lower()}")
        st.plotly_chart(style(f, 340, rev="trend"), width='stretch',
                        config={"displayModeBar": False})

        # Year-over-year overlay: one line per calendar year on a shared
        # month axis. This is where seasonality vs. real growth separates.
        yy = ts.copy()
        yy["yr"] = yy["d"].dt.year
        yy["mo"] = yy["d"].dt.month
        if yy["yr"].nunique() > 1:
            f2 = go.Figure()
            for i, (yr, g) in enumerate(yy.groupby("yr")):
                f2.add_scatter(x=g["mo"], y=g["v"], name=str(yr), mode="lines+markers",
                               line=dict(width=2.5, color=SEQ[i % len(SEQ)]),
                               marker=dict(size=5))
            f2.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                            ticktext=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
            f2.update_layout(title="Same-month comparison by year")
            st.plotly_chart(style(f2, 280, rev="yoy"), width='stretch',
                            config={"displayModeBar": False})

    with c2:
        # Month-on-month % change — a waterfall-style read of momentum.
        g = ts.copy()
        g["pct"] = g["v"].pct_change() * 100
        g = g.dropna(subset=["pct"])
        if len(g):
            f3 = go.Figure(go.Bar(
                x=g["d"], y=g["pct"],
                marker_color=[POS if v >= 0 else NEG for v in g["pct"]],
                hovertemplate="%{x|%b %Y}<br>%{y:+.1f}%<extra></extra>"))
            f3.update_layout(title="Month-on-month growth %")
            f3.add_hline(y=0, line=dict(color=MUTED, width=1))
            st.plotly_chart(style(f3, 250, legend=False, rev="mom"),
                            width='stretch', config={"displayModeBar": False})

        # Volume vs value on twin axes exposes price/mix effects: lines that
        # diverge mean you are selling more for less (or vice versa).
        f4 = go.Figure()
        f4.add_scatter(x=ts["d"], y=ts["v"], name=cfg["vlabel"], mode="lines",
                       line=dict(color=acc, width=2.5), fill="tozeroy",
                       fillcolor="rgba(34,211,238,.10)")
        f4.add_scatter(x=ts["d"], y=ts["u"], name=cfg["qlabel"], mode="lines",
                       line=dict(color=PRV, width=2, dash="dot"), yaxis="y2")
        f4.update_layout(title="Value vs volume",
                         yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                     color=PRV))
        st.plotly_chart(style(f4, 250, rev="vv"), width='stretch',
                        config={"displayModeBar": False})

        # Cumulative curve: steepness = run-rate, flat stretches = stalls.
        cum = ts.copy()
        cum["c"] = cum["v"].cumsum()
        f5 = go.Figure(go.Scatter(x=cum["d"], y=cum["c"], mode="lines",
                                  line=dict(color=POS, width=2.5), fill="tozeroy",
                                  fillcolor="rgba(52,211,153,.10)"))
        f5.update_layout(title="Cumulative value")
        st.plotly_chart(style(f5, 220, legend=False, rev="cum"),
                        width='stretch', config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# 2. MIX
# ---------------------------------------------------------------------------
if view == "Mix":
    dims = [d for d in cfg["dims"] if d in cols and len(distinct(brand, d)) <= 300]
    if not dims:
        st.info("No low-cardinality dimension available for mix analysis.")
    else:
        dim = st.selectbox("Break down by", dims, index=0, key="mixdim")

        by = q(brand, f'"{dim}" k, SUM("{VAL}") v, SUM("{VOL}") u',
               filters, m0, m1, group=f'"{dim}"', order="v DESC", limit=40)
        by = by[by["v"] != 0]

        c1, c2 = st.columns([3, 2])
        with c1:
            top = by.head(15).sort_values("v")
            f = go.Figure(go.Bar(
                x=top["v"], y=top["k"].astype(str), orientation="h",
                marker=dict(color=top["v"], colorscale=[[0, LINE], [1, acc]]),
                hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
            f.update_layout(title=f"Top {dim} by value")
            st.plotly_chart(style(f, 420, legend=False, rev="mixbar"),
                            width='stretch', config={"displayModeBar": False})

            # Stacked share over time for the top 6 — shows whether the mix
            # is actually shifting or just riding the total.
            top6 = by.head(6)["k"].astype(str).tolist()
            stk = q(brand, f'MonthKey mk, "{dim}" k, SUM("{VAL}") v',
                    filters, m0, m1, group=f'MonthKey, "{dim}"', order="MonthKey")
            stk["k"] = stk["k"].astype(str)
            stk = stk[stk["k"].isin(top6)]
            if len(stk):
                stk["d"] = pd.to_datetime(stk["mk"] + "-01")
                f2 = px.area(stk, x="d", y="v", color="k",
                             color_discrete_sequence=SEQ)
                f2.update_layout(title=f"{dim} mix over time (top 6)",
                                 legend_title_text="")
                f2.update_traces(hovertemplate="%{y:,.0f}<extra>%{fullData.name}</extra>")
                st.plotly_chart(style(f2, 300, rev="mixarea"),
                                width='stretch', config={"displayModeBar": False})

        with c2:
            f3 = go.Figure(go.Pie(
                labels=by.head(8)["k"].astype(str), values=by.head(8)["v"].abs(),
                hole=.62, marker=dict(colors=SEQ, line=dict(color=INK, width=2)),
                textinfo="percent", textfont=dict(size=10)))
            f3.update_layout(title=f"Share of value — top 8 {dim}",
                             annotations=[dict(text=money(TV), x=.5, y=.5,
                                               font=dict(size=15, color=TEXT,
                                                         family="JetBrains Mono"),
                                               showarrow=False)])
            st.plotly_chart(style(f3, 320, rev="pie"), width='stretch',
                            config={"displayModeBar": False})

            # Pareto: how many members carry 80% of value. Concentration risk
            # in one glance.
            pv = by.sort_values("v", ascending=False).copy()
            pv["cum"] = pv["v"].cumsum() / pv["v"].sum() * 100
            n80 = int((pv["cum"] <= 80).sum()) + 1
            f4 = make_pareto(pv, acc, f"Pareto — {dim} concentration")
            st.plotly_chart(style(f4, 300, rev="pareto"), width='stretch',
                            config={"displayModeBar": False})
            st.caption(f"**{n80}** of {len(pv)} {dim.lower()}s carry 80% of value.")

# ---------------------------------------------------------------------------
# 3. PRODUCTS
# ---------------------------------------------------------------------------
if view == "Products":
    pcol = cfg["product"] if cfg["product"] in cols else cfg["dims"][-1]
    pr = q(brand, f'"{pcol}" k, SUM("{VAL}") v, SUM("{VOL}") u',
           filters, m0, m1, group=f'"{pcol}"', order="v DESC", limit=300)
    pr = pr[pr["v"] > 0].copy()
    pr["asp"] = pr["v"] / pr["u"].replace(0, pd.NA)

    c1, c2 = st.columns([3, 2])
    with c1:
        top = pr.head(20).sort_values("v")
        top["lab"] = top["k"].astype(str).str.slice(0, 44)
        f = go.Figure(go.Bar(x=top["v"], y=top["lab"], orientation="h",
                             marker_color=acc, opacity=.85,
                             hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
        f.update_layout(title=f"Top 20 {pcol} by value")
        st.plotly_chart(style(f, 520, legend=False, rev="prodbar"),
                        width='stretch', config={"displayModeBar": False})

    with c2:
        # Value vs volume scatter. Off-diagonal points are the interesting
        # ones: high value / low units = premium; the reverse = traffic driver.
        sc = pr.head(120).dropna(subset=["asp"])
        if len(sc):
            f2 = go.Figure(go.Scatter(
                x=sc["u"], y=sc["v"], mode="markers",
                marker=dict(size=9, color=sc["asp"], colorscale="Viridis",
                            line=dict(width=.5, color=LINE),
                            colorbar=dict(title="price/unit", thickness=8,
                                          tickfont=dict(size=8))),
                text=sc["k"].astype(str).str.slice(0, 40),
                hovertemplate="%{text}<br>units %{x:,.0f}<br>value %{y:,.0f}<extra></extra>"))
            f2.update_layout(title="Value vs volume (colour = price/unit)")
            st.plotly_chart(style(f2, 300, legend=False, rev="scat"),
                            width='stretch', config={"displayModeBar": False})

        # Distribution of product value — a long right tail is the norm;
        # the shape tells you how many SKUs are effectively dormant.
        f3 = go.Figure(go.Histogram(x=pr["v"], nbinsx=36, marker_color=acc,
                                    opacity=.8))
        f3.update_layout(title=f"Distribution of value per {pcol}", bargap=.04)
        st.plotly_chart(style(f3, 240, legend=False, rev="hist"),
                        width='stretch', config={"displayModeBar": False})

    st.dataframe(
        pr.head(60).rename(columns={"k": pcol, "v": cfg["vlabel"],
                                    "u": cfg["qlabel"], "asp": "Price/unit"}),
        width='stretch', hide_index=True, height=260,
        column_config={
            cfg["vlabel"]: st.column_config.NumberColumn(format="%.0f"),
            cfg["qlabel"]: st.column_config.NumberColumn(format="%.0f"),
            "Price/unit": st.column_config.NumberColumn(format="%.2f"),
        })

# ---------------------------------------------------------------------------
# 4. CUSTOMERS
# ---------------------------------------------------------------------------
if view == "Customers":
    ecol = "Customer" if "Customer" in cols else cfg["entity"]
    cu = q(brand, f'"{ecol}" k, SUM("{VAL}") v, SUM("{VOL}") u, '
                  f'COUNT(DISTINCT MonthKey) AS "n_months"',
           filters, m0, m1, group=f'"{ecol}"', order="v DESC", limit=400)
    cu = cu[cu["v"] > 0].copy()

    if cu.empty:
        st.info("No customer-level data for this selection.")
    else:
        tot_v = cu["v"].sum()
        cu["share"] = cu["v"] / tot_v * 100
        cu["cum"] = cu["share"].cumsum()

        m = st.columns(4)
        m[0].markdown(kpi("Active " + ecol.lower() + "s", f"{len(cu):,}",
                          foot="with value in range", accent=acc), unsafe_allow_html=True)
        m[1].markdown(kpi("Top 10 share", f"{cu.head(10)['share'].sum():.1f}%",
                          foot="concentration", accent=CUR), unsafe_allow_html=True)
        m[2].markdown(kpi("Median value", money(cu["v"].median()),
                          foot=f"mean {money(cu['v'].mean())}", accent=POS),
                      unsafe_allow_html=True)
        m[3].markdown(kpi("Avg active months", f"{cu['n_months'].mean():.1f}",
                          foot=f"of {len(ts)} in range", accent=PRV),
                      unsafe_allow_html=True)

        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([3, 2])
        with c1:
            top = cu.head(18).sort_values("v")
            top["lab"] = top["k"].astype(str).str.slice(0, 40)
            f = go.Figure(go.Bar(x=top["v"], y=top["lab"], orientation="h",
                                 marker_color=acc, opacity=.85,
                                 hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
            f.update_layout(title=f"Top {ecol}s by value")
            st.plotly_chart(style(f, 470, legend=False, rev="cust"),
                            width='stretch', config={"displayModeBar": False})
        with c2:
            f2 = make_pareto(cu, acc, "Customer concentration (Pareto)")
            st.plotly_chart(style(f2, 300, rev="cpareto"), width='stretch',
                            config={"displayModeBar": False})
            n80 = int((cu["cum"] <= 80).sum()) + 1
            st.caption(f"**{n80}** {ecol.lower()}s ({n80/len(cu)*100:.1f}%) drive 80% of value.")

            f3 = go.Figure(go.Histogram(x=cu["n_months"], nbinsx=len(ts) or 12,
                                        marker_color=CUR, opacity=.85))
            f3.update_layout(title="How many months each is active", bargap=.06)
            st.plotly_chart(style(f3, 220, legend=False, rev="chist"),
                            width='stretch', config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# 5. SEASONALITY
# ---------------------------------------------------------------------------
if view == "Seasonality":
    hz = ts.copy()
    hz["yr"] = hz["d"].dt.year
    hz["mo"] = hz["d"].dt.month
    piv = hz.pivot_table(index="yr", columns="mo", values="v", aggfunc="sum")
    MO = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    piv = piv.reindex(columns=range(1, 13))

    c1, c2 = st.columns([3, 2])
    with c1:
        f = go.Figure(go.Heatmap(
            z=piv.values, x=MO, y=[str(i) for i in piv.index],
            colorscale=[[0, PANEL], [.5, acc], [1, "#FFFFFF"]],
            hovertemplate="%{y} %{x}<br>%{z:,.0f}<extra></extra>",
            colorbar=dict(thickness=8, tickfont=dict(size=8))))
        f.update_layout(title="Value heatmap — year × month")
        st.plotly_chart(style(f, 260, legend=False, rev="heat"),
                        width='stretch', config={"displayModeBar": False})

        # Index each month against the overall average = 100. Above 100 is a
        # structurally strong month, independent of which year it fell in.
        idx = hz.groupby("mo")["v"].mean()
        idx = (idx / idx.mean() * 100).reindex(range(1, 13))
        f2 = go.Figure(go.Bar(
            x=MO, y=idx.values,
            marker_color=[POS if (v or 0) >= 100 else NEG for v in idx.values],
            hovertemplate="%{x}<br>index %{y:.0f}<extra></extra>"))
        f2.add_hline(y=100, line=dict(color=MUTED, dash="dot", width=1))
        f2.update_layout(title="Seasonality index (100 = average month)")
        st.plotly_chart(style(f2, 240, legend=False, rev="season"),
                        width='stretch', config={"displayModeBar": False})

    with c2:
        # Spread of monthly outcomes per calendar year — a widening box means
        # the business is getting less predictable, not just bigger.
        f3 = go.Figure()
        for i, (yr, g) in enumerate(hz.groupby("yr")):
            f3.add_box(y=g["v"], name=str(yr), marker_color=SEQ[i % len(SEQ)],
                       boxmean=True)
        f3.update_layout(title="Monthly spread by year")
        st.plotly_chart(style(f3, 300, legend=False, rev="box"),
                        width='stretch', config={"displayModeBar": False})

        if len(ts) >= 4:
            q_ = ts.copy()
            q_["q"] = q_["d"].dt.year.astype(str) + " Q" + q_["d"].dt.quarter.astype(str)
            qa = q_.groupby("q", as_index=False)["v"].sum()
            f4 = go.Figure(go.Bar(x=qa["q"], y=qa["v"], marker_color=acc, opacity=.85,
                                  hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
            f4.update_layout(title="Quarterly value")
            st.plotly_chart(style(f4, 260, legend=False, rev="qtr"),
                            width='stretch', config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# 6. MOVERS
# ---------------------------------------------------------------------------
if view == "Movers":
    st.caption("Compares the last month in range against the month before it.")
    dims2 = [d for d in cfg["dims"] if d in cols]
    mdim = st.selectbox("Movement by", dims2,
                        index=min(len(dims2) - 1, dims2.index(cfg["product"])
                                  if cfg["product"] in dims2 else 0), key="mvdim")

    if len(ts) < 2:
        st.info("Need at least two months in range.")
    else:
        cur_mk, prv_mk = ts["mk"].iloc[-1], ts["mk"].iloc[-2]
        cur = q(brand, f'"{mdim}" k, SUM("{VAL}") v', filters, cur_mk, cur_mk,
                group=f'"{mdim}"')
        prv = q(brand, f'"{mdim}" k, SUM("{VAL}") v', filters, prv_mk, prv_mk,
                group=f'"{mdim}"')
        mg = cur.merge(prv, on="k", how="outer", suffixes=("_c", "_p")).fillna(0)
        mg["chg"] = mg["v_c"] - mg["v_p"]
        mg["pct"] = mg.apply(lambda r: (r["chg"] / abs(r["v_p"]) * 100)
                             if r["v_p"] else None, axis=1)
        mg = mg[(mg["v_c"] != 0) | (mg["v_p"] != 0)]

        c1, c2 = st.columns(2)
        for col, title, asc, colour in ((c1, f"Gainers · {prv_mk} → {cur_mk}", False, POS),
                                        (c2, f"Decliners · {prv_mk} → {cur_mk}", True, NEG)):
            d = mg.sort_values("chg", ascending=asc).head(12).sort_values("chg")
            d = d[d["chg"] > 0] if not asc else d[d["chg"] < 0]
            with col:
                if d.empty:
                    st.info("None.")
                    continue
                lab = d["k"].astype(str).str.slice(0, 38)
                f = go.Figure(go.Bar(x=d["chg"], y=lab, orientation="h",
                                     marker_color=colour, opacity=.85,
                                     hovertemplate="%{y}<br>%{x:+,.0f}<extra></extra>"))
                f.update_layout(title=title)
                st.plotly_chart(style(f, 340, legend=False, rev=f"mv{asc}"),
                                width='stretch',
                                config={"displayModeBar": False})

        # Waterfall from prior to current month, biggest swings named.
        w = mg.reindex(mg["chg"].abs().sort_values(ascending=False).index).head(12)
        f2 = go.Figure(go.Waterfall(
            orientation="v",
            x=[prv_mk] + w["k"].astype(str).str.slice(0, 18).tolist() + ["other", cur_mk],
            measure=["absolute"] + ["relative"] * len(w) + ["relative", "total"],
            y=[float(prv["v"].sum())] + w["chg"].tolist()
              + [float(mg["chg"].sum() - w["chg"].sum()), 0],
            increasing=dict(marker=dict(color=POS)),
            decreasing=dict(marker=dict(color=NEG)),
            totals=dict(marker=dict(color=acc)),
            connector=dict(line=dict(color=LINE)),
        ))
        f2.update_layout(title=f"What moved the month: {prv_mk} → {cur_mk}")
        f2.update_xaxes(tickangle=-40, tickfont=dict(size=9))
        st.plotly_chart(style(f2, 360, legend=False, rev="wf"),
                        width='stretch', config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# 7. DATA
# ---------------------------------------------------------------------------
if view == "Data":
    gdims = st.multiselect("Group by", [d for d in cfg["dims"] if d in cols],
                           default=[cfg["dims"][0]], key="gdim")
    gcols = ", ".join(f'"{d}"' for d in gdims) if gdims else "MonthKey"
    tb = q(brand, f'{gcols}, SUM("{VAL}") "{cfg["vlabel"]}", SUM("{VOL}") "{cfg["qlabel"]}"',
           filters, m0, m1, group=gcols, order=f'"{cfg["vlabel"]}" DESC', limit=2000)
    st.dataframe(tb, width='stretch', hide_index=True, height=430)
    st.download_button("Download CSV", tb.to_csv(index=False).encode(),
                       f"{brand.lower()}_{m0}_{m1}.csv", "text/csv")

st.markdown(f'<div class="perf">rendered in {(time.perf_counter()-t_start)*1000:.0f} ms · '
            f'{brand} · {len(ts)} months</div>', unsafe_allow_html=True)