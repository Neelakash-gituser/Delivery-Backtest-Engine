"""
tearsheet.py — Comprehensive Performance Tearsheet Module
Compatible with: Delivery-Backtest-Engine (Neelakash-gituser)

All stats returned as styled pandas DataFrames.
All charts rendered in Plotly (works in Jupyter & VSCode with Jupyter extension).

Usage:
    from tearsheet import Tearsheet

    ts = Tearsheet(
        portfolio_values = pd.Series(...),   # Daily NAV, DatetimeIndex
        benchmark_values = pd.Series(...),   # Daily benchmark NAV (optional)
        trades           = pd.DataFrame(...),# Trade blotter (optional)
        config           = config_dict,      # config.yaml as dict (optional)
        risk_free_rate   = 0.065,            # Annual RFR (default 6.5% India)
    )

    ts.summary()            # → returns dict of styled DataFrames (all metric tables)
    ts.plot_all()           # → renders all Plotly figures inline
    ts.plot_returns()       # → cumulative returns chart
    ts.plot_drawdown()      # → drawdown chart
    ts.plot_rolling()       # → rolling Sharpe + volatility
    ts.plot_distribution()  # → daily return distribution
    ts.plot_monthly()       # → monthly returns heatmap
    ts.plot_top_drawdowns() # → top drawdown periods table chart
    ts.metrics()            # → flat dict of all raw metric values
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import display


# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

FONT = "Menlo, 'Courier New', monospace"

COLORS = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "border":   "#30363d",
    "text":     "#e6edf3",
    "muted":    "#8b949e",
    "green":    "#3fb950",
    "red":      "#f85149",
    "blue":     "#58a6ff",
    "yellow":   "#d29922",
    "purple":   "#bc8cff",
    "orange":   "#ffa657",
    "grid":     "#21262d",
}

BASE_LAYOUT = dict(
    font=dict(family=FONT, size=12, color=COLORS["text"]),
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["panel"],
    xaxis=dict(
        gridcolor=COLORS["grid"], gridwidth=0.5,
        linecolor=COLORS["border"], tickcolor=COLORS["muted"],
        tickfont=dict(family=FONT, size=10),
    ),
    yaxis=dict(
        gridcolor=COLORS["grid"], gridwidth=0.5,
        linecolor=COLORS["border"], tickcolor=COLORS["muted"],
        tickfont=dict(family=FONT, size=10),
    ),
    legend=dict(
        bgcolor=COLORS["panel"], bordercolor=COLORS["border"],
        borderwidth=1, font=dict(family=FONT, size=11),
    ),
    margin=dict(l=60, r=40, t=60, b=50),
    hoverlabel=dict(
        bgcolor=COLORS["panel"], bordercolor=COLORS["border"],
        font=dict(family=FONT, size=11),
    ),
)


def _apply_layout(fig, title: str = "", height: int = 420) -> go.Figure:
    layout = dict(**BASE_LAYOUT, title=dict(
        text=title, font=dict(family=FONT, size=14, color=COLORS["text"]),
        x=0.02, xanchor="left",
    ), height=height)
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _rets(prices):     return prices.pct_change().dropna()
def _total_ret(p):     return p.iloc[-1] / p.iloc[0] - 1
def _n_years(p):       return (p.index[-1] - p.index[0]).days / 365.25

def _cagr(p):
    n = _n_years(p)
    return (p.iloc[-1] / p.iloc[0]) ** (1 / n) - 1 if n > 0 else 0.0

def _vol(p, T=252):    return _rets(p).std() * np.sqrt(T)

def _sharpe(p, rf=0.065, T=252):
    r = _rets(p)
    v = r.std() * np.sqrt(T)
    return (r.mean() * T - rf) / v if v else 0.0

def _sortino(p, rf=0.065, T=252):
    r = _rets(p)
    d = r[r < 0].std() * np.sqrt(T)
    return (r.mean() * T - rf) / d if d else 0.0

def _dd_series(p):
    return (p - p.cummax()) / p.cummax()

def _max_dd(p):        return _dd_series(p).min()

def _calmar(p):
    mdd = _max_dd(p)
    return _cagr(p) / abs(mdd) if mdd else 0.0

def _max_dd_dur(p):
    dd = _dd_series(p) < 0
    durs, start = [], None
    for d, v in dd.items():
        if v and start is None:  start = d
        elif not v and start:
            durs.append((d - start).days); start = None
    if start: durs.append((p.index[-1] - start).days)
    return max(durs) if durs else 0

def _recovery_factor(p):
    mdd = _max_dd(p)
    return _total_ret(p) / abs(mdd) if mdd else 0.0

def _var(r, c=0.95):   return float(np.percentile(r.dropna(), (1 - c) * 100))
def _cvar(r, c=0.95):
    v = _var(r, c)
    return float(r[r <= v].mean())

def _omega(r, thr=0.0):
    g = (r[r > thr] - thr).sum()
    l = (thr - r[r < thr]).sum()
    return g / l if l else np.inf

def _tail_ratio(r):
    p95 = abs(np.percentile(r.dropna(), 95))
    p5  = abs(np.percentile(r.dropna(), 5))
    return p95 / p5 if p5 else np.inf

def _win_rate(r):      return (r > 0).sum() / len(r) if len(r) else 0.0
def _profit_factor(r):
    g = r[r > 0].sum(); l = abs(r[r < 0].sum())
    return g / l if l else np.inf

def _beta(pr, br):
    a = pd.concat([pr, br], axis=1).dropna()
    if len(a) < 2: return 0.0
    c = np.cov(a.iloc[:, 0], a.iloc[:, 1])
    return c[0, 1] / c[1, 1] if c[1, 1] else 0.0

def _alpha(pr, br, rf=0.065, T=252):
    b = _beta(pr, br)
    return pr.mean() * T - (rf + b * (br.mean() * T - rf))

def _ir(pr, br, T=252):
    a = pr - br; te = a.std() * np.sqrt(T)
    return (a.mean() * T) / te if te else 0.0

def _treynor(pr, br, rf=0.065, T=252):
    b = _beta(pr, br)
    return (pr.mean() * T - rf) / b if b else 0.0

def _rolling_sharpe(p, w=63, rf=0.065, T=252):
    r = _rets(p)
    return (r.rolling(w).mean() * T - rf) / (r.rolling(w).std() * np.sqrt(T))

def _rolling_vol(p, w=21, T=252):
    return _rets(p).rolling(w).std() * np.sqrt(T) * 100

def _monthly_pivot(p):
    m = p.resample("ME").last().pct_change().dropna()
    m.index = m.index.to_period("M")
    df = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    piv = df.pivot(index="year", columns="month", values="ret")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    piv.columns = [month_names[c - 1] for c in piv.columns]
    piv["Annual"] = piv.apply(lambda r: np.prod([1 + v for v in r.dropna()]) - 1, axis=1)
    return piv

def _top_drawdowns(p, n=5):
    dd = _dd_series(p)
    rows, temp = [], dd.copy()
    for _ in range(n):
        if temp.min() == 0: break
        trough = temp.idxmin()
        val = temp[trough]
        pre = temp[:trough]
        start = pre[pre == 0].index[-1] if (pre == 0).any() else temp.index[0]
        post = temp[trough:]
        rec_cands = post[post >= 0]
        rec = rec_cands.index[0] if len(rec_cands) else None
        rows.append({
            "Start":       start.strftime("%Y-%m-%d"),
            "Trough":      trough.strftime("%Y-%m-%d"),
            "Recovery":    rec.strftime("%Y-%m-%d") if rec else "Ongoing",
            "Depth (%)":   round(val * 100, 2),
            "Duration (d)":(trough - start).days,
            "Recovery (d)":(rec - trough).days if rec else None,
        })
        end = rec if rec else temp.index[-1]
        temp[start:end] = 0
    return pd.DataFrame(rows)

def _trade_stats_dict(trades):
    if trades is None or trades.empty: return {}
    if "pnl" in trades.columns:
        pnl = trades["pnl"].dropna()
    elif {"buy_price", "sell_price", "quantity"}.issubset(trades.columns):
        pnl = (trades["sell_price"] - trades["buy_price"]) * trades["quantity"]
    else:
        return {}
    w = pnl[pnl > 0]; l = pnl[pnl < 0]
    wr = len(w) / len(pnl)
    pf = w.sum() / abs(l.sum()) if len(l) else np.inf
    return {
        "Total Trades":   len(pnl),
        "Winning Trades": len(w),
        "Losing Trades":  len(l),
        "Win Rate":       wr,
        "Profit Factor":  pf,
        "Avg Win (₹)":    w.mean() if len(w) else 0.0,
        "Avg Loss (₹)":   l.mean() if len(l) else 0.0,
        "Largest Win (₹)":pnl.max(),
        "Largest Loss (₹)":pnl.min(),
        "Expectancy (₹)": wr * (w.mean() if len(w) else 0) + (1 - wr) * (l.mean() if len(l) else 0),
        "Total P&L (₹)":  pnl.sum(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# TABLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v):  return f"{v * 100:.2f}%"
def _f2(v):   return f"{v:.4f}"
def _cur(v):  return f"₹{v:,.2f}"
def _int(v):  return f"{int(v):,}"

def _style_df(df):
    """Apply consistent styling to all DataFrames for tearsheet tables."""
    df = df.copy()

    # ✔ force dates/objects to stay untouched
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # optional safety: prevent accidental gradient calls elsewhere
    df[numeric_cols] = df[numeric_cols].astype(float)

    return (
        df.style
        .set_properties(**{
            "background-color": "#161b22",
            "color": "#e6edf3",
            "font-family": "Menlo, monospace",
            "font-size": "11px",
            "padding": "4px 8px"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#30363d"),
                    ("color", "#e6edf3"),
                    ("font-weight", "bold"),
                    ("padding", "6px 8px")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #30363d")
                ]
            }
        ])
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Tearsheet:
    """
    Comprehensive Plotly-based tearsheet for delivery/position backtests.

    Parameters
    ----------
    portfolio_values : pd.Series
        Daily portfolio NAV, DatetimeIndex.
    benchmark_values : pd.Series, optional
        Daily benchmark NAV. Pass None to skip benchmark comparisons.
    trades : pd.DataFrame, optional
        Trade blotter. Needs column 'pnl', or ('buy_price','sell_price','quantity').
        Optional 'ticker' column enables per-stock breakdown.
    config : dict, optional
        Loaded config.yaml as dict — shown in summary header.
    risk_free_rate : float
        Annual risk-free rate. Default 0.065 (6.5% — Indian T-Bill proxy).
    periods : int
        Trading days per year. Default 252.
    """

    def __init__(
        self,
        portfolio_values: pd.Series,
        benchmark_values: pd.Series = None,
        trades: pd.DataFrame = None,
        config: dict = None,
        risk_free_rate: float = 0.065,
        periods: int = 252,
    ):
        self.pv      = portfolio_values.dropna().sort_index()
        self.bv      = benchmark_values.dropna().sort_index() if benchmark_values is not None else None
        self.trades  = trades
        self.config  = config or {}
        self.rf      = risk_free_rate
        self.T       = periods

        self._pr     = _rets(self.pv)
        self._br     = _rets(self.bv) if self.bv is not None else None
        self._cache  = None

    # ── RAW METRICS ───────────────────────────────────────────────────────────

    def metrics(self) -> dict:
        """Return flat dict of all raw metric values."""
        if self._cache:
            return self._cache

        pr, T, rf = self._pr, self.T, self.rf
        m = {}

        # Period
        m["start_date"]          = str(self.pv.index[0].date())
        m["end_date"]            = str(self.pv.index[-1].date())
        m["n_days"]              = len(self.pv)
        m["n_years"]             = round(_n_years(self.pv), 2)

        # Returns
        m["total_return"]        = _total_ret(self.pv)
        m["cagr"]                = _cagr(self.pv)
        m["volatility"]          = _vol(self.pv, T)

        # Ratios
        m["sharpe"]              = _sharpe(self.pv, rf, T)
        m["sortino"]             = _sortino(self.pv, rf, T)
        m["calmar"]              = _calmar(self.pv)
        m["omega"]               = _omega(pr)
        m["tail_ratio"]          = _tail_ratio(pr)

        # Drawdown
        m["max_drawdown"]        = _max_dd(self.pv)
        m["max_dd_duration_days"]= _max_dd_dur(self.pv)
        m["recovery_factor"]     = _recovery_factor(self.pv)

        # Distribution
        m["best_day"]            = float(pr.max())
        m["worst_day"]           = float(pr.min())
        m["win_rate"]            = _win_rate(pr)
        m["profit_factor"]       = _profit_factor(pr)
        avg_w, avg_l             = pr[pr > 0].mean() if (pr > 0).any() else 0, \
                                   pr[pr < 0].mean() if (pr < 0).any() else 0
        m["avg_win_day"]         = float(avg_w)
        m["avg_loss_day"]        = float(avg_l)
        m["skewness"]            = float(pr.skew())
        m["kurtosis"]            = float(pr.kurtosis())
        m["var_95"]              = _var(pr)
        m["cvar_95"]             = _cvar(pr)

        # Benchmark
        if self.bv is not None:
            br = self._br
            m["bench_total_return"] = _total_ret(self.bv)
            m["bench_cagr"]         = _cagr(self.bv)
            m["bench_vol"]          = _vol(self.bv, T)
            m["bench_sharpe"]       = _sharpe(self.bv, rf, T)
            m["bench_max_dd"]       = _max_dd(self.bv)
            m["alpha"]              = _alpha(pr, br, rf, T)
            m["beta"]               = _beta(pr, br)
            m["information_ratio"]  = _ir(pr, br, T)
            m["treynor"]            = _treynor(pr, br, rf, T)

        # Trades
        m.update(_trade_stats_dict(self.trades))
        self._cache = m

        return m

    # ── SUMMARY TABLES ────────────────────────────────────────────────────────
    def summary(self) -> dict:
        """Return dict of styled DataFrames summarizing all key metrics."""
        m = self.metrics()
        tables = {}
        df = pd.DataFrame.from_dict(m, orient="index", columns=["Value"])

        # ── Clean metric names ─────────────────────────────
        df.index = df.index.str.replace("_", " ").str.title()

        # only CAGR in caps
        df.rename(index={"Cagr": "CAGR", "Var 95":"VaR 95", "Cvar 95":"CVaR 95", "Bench Cagr": "Bench CAGR", "Bench Max Dd": "Bench Max Drawdown",
                         "Max Dd Duration Days": "Max Drawdown Duration (Days)"}, inplace=True)

        # ── formatting helper ─────────────────────────────
        def format_value(k, v):
            if isinstance(v, (int, float, np.floating)):
                if k in [
                    "Total Return", "CAGR", "Volatility",
                    "Win Rate", "Var 95", "Cvar 95",
                    "Bench Total Return", "Bench Cagr",
                    "Bench Vol", "Bench Max Dd", "Max Drawdown"
                ]:
                    return f"{v * 100:.2f}%"
                return f"{v:.2f}"
            return v

        df["Value"] = [
            format_value(k, v) for k, v in df["Value"].items()
        ]

        tables["summary"] = _style_df(df)

        return tables

    # ── CHARTS ────────────────────────────────────────────────────────────────

    def plot_returns(self) -> go.Figure:
        """Cumulative returns chart: portfolio vs benchmark (in bps)."""
        pv_bps = (self.pv / self.pv.iloc[0] - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pv_bps.index,
            y=pv_bps.values,
            name="Portfolio",
            line=dict(color=COLORS["blue"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.0f}%<extra></extra>",
        ))

        if self.bv is not None:
            bv = self.bv.reindex(self.pv.index, method="ffill")
            bv_bps = (bv / bv.iloc[0] - 1) * 100
            bench_name = self.config.get("benchmark", "Benchmark")
            fig.add_trace(go.Scatter(
                x=bv_bps.index,
                y=bv_bps.values,
                name=bench_name,
                line=dict(color=COLORS["muted"], width=1.5, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.0f}%<extra></extra>",
            ))

        fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
        _apply_layout(fig, "Cumulative Returns (%)", height=450)
        fig.update_yaxes(ticksuffix="%")
        return fig

    def plot_drawdown(self) -> go.Figure:
        """Underwater / drawdown chart."""
        dd = _dd_series(self.pv) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color=COLORS["red"], width=1.2),
            fillcolor="rgba(248,81,73,0.25)",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
        ))

        if self.bv is not None:
            bv_dd = _dd_series(self.bv.reindex(self.pv.index, method="ffill")) * 100
            bench_name = self.config.get("benchmark", "Benchmark")
            fig.add_trace(go.Scatter(
                x=bv_dd.index, y=bv_dd.values,
                name=bench_name,
                line=dict(color=COLORS["muted"], width=1, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
            ))

        fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
        _apply_layout(fig, "Drawdown (%)", height=350)
        fig.update_yaxes(ticksuffix="%")
        return fig

    def plot_rolling(self) -> go.Figure:
        """Rolling Sharpe (63d) and Rolling Volatility (21d) — dual axis."""
        rs = _rolling_sharpe(self.pv, w=63, rf=self.rf, T=self.T).dropna()
        rv = _rolling_vol(self.pv, w=21, T=self.T)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=["Rolling Sharpe Ratio (63-day)", "Rolling Volatility 21d (Ann.)"],
        )

        # Sharpe
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs.values, name="Sharpe (63d)",
            line=dict(color=COLORS["purple"], width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.3f}<extra></extra>",
        ), row=1, col=1)
        fig.add_hline(y=0, line_color=COLORS["border"], line_width=1, row=1, col=1)
        fig.add_hline(y=1, line_color=COLORS["green"], line_width=0.8,
                      line_dash="dash", row=1, col=1)

        # Volatility
        fig.add_trace(go.Scatter(
            x=rv.index, y=rv.values, name="Vol 21d",
            line=dict(color=COLORS["orange"], width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:.2f}%<extra></extra>",
        ), row=2, col=1)
        if self.bv is not None:
            bvr = _rolling_vol(self.bv.reindex(self.pv.index, method="ffill"), w=21, T=self.T)
            bench_name = self.config.get("benchmark", "Benchmark")
            fig.add_trace(go.Scatter(
                x=bvr.index, y=bvr.values, name=f"{bench_name} Vol",
                line=dict(color=COLORS["muted"], width=1, dash="dash"),
            ), row=2, col=1)

        fig.update_layout(
            **{k: v for k, v in BASE_LAYOUT.items() if k != "xaxis" and k != "yaxis"},
            height=520, showlegend=True,
            title=dict(text="Rolling Metrics", font=dict(family=FONT, size=14, color=COLORS["text"]), x=0.02),
        )
        fig.update_xaxes(gridcolor=COLORS["grid"], linecolor=COLORS["border"],
                         tickfont=dict(family=FONT, size=10))
        fig.update_yaxes(gridcolor=COLORS["grid"], linecolor=COLORS["border"],
                         tickfont=dict(family=FONT, size=10))
        fig.update_yaxes(ticksuffix="%", row=2, col=1)
        for ann in fig.layout.annotations:
            ann.font.family = FONT
            ann.font.color  = COLORS["muted"]
            ann.font.size   = 11
        return fig

    def plot_distribution(self) -> go.Figure:
        """Daily return histogram with VaR/CVaR lines (clean, non-overlapping)."""
        pr_pct = self._pr * 100
        pos = pr_pct[pr_pct >= 0]
        neg = pr_pct[pr_pct < 0]
        m = self.metrics()

        fig = go.Figure()
        # Histogram (negative / positive split)
        fig.add_trace(go.Histogram(
            x=neg.values,
            name="Negative",
            marker_color=COLORS["red"],
            opacity=0.75,
            xbins=dict(size=0.15),
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))
        fig.add_trace(go.Histogram(
            x=pos.values,
            name="Positive",
            marker_color=COLORS["green"],
            opacity=0.75,
            xbins=dict(size=0.15),
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))

        # -------------------------
        # VaR line (manual, clean)
        # -------------------------
        var_x = m["var_95"] * 100
        fig.add_shape(
            type="line",
            x0=var_x, x1=var_x,
            y0=0, y1=1,
            yref="paper",
            line=dict(color=COLORS["yellow"], width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=var_x,
            y=1.05,
            yref="paper",
            showarrow=False,
            text=f"VaR95 {var_x:.2f}%",
            font=dict(color=COLORS["yellow"], size=10, family=FONT),
        )
        # -------------------------
        # CVaR line (slightly lower)
        # -------------------------
        cvar_x = m["cvar_95"] * 100
        fig.add_shape(
            type="line",
            x0=cvar_x, x1=cvar_x,
            y0=0, y1=1,
            yref="paper",
            line=dict(color=COLORS["orange"], width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=cvar_x,
            y=0.95,
            yref="paper",
            showarrow=False,
            text=f"CVaR95 {cvar_x:.2f}%",
            font=dict(color=COLORS["orange"], size=10, family=FONT),
        )

        # Layout
        fig.update_layout(
            barmode="overlay",
            legend=dict(orientation="h"),
        )
        fig.update_xaxes(ticksuffix="%")
        _apply_layout(fig, "Daily Return Distribution", height=420)

        return fig

    def plot_monthly(self) -> go.Figure:
        """Monthly returns heatmap."""
        piv = _monthly_pivot(self.pv)
        cols = list(piv.columns)
        years = list(piv.index.astype(str))
        z = piv.values * 100

        text = []
        for row in piv.values:
            text.append([f"{v*100:.1f}%" if not np.isnan(v) else "" for v in row])

        # Separate colorscale for Annual column
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=cols,
            y=years,
            text=text,
            texttemplate="%{text}",
            textfont=dict(family=FONT, size=10),
            colorscale=[
                [0.0,  COLORS["red"]],
                [0.5,  COLORS["panel"]],
                [1.0,  COLORS["green"]],
            ],
            zmid=0,
            showscale=True,
            colorbar=dict(
                ticksuffix="%",
                tickfont=dict(family=FONT, size=10, color=COLORS["muted"]),
                outlinecolor=COLORS["border"],
                outlinewidth=1,
            ),
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        ))
        _apply_layout(fig, "Monthly Returns Heatmap (%)", height=max(300, len(years) * 38 + 100))
        fig.update_yaxes(autorange="reversed")
        return fig

    def plot_top_drawdowns(self) -> go.Figure:
        """Top 5 drawdown periods as a Plotly table."""
        df = _top_drawdowns(self.pv, n=5)

        depth_colors = []
        for d in df["Depth (%)"]:
            intensity = min(abs(d) / 30, 1.0)
            r = int(248 * intensity); g = int(81 * intensity); b = int(73 * intensity)
            depth_colors.append(f"rgba({r},{g},{b},0.35)")

        row_colors = [depth_colors] + [[COLORS["panel"]] * len(df)] * (len(df.columns) - 1)

        fig = go.Figure(data=[go.Table(
            columnwidth=[110, 110, 110, 90, 100, 100],
            header=dict(
                values=[f"<b>{c}</b>" for c in df.columns],
                fill_color=COLORS["border"],
                align="center",
                font=dict(family=FONT, size=11, color=COLORS["text"]),
                line_color=COLORS["bg"],
                height=30,
            ),
            cells=dict(
                values=[df[c].tolist() for c in df.columns],
                fill_color=row_colors,
                align="center",
                font=dict(family=FONT, size=10, color=COLORS["text"]),
                line_color=COLORS["grid"],
                height=26,
                format=[None, None, None, ".2f", None, None],
            ),
        )])
        fig.update_layout(
            title=dict(text="Top 5 Drawdown Periods", font=dict(family=FONT, size=14, color=COLORS["text"]), x=0.0),
            paper_bgcolor=COLORS["bg"],
            margin=dict(l=10, r=10, t=45, b=10),
            height=250,
            font=dict(family=FONT),
        )
        return fig

    def plot_all(self):
        """Render all charts and tables inline (Jupyter / VSCode Jupyter)."""
        m = self.metrics()

        # Header
        cfg = self.config
        print(f"\n{'═'*62}")
        print(f"  PORTFOLIO TEARSHEET")
        print(f"  {m['start_date']}  →  {m['end_date']}  ({m['n_days']} days / {m['n_years']:.1f} yrs)")
        if cfg:
            freq  = cfg.get("rebalance_frequency", "—")
            cash  = cfg.get("cash", "—")
            bench = cfg.get("benchmark", "—")
            print(f"  Rebalance: {freq}  |  Capital: ₹{cash:,}  |  Benchmark: {bench}")
        print(f"{'═'*62}\n")

        # Summary tables
        tables = self.summary()
        for key in ["returns", "drawdown", "distribution", "benchmark", "trades"]:
            if tables.get(key):
                display(tables[key])

        # Charts
        self.plot_returns().show()
        self.plot_drawdown().show()
        self.plot_rolling().show()
        self.plot_distribution().show()
        self.plot_monthly().show()
        self.plot_top_drawdowns().show()