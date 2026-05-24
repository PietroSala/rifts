"""Dash dashboard: per-dataset model stats + EU histogram across time-points.

Reads metrics/baseline.csv, metrics/reference.csv, metrics/model_stats.json.
Select a dataset from the dropdown to see:
  - tuned-forest summary (best params, n_test, leaves, unused features, EU stats)
  - bar chart of |EU(i)| for i = 0..length-1 (one bar per time-point)
  - missing features (|EU(i)| = 0) highlighted with red markers above the axis

Run:
    python scripts/dashboard.py
Then open http://127.0.0.1:8050 in a browser.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from dash import Dash, Input, Output, State, ALL, ctx, dcc, html, no_update

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import build_hasse  # noqa: E402

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
BASELINE = REPO_ROOT / "metrics" / "baseline.csv"
REFERENCE = REPO_ROOT / "metrics" / "reference.csv"
MODEL_STATS = REPO_ROOT / "metrics" / "model_stats.json"

MAXIAXP_DIR = REPO_ROOT / "max-iaxp"
GREEDY_DB = REPO_ROOT / "sweeps" / "maximal_reasons" / "sweep.db"
REFINE_DB = REPO_ROOT / "sweeps" / "refinements" / "sweep.db"

# Per-rank colours, must match render_comparison.COLORS.
RANK_COLORS = ["#2ecc71", "#a3e4b3", "#f1c40f", "#e67e22", "#e74c3c"]


def _load_all() -> tuple[dict, dict, dict]:
    baseline = {r["dataset"]: r for r in csv.DictReader(BASELINE.open())}
    reference = {r["dataset"]: r for r in csv.DictReader(REFERENCE.open())}
    model_stats = json.loads(MODEL_STATS.read_text())
    return baseline, reference, model_stats


def _rank_dense(values: list[float]) -> list[int]:
    order = sorted(set(values), reverse=True)
    pos = {v: i + 1 for i, v in enumerate(order)}
    return [pos[v] for v in values]


def _ours_rank(name: str, baseline: dict, reference: dict) -> int | None:
    r = baseline.get(name)
    ref = reference.get(name)
    if not r or not ref or not ref["acc_tsf"] or not ref["acc_1nn_dtw"]:
        return None
    ours = float(r["test_acc"])
    tsf = float(ref["acc_tsf"])
    dtw = float(ref["acc_1nn_dtw"])
    return _rank_dense([ours, tsf, 0.95 * tsf, dtw, 0.95 * dtw])[0]


BASELINE_TBL, REFERENCE_TBL, MODEL_STATS_TBL = _load_all()
DATASETS = sorted(MODEL_STATS_TBL.keys())


def _compute_ours_rank_table() -> dict[str, int]:
    """For each dataset, ours-rank within the 5-cell row
    {ours, tsf, 0.95·tsf, dtw, 0.95·dtw}. Missing datasets get 0."""
    out: dict[str, int] = {}
    for name in DATASETS:
        b = BASELINE_TBL.get(name)
        ref = REFERENCE_TBL.get(name)
        if not b or not ref or not ref.get("acc_tsf") or not ref.get("acc_1nn_dtw"):
            continue
        ours = float(b["test_acc"])
        tsf = float(ref["acc_tsf"])
        dtw = float(ref["acc_1nn_dtw"])
        values = [ours, tsf, 0.95 * tsf, dtw, 0.95 * dtw]
        order = sorted(set(values), reverse=True)
        pos = {v: i + 1 for i, v in enumerate(order)}
        out[name] = pos[ours]
    return out


OURS_RANK_BY_DS: dict[str, int] = _compute_ours_rank_table()
TITLE_COLOR_BY_DS: dict[str, str] = {
    name: RANK_COLORS[r - 1] for name, r in OURS_RANK_BY_DS.items() if 1 <= r <= 5
}
RANK_LABEL = {1: "1st 🟢", 2: "2nd 🟩", 3: "3rd 🟡", 4: "4th 🟠", 5: "5th 🔴"}


def _rank_by(values: dict[str, float], ascending: bool = False) -> dict[str, int]:
    order = sorted(set(values.values()), reverse=not ascending)
    pos = {v: i + 1 for i, v in enumerate(order)}
    return {k: pos[v] for k, v in values.items()}


RANK_GETTERS: dict[str, tuple] = {
    "used":     (lambda n: int(MODEL_STATS_TBL[n]["length"]) - int(MODEL_STATS_TBL[n]["n_unused_features"]),
                                                                                          False),
    "leaves":   (lambda n: int(MODEL_STATS_TBL[n]["total_leaves"]),                       False),
    "eu_mean":  (lambda n: float(MODEL_STATS_TBL[n]["eu_stats"]["mean"]),                 False),
    "eu_min":   (lambda n: int(MODEL_STATS_TBL[n]["eu_stats"]["min"]),                    False),
    "eu_max":   (lambda n: int(MODEL_STATS_TBL[n]["eu_stats"]["max"]),                    False),
    "eu_std":   (lambda n: float(MODEL_STATS_TBL[n]["eu_stats"]["std"]),                  True),
    "lpt_mean": (lambda n: float(MODEL_STATS_TBL[n]["leaves_per_tree_stats"]["mean"]),    False),
    "lpt_min":  (lambda n: int(MODEL_STATS_TBL[n]["leaves_per_tree_stats"]["min"]),       False),
    "lpt_max":  (lambda n: int(MODEL_STATS_TBL[n]["leaves_per_tree_stats"]["max"]),       False),
    "lpt_std":  (lambda n: float(MODEL_STATS_TBL[n]["leaves_per_tree_stats"]["std"]),     True),
}
RANKS = {
    col: _rank_by({n: get(n) for n in DATASETS}, ascending=asc)
    for col, (get, asc) in RANK_GETTERS.items()
}
N_DS = len(DATASETS)


app = Dash(__name__)
app.title = "RIfTS baseline dashboard"

# ----------------------------------------------------------------------------
# Sample explorer: per-sample Max-iAXp vs RIfTS reason side-by-side.
# ----------------------------------------------------------------------------

def _maxiaxp_intervals(name: str, samp: int) -> dict[int, tuple[float, float]] | None:
    p = MAXIAXP_DIR / name / "results.csv"
    if not p.exists():
        return None
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                idx = int(row["sample_idx"])
            except (KeyError, ValueError):
                continue
            if idx != samp:
                continue
            if row.get("solver_status") != "ok":
                return None
            iv = json.loads(row["intervals_json"])
            out: dict[int, tuple[float, float]] = {}
            for k, v in iv.items():
                lo, _lo_open, hi, _hi_open = v
                lo_val = float(lo) if lo is not None else -np.inf
                hi_val = float(hi) if hi is not None else  np.inf
                out[int(k)] = (lo_val, hi_val)
            return out
    return None


def _n_eu_total(name: str) -> int:
    ms = MODEL_STATS_TBL.get(name)
    if not ms:
        return 0
    eu = ms.get("eu_per_feature") or []
    return int(sum(int(e) for e in eu) + len(eu))


def _rifts_sources(name: str, samp: int) -> list[dict]:
    """List of available RIfTS reason sources for (dataset, sample), each
    a dict {value, label, rho, cov}. Includes greedy (when present) and
    only those refinements that strictly improved on their starting rho.
    Ordering: greedy first, then improving refinements by refinement_idx."""
    out: list[dict] = []
    n_eu = _n_eu_total(name)

    if GREEDY_DB.exists():
        conn = sqlite3.connect(GREEDY_DB)
        row = conn.execute(
            "SELECT reason_pos_json FROM reasons "
            "WHERE dataset=? AND sample=?",
            (name, samp),
        ).fetchone()
        conn.close()
        if row is not None:
            pos = json.loads(row[0])
            rho = sum(int(p[1]) - int(p[0]) for p in pos.values())
            out.append({"value": "greedy", "_base": "greedy", "rho": rho})

    if REFINE_DB.exists():
        conn = sqlite3.connect(REFINE_DB)
        rows = conn.execute(
            "SELECT refinement_idx, found_rho, improvement, is_final_max, "
            "       certified_maximum "
            "FROM refinements WHERE dataset=? AND sample=? "
            "ORDER BY refinement_idx ASC",
            (name, samp),
        ).fetchall()
        conn.close()
        for idx, rho, imp, is_final, certified in rows:
            if not imp:
                # Skip refinements that did not improve over their start.
                continue
            tags = []
            if is_final:
                tags.append("best")
            if certified:
                tags.append("certified")
            out.append({
                "value": f"refinement:{idx}",
                "_base": f"refinement #{idx}",
                "rho": int(rho),
                "_tags": tags,
            })

    for s in out:
        cov = s["rho"] / n_eu if n_eu > 0 else None
        s["cov"] = cov
        tag_str = ""
        if s.get("_tags"):
            tag_str = f"  ({', '.join(s['_tags'])})"
        cov_str = f"cov={cov:.3f}" if cov is not None else f"ρ={s['rho']}"
        s["label"] = f"{s['_base']} ({cov_str}){tag_str}"
    return out


def _rifts_default_source(sources: list[dict]) -> str | None:
    if not sources:
        return None
    refinements = [s for s in sources if s["value"].startswith("refinement:")]
    if refinements:
        return max(refinements, key=lambda s: s["rho"])["value"]
    return sources[0]["value"]


def _rifts_intervals(name: str, samp: int,
                     source: str) -> tuple[dict[int, tuple[float, float]], str] | None:
    """Return (intervals, source-label) for the requested RIfTS source.
    `source` is either 'greedy' or 'refinement:<idx>'."""
    if source == "greedy" and GREEDY_DB.exists():
        conn = sqlite3.connect(GREEDY_DB)
        row = conn.execute(
            "SELECT reason_threshold_json FROM reasons "
            "WHERE dataset=? AND sample=?",
            (name, samp),
        ).fetchone()
        conn.close()
        if row is not None:
            return _thresh_to_intervals(json.loads(row[0])), "greedy"
    if source.startswith("refinement:") and REFINE_DB.exists():
        idx = int(source.split(":", 1)[1])
        conn = sqlite3.connect(REFINE_DB)
        row = conn.execute(
            "SELECT reason_threshold_json FROM refinements "
            "WHERE dataset=? AND sample=? AND refinement_idx=?",
            (name, samp, idx),
        ).fetchone()
        conn.close()
        if row is not None:
            return _thresh_to_intervals(json.loads(row[0])), f"refinement #{idx}"
    return None


def _thresh_to_intervals(th: dict) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for k, v in th.items():
        b, e = v
        b_val = float(b) if b is not None else -np.inf
        e_val = float(e) if e is not None else  np.inf
        if b_val != -np.inf or e_val != np.inf:
            out[int(k)] = (b_val, e_val)
    return out


def _sample_series(name: str, samp: int) -> tuple[np.ndarray, int] | None:
    p = MAXIAXP_DIR / name / "samples.parquet"
    if not p.exists():
        return None
    ds = pd.read_parquet(p)
    n_features = sum(1 for c in ds.columns if c.startswith("x_"))
    rows = ds[ds["sample_idx"] == samp]
    if rows.empty:
        if samp < 0 or samp >= len(ds):
            return None
        rows = ds.iloc[[samp]]
    ts = np.asarray(
        rows.iloc[0][[f"x_{i}" for i in range(n_features)]].tolist(),
        dtype=float,
    )
    return ts, n_features


def _samples_for_dataset(name: str) -> list[int]:
    samples: set[int] = set()
    # samples solved by Max-iAXp
    p = MAXIAXP_DIR / name / "results.csv"
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("solver_status") == "ok":
                    try:
                        samples.add(int(row["sample_idx"]))
                    except (KeyError, ValueError):
                        pass
    # samples covered by our greedy / refinement pipelines
    for db in (GREEDY_DB, REFINE_DB):
        if not db.exists():
            continue
        conn = sqlite3.connect(db)
        tbl = "reasons" if db == GREEDY_DB else "refinement_chain_summary"
        rows = conn.execute(
            f"SELECT DISTINCT sample FROM {tbl} WHERE dataset=?", (name,)
        ).fetchall()
        conn.close()
        for (s,) in rows:
            samples.add(int(s))
    return sorted(samples)


def _add_intervals_to_panel(fig, row, ts, iv, colour):
    y_min = float(ts.min()) - 0.05 * (float(ts.max()) - float(ts.min()) + 1e-3)
    y_max = float(ts.max()) + 0.05 * (float(ts.max()) - float(ts.min()) + 1e-3)
    for i, (lo, hi) in iv.items():
        l = y_min if not np.isfinite(lo) else max(lo, y_min)
        h = y_max if not np.isfinite(hi) else min(hi, y_max)
        fig.add_shape(
            type="rect",
            x0=i - 0.45, x1=i + 0.45, y0=l, y1=h,
            fillcolor=colour, opacity=0.35, line_width=0,
            row=row, col=1,
        )


def _maxiaxp_coverage(name: str, samp: int) -> float | None:
    p = MAXIAXP_DIR / name / "results.csv"
    if not p.exists():
        return None
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                if int(row["sample_idx"]) != samp:
                    continue
            except (KeyError, ValueError):
                continue
            if row.get("solver_status") != "ok":
                return None
            try:
                return float(row["coverage_convB"])
            except (KeyError, ValueError, TypeError):
                return None
    return None


def _build_sample_figure(name: str, samp: int, source: str | None) -> go.Figure:
    sd = _sample_series(name, samp)
    if sd is None:
        return go.Figure(layout={"title": f"{name} — sample {samp}: time series unavailable"})
    ts, F = sd
    mx = _maxiaxp_intervals(name, samp)
    mx_cov = _maxiaxp_coverage(name, samp)
    ours = _rifts_intervals(name, samp, source) if source else None
    ours_cov: float | None = None
    if ours is not None:
        for s in _rifts_sources(name, samp):
            if s["value"] == source:
                ours_cov = s.get("cov")
                break

    def _cov_tag(c: float | None) -> str:
        return f", cov={c:.3f}" if c is not None else ""

    if mx is not None:
        top_title = (f"Max-iAXp explanation ({len(mx)} features"
                     f"{_cov_tag(mx_cov)})")
    else:
        top_title = "Max-iAXp explanation (not solved)"

    if ours is not None:
        bot_title = (f"RIfTS {ours[1]} reason ({len(ours[0])} features"
                     f"{_cov_tag(ours_cov)})")
    else:
        bot_title = "RIfTS reason (none yet)"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.12,
        # Row 1 = RIfTS (sits right below the selectors); row 2 = Max-iAXp.
        subplot_titles=(bot_title, top_title),
    )
    x = np.arange(F)
    for r in (1, 2):
        fig.add_trace(
            go.Scatter(x=x, y=ts, mode="lines+markers",
                       line=dict(color="#222", width=1),
                       marker=dict(color="#222", size=3),
                       hovertemplate="t=%{x}<br>x=%{y:.3f}<extra></extra>",
                       showlegend=False),
            row=r, col=1,
        )
    if ours is not None:
        _add_intervals_to_panel(fig, 1, ts, ours[0], "#fb8500")
    if mx is not None:
        _add_intervals_to_panel(fig, 2, ts, mx, "#3a86ff")
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_yaxes(title_text="value", row=2, col=1)
    fig.update_xaxes(title_text="time-point index", row=2, col=1)

    def _fmt(c: float | None) -> str:
        return f"{c:.3f}" if c is not None else "n/a"

    fig.update_layout(
        title=(f"{name} — sample {samp}  ·  "
               f"RIfTS cov={_fmt(ours_cov)}  vs  Max-iAXp cov={_fmt(mx_cov)}"),
        template="plotly_white",
        height=520,
        margin=dict(l=50, r=20, t=70, b=50),
        showlegend=False,
    )
    return fig


_dataset_tab = html.Div([
    html.Div(
        [
            html.Label("dataset"),
            dcc.Dropdown(
                id="dataset",
                options=[{"label": d, "value": d} for d in DATASETS],
                value=DATASETS[0] if DATASETS else None,
                clearable=False,
                style={"width": "320px"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "12px",
               "marginBottom": "12px", "marginTop": "12px"},
    ),
    html.Div(
        [
            "Numbers in parentheses are across-dataset ranks. ",
            html.B("(1) = most demanding for our pipeline"),
            ". For most columns that means the highest value; for the two σ ",
            "columns it is the ",
            html.I("lowest"),
            " σ — uniform medium-rich features compound multiplicatively and ",
            "are harder for our pipeline.",
        ],
        style={"fontSize": "12px", "color": "#34495e", "marginBottom": "12px"},
    ),
    html.Div(id="info-panel", style={"marginBottom": "16px"}),
    dcc.Graph(id="eu-histogram"),
    html.Div(
        [
            html.Label("sample", style={"marginRight": "8px"}),
            dcc.Dropdown(
                id="sample-pick", clearable=True,
                placeholder="(select a sample with available reasons)",
                style={"width": "200px"},
            ),
            html.Label("RIfTS source", style={"marginLeft": "16px",
                                              "marginRight": "8px"}),
            dcc.Dropdown(
                id="rifts-source", clearable=False,
                placeholder="(select a sample first)",
                style={"width": "320px"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "8px",
               "marginTop": "16px", "marginBottom": "4px",
               "flexWrap": "wrap"},
    ),
    dcc.Graph(id="sample-explorer"),
])


_axis_options = [
    {"label": f"  {spec['label']}  ({key})", "value": key}
    for key, spec in build_hasse.AXES.items()
]

_AXIS_MODE_OPTIONS = [
    {"label": "dom+viz", "value": "both"},
    {"label": "dom",     "value": "dom"},
    {"label": "viz",     "value": "viz"},
    {"label": "none",    "value": "none"},
]


def _axis_mode_row(key: str, spec: dict, default: str = "both"):
    return html.Div(
        [
            html.Span(f"{spec['label']} ({key})",
                      style={"fontSize": "12px", "flex": "1",
                             "marginRight": "8px"}),
            dcc.RadioItems(
                id={"type": "hasse-axis-mode", "axis": key},
                options=_AXIS_MODE_OPTIONS,
                value=default,
                inline=True,
                inputStyle={"marginRight": "2px", "marginLeft": "6px"},
                labelStyle={"fontSize": "11px", "marginRight": "0px"},
            ),
        ],
        style={"display": "flex", "alignItems": "center",
               "padding": "3px 4px",
               "borderBottom": "1px solid #f5f6f7"},
    )

_hasse_tab = html.Div([
    html.Div(
        [
            html.Div(
                [
                    html.Label("axis role per complexity dimension",
                               style={"fontSize": "13px", "fontWeight": 600}),
                    html.Div(
                        "dom+viz: drives the partial order and is shown in "
                        "the node label · dom: drives the partial order, "
                        "not shown · viz: shown in the node label, does not "
                        "drive the partial order · none: ignored.",
                        style={"fontSize": "11px", "color": "#7f8c8d",
                               "margin": "4px 0 6px 0"},
                    ),
                    html.Div(
                        [
                            _axis_mode_row(key, spec)
                            for key, spec in build_hasse.AXES.items()
                        ],
                        style={"maxHeight": "320px", "overflowY": "auto",
                               "border": "1px solid #ecf0f1", "padding": "6px",
                               "borderRadius": "6px", "marginBottom": "10px"},
                    ),
                    # Derived stores consumed by every downstream callback.
                    dcc.Store(id="hasse-compare-axes",
                              data=list(build_hasse.AXES.keys())),
                    dcc.Store(id="hasse-visualize-axes", data=[]),
                ],
                style={"flex": "0 0 360px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("layout direction",
                                       style={"fontSize": "13px", "marginRight": "8px"}),
                            dcc.RadioItems(
                                id="hasse-rankdir",
                                options=[
                                    {"label": " TB (top→bottom, default)", "value": "TB"},
                                    {"label": " BT (bottom→top)",          "value": "BT"},
                                    {"label": " LR (left→right)",          "value": "LR"},
                                    {"label": " RL (right→left)",          "value": "RL"},
                                ],
                                value="TB",
                                inline=True,
                                inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                                style={"fontSize": "13px", "marginBottom": "8px"},
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            html.Label("rank scope (for displayed (rank) numbers)",
                                       style={"fontSize": "13px", "marginRight": "8px"}),
                            dcc.RadioItems(
                                id="hasse-rank-scope",
                                options=[
                                    {"label": " all 109 datasets",   "value": "all"},
                                    {"label": " current filter only", "value": "filter"},
                                ],
                                value="all",
                                inline=True,
                                inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                                style={"fontSize": "13px", "marginBottom": "8px"},
                            ),
                        ],
                    ),
                    dcc.Checklist(
                        id="hasse-drop-isolated",
                        options=[{"label": "  drop datasets with no cover edges (cleaner picture)", "value": "drop"}],
                        value=["drop"],
                        style={"fontSize": "13px", "marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("include datasets where RF ranks:",
                                       style={"fontSize": "13px", "marginRight": "8px"}),
                            dcc.Checklist(
                                id="hasse-rank-filter",
                                options=[
                                    {"label": f"  {RANK_LABEL[k]} ",
                                     "value": k}
                                    for k in (1, 2, 3, 4, 5)
                                ],
                                value=[1, 2, 3, 4, 5],
                                inline=True,
                                inputStyle={"marginRight": "3px", "marginLeft": "10px"},
                                style={"fontSize": "13px", "marginBottom": "8px"},
                            ),
                        ],
                    ),
                    html.Div(id="hasse-summary",
                             style={"fontSize": "12px", "color": "#34495e",
                                    "marginBottom": "8px"}),
                    html.Div(
                        "Each node shows the dataset name and the values of the selected axes; the trailing "
                        "number in parentheses is the rank (1 = most demanding). Compare axes (driving "
                        "dominance) appear in bold inside the node; visualize axes appear plain. "
                        "Use the scroll wheel to zoom, drag to pan, or the +/−/⊙ buttons in the corner.",
                        style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Span("download: ",
                                      style={"fontSize": "13px", "marginRight": "6px"}),
                            html.Button("SVG", id="dl-svg",
                                        n_clicks=0,
                                        style={"marginRight": "6px", "fontSize": "13px"}),
                            html.Button("PNG", id="dl-png",
                                        n_clicks=0,
                                        style={"marginRight": "6px", "fontSize": "13px"}),
                            html.Button("DOT", id="dl-dot",
                                        n_clicks=0,
                                        style={"marginRight": "6px", "fontSize": "13px"}),
                            html.Button("CSV (inverse topo)", id="dl-csv",
                                        n_clicks=0,
                                        title="inverse topological order over the filtered set — least demanding first",
                                        style={"fontSize": "13px"}),
                            dcc.Download(id="hasse-download"),
                        ],
                    ),
                ],
                style={"flex": "1 1 auto", "marginLeft": "16px"},
            ),
        ],
        style={"display": "flex", "marginTop": "12px", "marginBottom": "12px"},
    ),
    html.Div(id="hasse-output",
             style={"border": "1px solid #ecf0f1", "borderRadius": "6px",
                    "padding": "0", "background": "#fdfdfd",
                    "minHeight": "780px", "overflow": "hidden"}),
])


_SVG_PAN_ZOOM_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<style>
  html, body { margin:0; padding:0; height:100%; background:#fdfdfd; }
  #wrap { width:100%; height:100%; }
  #wrap > svg { width:100%; height:100%; display:block; }
</style>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
</head>
<body>
<div id="wrap">__SVG__</div>
<script>
window.addEventListener('load', function () {
  var svgEl = document.querySelector('#wrap > svg');
  if (!svgEl) return;
  svgEl.removeAttribute('width');
  svgEl.removeAttribute('height');
  svgEl.style.maxWidth = '100%';
  svgEl.style.maxHeight = '100%';
  svgPanZoom(svgEl, {
    zoomEnabled: true, panEnabled: true,
    controlIconsEnabled: true, fit: true, center: true,
    minZoom: 0.1, maxZoom: 30, zoomScaleSensitivity: 0.4,
  });
});
</script>
</body>
</html>
"""


def _wrap_svg_for_zoom(svg: str) -> str:
    return _SVG_PAN_ZOOM_HTML_TEMPLATE.replace("__SVG__", svg)


app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "padding": "16px",
           "maxWidth": "1200px", "margin": "0 auto"},
    children=[
        html.H2("RIfTS baseline dashboard"),
        html.Div(
            "Note: throughout this dashboard \"RF\" refers to a classical "
            "Random Forest classifier (scikit-learn) trained on the raw "
            "time-point representation of each series.",
            style={"fontSize": "12px", "color": "#7f8c8d",
                   "marginTop": "-6px", "marginBottom": "12px",
                   "fontStyle": "italic"},
        ),
        dcc.Tabs(id="tabs", value="dataset", children=[
            dcc.Tab(label="Dataset explorer", value="dataset", children=[_dataset_tab]),
            dcc.Tab(label="Hasse diagram", value="hasse", children=[_hasse_tab]),
        ]),
    ],
)


def _info_card(name: str) -> html.Div:
    r = BASELINE_TBL.get(name, {})
    ref = REFERENCE_TBL.get(name, {})
    ms = MODEL_STATS_TBL.get(name, {})
    if not ms:
        return html.Div("(no model stats for this dataset)")

    try:
        params = json.loads(r.get("best_params", "{}"))
    except json.JSONDecodeError:
        params = {}
    ours = float(r.get("test_acc") or 0.0)
    tsf = float(ref.get("acc_tsf") or 0.0)
    dtw = float(ref.get("acc_1nn_dtw") or 0.0)
    rank = _ours_rank(name, BASELINE_TBL, REFERENCE_TBL) or 0
    rank_color = RANK_COLORS[rank - 1] if 1 <= rank <= 5 else "#bdc3c7"
    rank_label = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}.get(rank, "—")
    es = ms["eu_stats"]

    def cell(label: str, value: str, color: str | None = None) -> html.Div:
        return html.Div(
            [html.Div(label, style={"fontSize": "11px", "color": "#7f8c8d"}),
             html.Div(value, style={"fontSize": "16px", "fontWeight": 600})],
            style={
                "padding": "8px 12px",
                "border": "1px solid #ecf0f1",
                "borderRadius": "6px",
                "background": color or "#fff",
                "minWidth": "90px",
            },
        )

    def vr(label: str, raw, fmt: str, col: str) -> html.Div:
        rk = RANKS[col].get(name, "")
        return cell(label, f"{fmt.format(raw)}  ({rk}/{N_DS})")

    lpt = ms["leaves_per_tree_stats"]
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))", "gap": "8px"},
        children=[
            cell("RF rank (within row)", rank_label, rank_color),
            cell("RF test acc", f"{ours:.3f}"),
            cell("TSF ref", f"{tsf:.3f}"),
            cell("1NN-DTW ref", f"{dtw:.3f}"),
            cell("length / n_features", str(ms["length"])),
            cell("n_test", str(ms["n_test"])),
            cell("n_classes", str(ms["n_classes"])),
            vr("total leaves", ms["total_leaves"], "{:,}", "leaves"),
            cell("unused features", str(ms["n_unused_features"])),
            vr("used features", ms["length"] - ms["n_unused_features"], "{:d}", "used"),
            vr("EU mean μ (used)", es["mean"], "{:.2f}", "eu_mean"),
            vr("EU min (used)", es["min"], "{:d}", "eu_min"),
            vr("EU max (used)", es["max"], "{:d}", "eu_max"),
            vr("EU std σ (used) — low = demanding", es["std"], "{:.2f}", "eu_std"),
            vr("leaves mean μ / tree", lpt["mean"], "{:.1f}", "lpt_mean"),
            vr("leaves min / tree", lpt["min"], "{:d}", "lpt_min"),
            vr("leaves max / tree", lpt["max"], "{:d}", "lpt_max"),
            vr("leaves std σ / tree — low = demanding", lpt["std"], "{:.2f}", "lpt_std"),
            cell("n_estimators / depth", f"{params.get('n_estimators','?')} / {params.get('max_depth','?')}"),
        ],
    )


@app.callback(Output("info-panel", "children"), Input("dataset", "value"))
def _update_info(name: str) -> html.Div:
    return _info_card(name)


@app.callback(
    Output("sample-pick", "options"),
    Output("sample-pick", "value"),
    Input("dataset", "value"),
)
def _update_sample_options(name: str):
    if not name:
        return [], None
    samples = _samples_for_dataset(name)
    if not samples:
        return [], None
    return [{"label": str(s), "value": s} for s in samples], samples[0]


@app.callback(
    Output("rifts-source", "options"),
    Output("rifts-source", "value"),
    Input("dataset", "value"),
    Input("sample-pick", "value"),
)
def _update_rifts_sources(name: str, samp):
    if not name or samp is None:
        return [], None
    sources = _rifts_sources(name, int(samp))
    if not sources:
        return [], None
    return (
        [{"label": s["label"], "value": s["value"]} for s in sources],
        _rifts_default_source(sources),
    )


@app.callback(
    Output("sample-explorer", "figure"),
    Input("dataset", "value"),
    Input("sample-pick", "value"),
    Input("rifts-source", "value"),
)
def _update_sample_explorer(name: str, samp, source):
    if not name or samp is None:
        return go.Figure()
    return _build_sample_figure(name, int(samp), source)


@app.callback(
    Output("hasse-compare-axes", "data"),
    Output("hasse-visualize-axes", "data"),
    Input({"type": "hasse-axis-mode", "axis": ALL}, "value"),
    State({"type": "hasse-axis-mode", "axis": ALL}, "id"),
)
def _sync_axis_modes(values, ids):
    compare: list[str] = []
    visualize: list[str] = []
    for v, ident in zip(values or [], ids or []):
        key = ident["axis"]
        if v in ("both", "dom"):
            compare.append(key)
        if v in ("both", "viz"):
            visualize.append(key)
    return compare, visualize


@app.callback(Output("eu-histogram", "figure"), Input("dataset", "value"))
def _update_chart(name: str) -> go.Figure:
    ms = MODEL_STATS_TBL.get(name)
    if not ms:
        return go.Figure()
    eu = ms["eu_per_feature"]
    L = len(eu)
    x = list(range(L))
    unused_x = [i for i, v in enumerate(eu) if v == 0]
    fig = go.Figure()
    fig.add_bar(
        x=x, y=eu,
        marker_color=["#bdc3c7" if v == 0 else "#3498db" for v in eu],
        name="|EU(i)|",
        hovertemplate="t=%{x}<br>|EU|=%{y}<extra></extra>",
    )
    if unused_x:
        marker_y = max(eu) * 0.05 if max(eu) > 0 else 1
        fig.add_scatter(
            x=unused_x, y=[marker_y] * len(unused_x),
            mode="markers",
            marker=dict(symbol="x", color="#e74c3c", size=8),
            name=f"unused ({len(unused_x)})",
            hovertemplate="unused t=%{x}<extra></extra>",
        )
    fig.update_layout(
        title=f"{name} — |EU(i)| per time-point",
        xaxis_title="time-point index i",
        yaxis_title="|EU(i)|",
        template="plotly_white",
        height=440,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        bargap=0.05,
    )
    return fig


@app.callback(
    Output("hasse-output", "children"),
    Output("hasse-summary", "children"),
    Input("hasse-compare-axes", "data"),
    Input("hasse-visualize-axes", "data"),
    Input("hasse-drop-isolated", "value"),
    Input("hasse-rankdir", "value"),
    Input("hasse-rank-filter", "value"),
    Input("hasse-rank-scope", "value"),
)
def _update_hasse(compare_axes, visualize_axes, drop_iso, rankdir, rank_filter, rank_scope):
    compare_axes = compare_axes or []
    visualize_axes = visualize_axes or []
    drop = "drop" in (drop_iso or [])
    selected_ranks = set(rank_filter or [])
    rank_scope = rank_scope or "all"
    if not compare_axes:
        return (
            html.Div("Select at least one compare axis on the left.",
                     style={"color": "#7f8c8d", "padding": "12px"}),
            "",
        )
    if not selected_ranks:
        return (
            html.Div("Select at least one ours-rank colour above.",
                     style={"color": "#7f8c8d", "padding": "12px"}),
            "",
        )
    dataset_filter = {
        name for name, r in OURS_RANK_BY_DS.items() if r in selected_ranks
    }
    dot_text, n_kept, n_edges, n_dom = build_hasse.build_dot_text(
        MODEL_STATS_TBL,
        compare_axes=list(compare_axes),
        visualize_axes=list(visualize_axes),
        drop_isolated=drop,
        rankdir=rankdir or "TB",
        title_color_by_ds=TITLE_COLOR_BY_DS,
        dataset_filter=dataset_filter,
        rank_scope=rank_scope,
    )
    svg = build_hasse.render_svg(dot_text)
    rank_summary = ", ".join(RANK_LABEL[k] for k in sorted(selected_ranks))
    n_vis = sum(1 for a in visualize_axes if a not in set(compare_axes))
    summary = (
        f"{len(compare_axes)} compare + {n_vis} visualize axes · "
        f"rankdir={rankdir} · rank scope={rank_scope} · "
        f"filter [{rank_summary}] ({len(dataset_filter)} datasets) · "
        f"{n_dom} dominance pairs · {n_edges} cover edges · "
        f"{n_kept} in diagram"
        + (" (isolated dropped)" if drop else "")
    )
    if svg is None:
        return (
            html.Div("`dot` not on PATH — cannot render SVG. Run "
                     "`scripts/build_hasse.py` and check `metrics/comparison_hasse.dot` "
                     "manually.", style={"color": "#c0392b", "padding": "12px"}),
            summary,
        )
    return (
        html.Iframe(
            srcDoc=_wrap_svg_for_zoom(svg),
            style={"width": "100%", "height": "780px", "border": "none"},
        ),
        summary,
    )


def _build_dot_from_state(compare_axes, visualize_axes, drop_iso, rankdir,
                          rank_filter, rank_scope) -> str | None:
    compare_axes = compare_axes or []
    if not compare_axes:
        return None
    selected_ranks = set(rank_filter or [])
    if not selected_ranks:
        return None
    dataset_filter = {n for n, r in OURS_RANK_BY_DS.items() if r in selected_ranks}
    dot_text, *_ = build_hasse.build_dot_text(
        MODEL_STATS_TBL,
        compare_axes=list(compare_axes),
        visualize_axes=list(visualize_axes or []),
        drop_isolated="drop" in (drop_iso or []),
        rankdir=rankdir or "TB",
        title_color_by_ds=TITLE_COLOR_BY_DS,
        dataset_filter=dataset_filter,
        rank_scope=rank_scope or "all",
    )
    return dot_text


@app.callback(
    Output("hasse-download", "data"),
    Input("dl-svg", "n_clicks"),
    Input("dl-png", "n_clicks"),
    Input("dl-dot", "n_clicks"),
    Input("dl-csv", "n_clicks"),
    State("hasse-compare-axes", "data"),
    State("hasse-visualize-axes", "data"),
    State("hasse-drop-isolated", "value"),
    State("hasse-rankdir", "value"),
    State("hasse-rank-filter", "value"),
    State("hasse-rank-scope", "value"),
    prevent_initial_call=True,
)
def _download(_svg_clicks, _png_clicks, _dot_clicks, _csv_clicks,
              compare_axes, visualize_axes, drop_iso, rankdir,
              rank_filter, rank_scope):
    trigger = ctx.triggered_id
    if trigger is None:
        return no_update
    compare_axes = compare_axes or []
    visualize_axes = visualize_axes or []
    selected_ranks = set(rank_filter or [])
    if not compare_axes or not selected_ranks:
        return no_update
    dataset_filter = {n for n, r in OURS_RANK_BY_DS.items() if r in selected_ranks}

    if trigger == "dl-csv":
        csv_text = build_hasse.topological_csv(
            MODEL_STATS_TBL,
            compare_axes=list(compare_axes),
            visualize_axes=list(visualize_axes),
            dataset_filter=dataset_filter,
            ours_rank_by_ds=OURS_RANK_BY_DS,
            rank_scope=rank_scope or "all",
        )
        return dict(content=csv_text, filename="hasse_inverse_topo.csv",
                    type="text/csv")

    dot_text = _build_dot_from_state(
        compare_axes, visualize_axes, drop_iso, rankdir, rank_filter, rank_scope,
    )
    if dot_text is None:
        return no_update
    if trigger == "dl-dot":
        return dict(content=dot_text, filename="hasse.dot")
    if trigger == "dl-svg":
        svg = build_hasse.render_svg(dot_text)
        if svg is None:
            return no_update
        return dict(content=svg, filename="hasse.svg",
                    type="image/svg+xml")
    if trigger == "dl-png":
        import subprocess
        import shutil
        dot_bin = shutil.which("dot")
        if dot_bin is None:
            return no_update
        proc = subprocess.run(
            [dot_bin, "-Tpng"], input=dot_text.encode(), capture_output=True,
        )
        if proc.returncode != 0:
            return no_update
        return dict(content=base64.b64encode(proc.stdout).decode(),
                    filename="hasse.png",
                    type="image/png",
                    base64=True)
    return no_update


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RIfTS baseline dashboard")
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface to bind (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8050,
                    help="TCP port to bind (default: 8050; use a non-default "
                         "value when running alongside the live dashboard)")
    ap.add_argument("--debug", action="store_true",
                    help="enable Dash debug mode")
    args = ap.parse_args()
    print(f"serving on http://{args.host}:{args.port}")
    app.run(debug=args.debug, host=args.host, port=args.port)
