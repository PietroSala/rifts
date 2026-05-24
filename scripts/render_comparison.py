"""Render a colour-coded Markdown comparison of ours / TSF / 1NN-DTW.

For each row the five numeric cells (ours, tsf, 0.95·tsf, 1nn-dtw, 0.95·dtw)
are ranked from highest (1st) to lowest (5th); the rank determines the
background colour. Ties share the better rank. Two extra columns are appended
for context: series length and the best hyperparameters of our tuned forest
(the one that produced the reported test accuracy).

Also writes a summary of how the `ours` column ranks across all datasets
(absolute and cumulative count per rank) plus an interactive Plotly chart
saved as metrics/comparison_ours.html.

Writes metrics/comparison.md (+ comparison_ours.html).
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
BASELINE = REPO_ROOT / "metrics" / "baseline.csv"
REFERENCE = REPO_ROOT / "metrics" / "reference.csv"
MODEL_STATS = REPO_ROOT / "metrics" / "model_stats.json"
OUT = REPO_ROOT / "metrics" / "comparison.md"

# Per-rank background colours, 1st through 5th.
COLORS = ["#2ecc71", "#a3e4b3", "#f1c40f", "#e67e22", "#e74c3c"]

NUMERIC_COLS = ["ours", "tsf", "0.95·tsf", "1nn-dtw", "0.95·dtw"]


def rank_dense(values: list[float]) -> list[int]:
    """Return 1-based rank per position, highest = 1, ties share the better rank."""
    order = sorted(set(values), reverse=True)
    pos = {v: i + 1 for i, v in enumerate(order)}
    return [pos[v] for v in values]


def cell(value: float, rank: int) -> str:
    color = COLORS[min(rank - 1, len(COLORS) - 1)]
    return f'<td style="background-color:{color};text-align:right;padding:2px 8px">{value:.3f}</td>'


def format_params(raw: str) -> str:
    if not raw:
        return ""
    try:
        p = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    cw = p.get("class_weight")
    cw = "bal" if cw == "balanced" else ("-" if cw in (None, "None", "null") else str(cw))
    mf = p.get("max_features")
    mf = f"{mf:g}" if isinstance(mf, (int, float)) else str(mf)
    return (
        f"n={p.get('n_estimators','?')}, "
        f"d={p.get('max_depth','?')}, "
        f"leaf={p.get('min_samples_leaf','?')}, "
        f"mf={mf}, "
        f"cw={cw}"
    )


def render_summary(ours_ranks: list[int]) -> tuple[str, list[int], list[int]]:
    """Build the rank-summary table for the `ours` column and the count vectors."""
    counts = [sum(1 for r in ours_ranks if r == k) for k in range(1, 6)]
    cumulative = [sum(counts[:k]) for k in range(1, 6)]
    total = sum(counts) or 1
    rank_labels = ["1st", "2nd", "3rd", "4th", "5th"]

    out: list[str] = []
    out.append(f"\n## Where `ours` ranks (across {total} compared datasets)\n")
    out.append("<table>")
    out.append(
        '<thead><tr><th>rank</th><th style="text-align:right">count</th>'
        '<th style="text-align:right">share</th>'
        '<th style="text-align:right">cumulative</th>'
        '<th style="text-align:right">cumulative share</th></tr></thead>'
    )
    out.append("<tbody>")
    for k in range(5):
        color = COLORS[k]
        share = counts[k] / total
        cum_share = cumulative[k] / total
        out.append(
            f'<tr>'
            f'<td style="background-color:{color};padding:2px 8px">{rank_labels[k]}</td>'
            f'<td style="text-align:right">{counts[k]}</td>'
            f'<td style="text-align:right">{share:.1%}</td>'
            f'<td style="text-align:right">{cumulative[k]}</td>'
            f'<td style="text-align:right">{cum_share:.1%}</td>'
            f'</tr>'
        )
    out.append("</tbody></table>")
    return "\n".join(out), counts, cumulative


def render_chart(counts: list[int], cumulative: list[int], out_path: Path) -> None:
    rank_labels = ["1st", "2nd", "3rd", "4th", "5th"]
    fig = go.Figure()
    fig.add_bar(
        x=rank_labels, y=counts, marker_color=COLORS,
        text=counts, textposition="outside", name="per rank",
    )
    fig.add_scatter(
        x=rank_labels, y=cumulative, mode="lines+markers+text",
        text=cumulative, textposition="top center",
        line=dict(color="#34495e", width=2),
        marker=dict(size=8, color="#34495e"),
        name="cumulative",
        yaxis="y2",
    )
    fig.update_layout(
        title="Rank of `ours` across compared datasets",
        xaxis_title="rank",
        yaxis=dict(title="count per rank"),
        yaxis2=dict(title="cumulative", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        bargap=0.3,
        template="plotly_white",
        height=420,
    )
    pio.write_html(fig, file=str(out_path), include_plotlyjs="cdn", full_html=True)


def _rank_by(values: dict[str, float], ascending: bool = False) -> dict[str, int]:
    """Map dataset -> rank. By default descending (1 = highest); set
    ascending=True to reverse (1 = lowest). Ties share the better rank."""
    order = sorted(set(values.values()), reverse=not ascending)
    pos = {v: i + 1 for i, v in enumerate(order)}
    return {k: pos[v] for k, v in values.items()}


def _val_with_rank(value, rank, fmt: str = "{:g}") -> str:
    return f"{fmt.format(value)} <small style=\"color:#7f8c8d\">({rank})</small>"


def main() -> None:
    ref = {r["dataset"]: r for r in csv.DictReader(REFERENCE.open())}
    rows = list(csv.DictReader(BASELINE.open()))
    model_stats = json.loads(MODEL_STATS.read_text()) if MODEL_STATS.exists() else {}

    # Per-column ranks across datasets. Rank 1 is the most "demanding" value
    # for our pipeline:
    #   - default: descending (higher value = harder = rank 1).
    #   - σ columns: ascending (lower σ = more uniform medium-rich features =
    #     compound multiplicatively = harder for us = rank 1).
    # length / n_test / unused carry no rank — they are dataset descriptors
    # rather than model-difficulty signals. The model-difficulty axis on the
    # feature count is `used = length - unused` and is ranked instead.
    rank_specs = {
        "n_trees":  (lambda n: int(model_stats[n]["n_trees"]),                           False),
        "used":     (lambda n: int(model_stats[n]["length"]) - int(model_stats[n]["n_unused_features"]),
                                                                                         False),
        "leaves":   (lambda n: int(model_stats[n]["total_leaves"]),                      False),
        "eu_mean":  (lambda n: float(model_stats[n]["eu_stats"]["mean"]),                False),
        "eu_min":   (lambda n: int(model_stats[n]["eu_stats"]["min"]),                   False),
        "eu_max":   (lambda n: int(model_stats[n]["eu_stats"]["max"]),                   False),
        "eu_std":   (lambda n: float(model_stats[n]["eu_stats"]["std"]),                 True),
        "lpt_mean": (lambda n: float(model_stats[n]["leaves_per_tree_stats"]["mean"]),   False),
        "lpt_min":  (lambda n: int(model_stats[n]["leaves_per_tree_stats"]["min"]),      False),
        "lpt_max":  (lambda n: int(model_stats[n]["leaves_per_tree_stats"]["max"]),      False),
        "lpt_std":  (lambda n: float(model_stats[n]["leaves_per_tree_stats"]["std"]),    True),
    }
    ranks: dict[str, dict[str, int]] = {}
    for col, (get, ascending) in rank_specs.items():
        vals = {n: get(n) for n in model_stats}
        ranks[col] = _rank_by(vals, ascending=ascending)

    lines: list[str] = []
    lines.append("# Baseline RF vs published references (TSF, 1NN-DTW)\n")
    lines.append(
        "Per row the five accuracy cells are ranked **within the row** (green = 1st, "
        "light-green = 2nd, yellow = 3rd, orange = 4th, red = 5th). All other numeric "
        "cells carry an **across-dataset rank in parentheses**: `(1)` is the dataset on "
        "which the value is **most demanding for our pipeline**. By default this means "
        "the *highest* value (more features, more leaves, larger EU). "
        "**Exception:** for the two σ columns (EU&nbsp;σ and leaves&nbsp;σ) `(1)` is "
        "the *lowest* σ — uniform medium-rich features compound multiplicatively and "
        "are harder for our pipeline, so low σ is the demanding regime.\n"
    )
    lines.append("<table>")
    lines.append("<thead><tr>")
    lines.append('<th style="text-align:left">dataset</th>')
    lines.append('<th style="text-align:right">length</th>')
    lines.append('<th style="text-align:right">n_test</th>')
    for c in NUMERIC_COLS:
        lines.append(f'<th style="text-align:right">{c}</th>')
    lines.append('<th style="text-align:right" title="number of trees in the tuned forest">n_trees</th>')
    lines.append('<th style="text-align:right">leaves</th>')
    lines.append('<th style="text-align:right">unused</th>')
    lines.append('<th style="text-align:right" title="length - unused = features used by at least one tree">used</th>')
    lines.append('<th style="text-align:right" title="mean over used features">EU&nbsp;mean&nbsp;μ<sub>(used)</sub></th>')
    lines.append('<th style="text-align:right" title="min over used features">EU&nbsp;min<sub>(used)</sub></th>')
    lines.append('<th style="text-align:right" title="max over used features">EU&nbsp;max<sub>(used)</sub></th>')
    lines.append('<th style="text-align:right" title="std over used features">EU&nbsp;std&nbsp;σ<sub>(used)</sub></th>')
    lines.append('<th style="text-align:right" title="mean leaves per tree">leaves&nbsp;mean&nbsp;μ</th>')
    lines.append('<th style="text-align:right" title="min leaves per tree">leaves&nbsp;min</th>')
    lines.append('<th style="text-align:right" title="max leaves per tree">leaves&nbsp;max</th>')
    lines.append('<th style="text-align:right" title="std leaves per tree">leaves&nbsp;std&nbsp;σ</th>')
    lines.append('<th style="text-align:left">best params (ours)</th>')
    lines.append("</tr></thead>")
    lines.append("<tbody>")

    ours_ranks: list[int] = []
    n_rendered = 0
    for r in rows:
        name = r["dataset"]
        ours = float(r["test_acc"])
        if name not in ref:
            continue
        tsf = float(ref[name]["acc_tsf"]) if ref[name]["acc_tsf"] else None
        dtw = float(ref[name]["acc_1nn_dtw"]) if ref[name]["acc_1nn_dtw"] else None
        if tsf is None or dtw is None:
            continue
        values = [ours, tsf, 0.95 * tsf, dtw, 0.95 * dtw]
        row_ranks = rank_dense(values)
        ours_ranks.append(row_ranks[0])
        cells = "".join(cell(v, k) for v, k in zip(values, row_ranks))
        params = format_params(r.get("best_params", ""))
        ms = model_stats.get(name)
        if ms:
            es = ms["eu_stats"]
            lpt = ms["leaves_per_tree_stats"]

            def vr(col_key: str, raw_val, fmt: str = "{:g}") -> str:
                rk = ranks[col_key].get(name, "")
                return f'<td style="text-align:right">{_val_with_rank(raw_val, rk, fmt)}</td>'

            used = ms["length"] - ms["n_unused_features"]
            length_cell = f'<td style="text-align:right">{ms["length"]}</td>'
            n_test_cell = f'<td style="text-align:right">{ms["n_test"]}</td>'
            eu_cells = (
                vr("n_trees", ms["n_trees"])
                + vr("leaves", ms["total_leaves"])
                + f'<td style="text-align:right">{ms["n_unused_features"]}</td>'
                + vr("used", used, "{:d}")
                + vr("eu_mean", es["mean"], "{:.2f}")
                + vr("eu_min", es["min"])
                + vr("eu_max", es["max"])
                + vr("eu_std", es["std"], "{:.2f}")
                + vr("lpt_mean", lpt["mean"], "{:.1f}")
                + vr("lpt_min", lpt["min"])
                + vr("lpt_max", lpt["max"])
                + vr("lpt_std", lpt["std"], "{:.2f}")
            )
        else:
            length_cell = f'<td style="text-align:right">{r.get("length","")}</td>'
            n_test_cell = f'<td style="text-align:right">{r.get("n_test","")}</td>'
            eu_cells = '<td style="text-align:right">-</td>' * 12
        lines.append(
            f"<tr><td>{name}</td>"
            f"{length_cell}"
            f"{n_test_cell}"
            f"{cells}"
            f"{eu_cells}"
            f"<td><code>{params}</code></td></tr>"
        )
        n_rendered += 1

    lines.append("</tbody>")
    lines.append("</table>")

    summary, counts, cumulative = render_summary(ours_ranks)
    lines.append(summary)

    chart_path = OUT.parent / "comparison_ours.html"
    render_chart(counts, cumulative, chart_path)
    lines.append(
        f"\nInteractive Plotly chart: [`{chart_path.name}`]({chart_path.name})\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({n_rendered} rows)")
    print(f"wrote {chart_path}")


if __name__ == "__main__":
    main()
