"""Hasse diagram of the dataset partial order over selectable demanding-ness axes.

A dataset A "dominates" B iff on every chosen axis A is at least as demanding
as B (rank(A) ≤ rank(B), since rank 1 = most demanding by construction), and
strictly more demanding on at least one axis. The Hasse diagram is the
transitive reduction (cover relation), with the most demanding datasets at
the top (rankdir = BT).

Axes available (matching render_comparison.py / dashboard.py):
  used, leaves, eu_mean, eu_min, eu_max, eu_std, lpt_mean, lpt_min,
  lpt_max, lpt_std. The two σ axes are ascending (low σ = more demanding);
  everything else is descending.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
MODEL_STATS = REPO_ROOT / "metrics" / "model_stats.json"
OUT_DOT = REPO_ROOT / "metrics" / "comparison_hasse.dot"
OUT_PNG = REPO_ROOT / "metrics" / "comparison_hasse.png"


def _used(s):    return int(s["length"]) - int(s["n_unused_features"])
def _leaves(s):  return int(s["total_leaves"])
def _ntrees(s):  return int(s["n_trees"])
def _eum(s):     return float(s["eu_stats"]["mean"])
def _eumn(s):    return int(s["eu_stats"]["min"])
def _eumx(s):    return int(s["eu_stats"]["max"])
def _eustd(s):   return float(s["eu_stats"]["std"])
def _lptm(s):    return float(s["leaves_per_tree_stats"]["mean"])
def _lptmn(s):   return int(s["leaves_per_tree_stats"]["min"])
def _lptmx(s):   return int(s["leaves_per_tree_stats"]["max"])
def _lptstd(s):  return float(s["leaves_per_tree_stats"]["std"])

AXES: dict[str, dict] = {
    "n_trees":  {"label": "number of trees",       "getter": _ntrees,  "ascending": False, "fmt": "{:d}"},
    "used":     {"label": "used (length−unused)",  "getter": _used,    "ascending": False, "fmt": "{:d}"},
    "leaves":   {"label": "total leaves",          "getter": _leaves,  "ascending": False, "fmt": "{:d}"},
    "eu_mean":  {"label": "EU mean μ",             "getter": _eum,     "ascending": False, "fmt": "{:.2f}"},
    "eu_min":   {"label": "EU min",                "getter": _eumn,    "ascending": False, "fmt": "{:d}"},
    "eu_max":   {"label": "EU max",                "getter": _eumx,    "ascending": False, "fmt": "{:d}"},
    "eu_std":   {"label": "EU std σ (low=harder)", "getter": _eustd,   "ascending": True,  "fmt": "{:.2f}"},
    "lpt_mean": {"label": "leaves μ / tree",       "getter": _lptm,    "ascending": False, "fmt": "{:.1f}"},
    "lpt_min":  {"label": "leaves min / tree",     "getter": _lptmn,   "ascending": False, "fmt": "{:d}"},
    "lpt_max":  {"label": "leaves max / tree",     "getter": _lptmx,   "ascending": False, "fmt": "{:d}"},
    "lpt_std":  {"label": "leaves σ / tree (low=harder)", "getter": _lptstd, "ascending": True, "fmt": "{:.2f}"},
}


def _rank_by(values: dict[str, float], ascending: bool) -> dict[str, int]:
    order = sorted(set(values.values()), reverse=not ascending)
    pos = {v: i + 1 for i, v in enumerate(order)}
    return {k: pos[v] for k, v in values.items()}


def build_rank_table(
    model_stats: dict[str, dict], axis_keys: Iterable[str]
) -> tuple[list[str], list[list[int]], list[list]]:
    """Returns (datasets, rank_matrix, raw_value_matrix).

    rank_matrix[axis_i][ds_i] = 1-based rank, 1 = most demanding.
    raw_value_matrix[axis_i][ds_i] = the raw axis value (for node labels).
    """
    datasets = sorted(model_stats.keys())
    rank_matrix: list[list[int]] = []
    raw_matrix: list[list] = []
    for key in axis_keys:
        spec = AXES[key]
        vals = {n: spec["getter"](model_stats[n]) for n in datasets}
        r = _rank_by(vals, ascending=spec["ascending"])
        rank_matrix.append([r[n] for n in datasets])
        raw_matrix.append([vals[n] for n in datasets])
    return datasets, rank_matrix, raw_matrix


def dominance(rank_matrix: list[list[int]]) -> list[set[int]]:
    """For each dataset index a, the set of indices b that a strictly dominates."""
    if not rank_matrix:
        return []
    n = len(rank_matrix[0])
    dom: list[set[int]] = [set() for _ in range(n)]
    cols = list(zip(*rank_matrix))
    for a in range(n):
        ra = cols[a]
        for b in range(n):
            if a == b:
                continue
            rb = cols[b]
            ge = strict = True
            ge_all = True
            any_strict = False
            for x, y in zip(ra, rb):
                if x > y:
                    ge_all = False
                    break
                if x < y:
                    any_strict = True
            if ge_all and any_strict:
                dom[a].add(b)
    return dom


def transitive_reduction(dom: list[set[int]]) -> list[set[int]]:
    """Cover relation: A → B iff A > B and there is no C with A > C > B."""
    n = len(dom)
    cover: list[set[int]] = [set() for _ in range(n)]
    for a in range(n):
        direct = set(dom[a])
        for c in dom[a]:
            direct -= dom[c]
        cover[a] = direct
    return cover


def _html_label(
    name: str,
    rows_data: list[tuple],  # (axis_key, value_str, rank, is_compare)
    title_color: str,
) -> str:
    """Graphviz HTML-like label: a bordered table; only the title row is filled
    with `title_color`. Body rows have no background, just cell borders. Axes
    used for dominance (is_compare=True) are rendered in **bold**; visualize-
    only axes appear plain."""
    rows = []
    for key, val_str, rank, is_compare in rows_data:
        name_html = f"<B>{key}</B>" if is_compare else key
        rows.append(
            f'<TR>'
            f'<TD ALIGN="LEFT"><FONT FACE="Helvetica" POINT-SIZE="9">{name_html}</FONT></TD>'
            f'<TD ALIGN="RIGHT"><FONT FACE="Helvetica" POINT-SIZE="9">'
            f'{val_str} <FONT COLOR="#7f8c8d">({rank})</FONT></FONT></TD>'
            f'</TR>'
        )
    rows_html = "".join(rows)
    return (
        f'<<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="2" '
        f'COLOR="#bdc3c7">'
        f'<TR><TD COLSPAN="2" ALIGN="CENTER" BGCOLOR="{title_color}">'
        f'<FONT FACE="Helvetica" POINT-SIZE="11"><B>{name}</B></FONT>'
        f'</TD></TR>'
        f'{rows_html}'
        f'</TABLE>>'
    )


DEFAULT_TITLE_COLOR = "#ecf0f1"


def build_dot_text(
    model_stats: dict[str, dict],
    compare_axes: list[str],
    visualize_axes: list[str] | None = None,
    drop_isolated: bool = True,
    rankdir: str = "TB",
    title_color_by_ds: dict[str, str] | None = None,
    dataset_filter: set[str] | None = None,
    rank_scope: str = "all",
) -> tuple[str, int, int, int]:
    """Returns (dot_text, n_kept, n_cover_edges, n_dominance_pairs).

    - `compare_axes`: axes used for dominance / cover (drive the partial order).
    - `visualize_axes`: extra axes shown in node tables but **not** used for
      dominance. Overlaps with `compare_axes` are deduped.
    - `rankdir`: TB / BT / LR / RL.
    - `title_color_by_ds`: per-dataset hex colour for the title row.
    - `dataset_filter`: if given, only those datasets appear in the diagram;
      dominance + cover are recomputed within the subset so cover edges
      always sit between two visible nodes.
    - `rank_scope`: "all" → ranks shown in node labels are computed over the
      full 109 datasets; "filter" → ranks recomputed over the visible subset
      only (useful when the filter is tight and you want within-subset
      positioning).
    """
    if rankdir not in {"TB", "BT", "LR", "RL"}:
        rankdir = "TB"
    if not compare_axes:
        return (
            'digraph hasse { graph [label="select at least one compare axis", '
            'labelloc=t, fontname="Helvetica"]; }', 0, 0, 0,
        )

    visualize_axes = visualize_axes or []
    visualize_axes = [a for a in visualize_axes if a not in set(compare_axes)]
    all_display_axes = list(compare_axes) + visualize_axes
    compare_set = set(compare_axes)

    # Dominance is computed strictly over compare_axes, with global ranks.
    datasets, rank_matrix, raw_matrix = build_rank_table(model_stats, compare_axes)
    dom = dominance(rank_matrix)

    n = len(datasets)
    if dataset_filter is None:
        visible = set(range(n))
    else:
        visible = {i for i, name in enumerate(datasets) if name in dataset_filter}
        dom = [
            (d & visible) if i in visible else set()
            for i, d in enumerate(dom)
        ]
    cover = transitive_reduction(dom)
    n_dom_total = sum(len(d) for d in dom)

    incident: set[int] = set()
    for a, succs in enumerate(cover):
        if succs:
            incident.add(a)
        incident.update(succs)
    keep = visible & incident if drop_isolated else visible

    # Ranks for display (compare + visualize) computed over the chosen scope.
    if rank_scope == "filter" and dataset_filter is not None:
        scope_dss = [datasets[i] for i in sorted(visible)]
    else:
        scope_dss = list(datasets)
    scope_set = set(scope_dss)

    display_rank: dict[str, dict[str, int]] = {}
    display_raw:  dict[str, dict[str, object]] = {}
    for axis in all_display_axes:
        spec = AXES[axis]
        vals_all = {nm: spec["getter"](model_stats[nm]) for nm in datasets}
        vals_scope = {nm: vals_all[nm] for nm in scope_dss}
        display_rank[axis] = _rank_by(vals_scope, ascending=spec["ascending"])
        display_raw[axis] = vals_all

    direction_label = {
        "TB": "top → bottom (most demanding at top)",
        "BT": "bottom → top (most demanding at bottom)",
        "LR": "left → right (most demanding on the left)",
        "RL": "right → left (most demanding on the right)",
    }[rankdir]
    scope_label = "all 109" if rank_scope == "all" else f"filter ({len(scope_dss)})"
    title_color_by_ds = title_color_by_ds or {}

    lines: list[str] = []
    lines.append("digraph hasse {")
    lines.append(f"  rankdir={rankdir};")
    lines.append('  graph [fontname="Helvetica", labelloc=t, fontsize=11, '
                 f'label="Hasse — {direction_label}; ranks vs {scope_label}; '
                 f'edges = cover; bold axes drive dominance"];')
    lines.append('  node  [shape=plaintext, fontname="Helvetica"];')
    lines.append('  edge  [arrowsize=0.6, color="#7f8c8d"];')
    n_kept = 0
    for i, name in enumerate(datasets):
        if i not in keep:
            continue
        n_kept += 1
        rows_data = []
        for axis in all_display_axes:
            spec = AXES[axis]
            val = display_raw[axis][name]
            val_str = spec["fmt"].format(val)
            rk = display_rank[axis].get(name, "—")
            rows_data.append((axis, val_str, rk, axis in compare_set))
        title_color = title_color_by_ds.get(name, DEFAULT_TITLE_COLOR)
        label = _html_label(name, rows_data, title_color)
        lines.append(f'  "{name}" [label={label}];')
    n_edges = 0
    for a, succs in enumerate(cover):
        for b in sorted(succs):
            if a not in keep or b not in keep:
                continue
            lines.append(f'  "{datasets[a]}" -> "{datasets[b]}";')
            n_edges += 1
    lines.append("}")
    return "\n".join(lines) + "\n", n_kept, n_edges, n_dom_total


def topological_order_inverse(
    model_stats: dict[str, dict],
    compare_axes: list[str],
    dataset_filter: set[str] | None = None,
) -> list[str]:
    """Inverse topological order over the filtered set: least demanding first.

    For each dominance pair A→B (A dominates B), B comes **before** A in the
    returned list. Within an antichain (mutually incomparable nodes), ties
    are broken by `sum of ranks over compare_axes` — higher sum (less demanding
    overall) appears first; further ties broken by dataset name.
    """
    import heapq

    datasets, rank_matrix, _ = build_rank_table(model_stats, compare_axes)
    n = len(datasets)
    if dataset_filter is None:
        idxs = list(range(n))
    else:
        idxs = [i for i, name in enumerate(datasets) if name in dataset_filter]
    idx_set = set(idxs)

    dom = dominance(rank_matrix)
    dom_f = [(dom[i] & idx_set) if i in idx_set else set() for i in range(n)]

    predecessors: dict[int, set[int]] = {i: set() for i in idxs}
    for p in idxs:
        for b in dom_f[p]:
            predecessors[b].add(p)

    out_deg = {i: len(dom_f[i]) for i in idxs}
    score = {i: sum(rank_matrix[ax][i] for ax in range(len(rank_matrix))) for i in idxs}

    # min-heap on (-score, name, i) — highest score (least demanding) first.
    h = [(-score[i], datasets[i], i) for i in idxs if out_deg[i] == 0]
    heapq.heapify(h)
    order: list[str] = []
    while h:
        _, _, i = heapq.heappop(h)
        order.append(datasets[i])
        for p in predecessors[i]:
            out_deg[p] -= 1
            if out_deg[p] == 0:
                heapq.heappush(h, (-score[p], datasets[p], p))
    return order


def topological_csv(
    model_stats: dict[str, dict],
    compare_axes: list[str],
    visualize_axes: list[str] | None = None,
    dataset_filter: set[str] | None = None,
    ours_rank_by_ds: dict[str, int] | None = None,
    rank_scope: str = "all",
) -> str:
    """Build a CSV (returned as a string) representing the inverse topological
    order over the filtered set. Columns: topo_position, dataset, ours_rank,
    total_rank_sum, then for each compare/visualize axis: value, rank.
    """
    import csv as _csv
    import io as _io

    visualize_axes = visualize_axes or []
    visualize_axes = [a for a in visualize_axes if a not in set(compare_axes)]
    all_axes = list(compare_axes) + visualize_axes

    ordered = topological_order_inverse(model_stats, compare_axes, dataset_filter)

    # Per-axis ranks over the requested scope.
    datasets = sorted(model_stats.keys())
    scope_dss = list(ordered) if rank_scope == "filter" else datasets
    ranks_per_axis: dict[str, dict[str, int]] = {}
    raw_per_axis: dict[str, dict[str, object]] = {}
    for axis in all_axes:
        spec = AXES[axis]
        vals_all = {n: spec["getter"](model_stats[n]) for n in datasets}
        vals_scope = {n: vals_all[n] for n in scope_dss}
        ranks_per_axis[axis] = _rank_by(vals_scope, ascending=spec["ascending"])
        raw_per_axis[axis] = vals_all

    # Sum of compare-axis ranks (over GLOBAL rank_matrix used inside topo).
    _datasets, rank_matrix_compare, _ = build_rank_table(model_stats, compare_axes)
    name_to_idx = {n: i for i, n in enumerate(_datasets)}
    rank_sum = {
        n: sum(rank_matrix_compare[ax][name_to_idx[n]] for ax in range(len(compare_axes)))
        for n in ordered
    }

    buf = _io.StringIO()
    w = _csv.writer(buf)
    header = ["topo_position", "dataset", "ours_rank", "total_rank_sum_compare"]
    for axis in all_axes:
        role = "compare" if axis in set(compare_axes) else "visualize"
        header += [f"{axis}_value_{role}", f"{axis}_rank_{role}"]
    w.writerow(header)
    ours_rank_by_ds = ours_rank_by_ds or {}
    for pos, name in enumerate(ordered):
        row = [
            pos, name,
            ours_rank_by_ds.get(name, ""),
            rank_sum.get(name, ""),
        ]
        for axis in all_axes:
            spec = AXES[axis]
            v = raw_per_axis[axis].get(name)
            r = ranks_per_axis[axis].get(name, "")
            row += [spec["fmt"].format(v) if v is not None else "", r]
        w.writerow(row)
    return buf.getvalue()


def render_svg(dot_text: str) -> str | None:
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        return None
    proc = subprocess.run(
        [dot_bin, "-Tsvg"], input=dot_text, capture_output=True, text=True
    )
    return proc.stdout if proc.returncode == 0 else None


def render_png_from_dot(dot_text: str, png_path: Path) -> bool:
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        return False
    proc = subprocess.run(
        [dot_bin, "-Tpng", "-o", str(png_path)], input=dot_text, text=True
    )
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs="+", default=list(AXES.keys()),
                        help=f"axes used for dominance (default: all)")
    parser.add_argument("--visualize", nargs="*", default=[],
                        help="extra display-only axes (no effect on dominance)")
    parser.add_argument("--rankdir", choices=["TB", "BT", "LR", "RL"], default="TB")
    parser.add_argument("--rank-scope", choices=["all", "filter"], default="all")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--keep-isolated", action="store_true")
    args = parser.parse_args()

    model_stats = json.loads(MODEL_STATS.read_text())
    dot_text, n_kept, n_edges, n_dom = build_dot_text(
        model_stats,
        compare_axes=args.compare,
        visualize_axes=args.visualize,
        drop_isolated=not args.keep_isolated,
        rankdir=args.rankdir,
        rank_scope=args.rank_scope,
    )
    OUT_DOT.write_text(dot_text)
    print(f"wrote {OUT_DOT} ({n_kept} nodes, {n_edges} cover edges, {n_dom} dominance pairs)")
    if not args.no_png:
        if render_png_from_dot(dot_text, OUT_PNG):
            print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
