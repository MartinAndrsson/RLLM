"""The deliverable table: what was tried, what it measured, and how the alternatives compare.

Two views, both read straight from the registry:

  * the flat table — every experiment at one fidelity, best first, with the knobs that produced it;
  * grouped by a knob (`--by ALGO`) — the like-for-like comparison, e.g. algorithm against algorithm.

The grouped view is deliberately blunt about comparability: a group's rows are only worth comparing when
they were run at the same rung with the same seed count, so the table states the seed count per row and
flags groups that were only ever screened. Nothing here consults a model.
"""
from __future__ import annotations

from statistics import mean

from . import ladder
from .brief import ProblemBrief
from .interfaces import ProblemAdapter
from .registry import Registry


def _rows(adapter: ProblemAdapter, work_dir, fidelity: str) -> list[dict]:
    specs = {s.id: s for s in Registry(work_dir).all_specs()}
    rung_keys = {k for f in adapter.fidelity_levels() for k in f.overrides}
    out = []
    for eid, metrics in ladder.rank(adapter, work_dir, fidelity):
        spec = specs.get(eid)
        config = {**adapter.base_config(), **(spec.config if spec else {})}
        out.append({
            "id": eid,
            "n_seeds": int(metrics.get("n_seeds", 0)),
            "metrics": {k: v for k, v in metrics.items() if k != "n_seeds"},
            "config": {k: v for k, v in config.items() if k not in rung_keys},
            "hypothesis": spec.hypothesis if spec else "",
        })
    return out


def _fmt(value) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def flat_table(adapter: ProblemAdapter, brief: ProblemBrief, work_dir, fidelity: str,
               show_knobs: list[str] | None = None) -> str:
    rows = _rows(adapter, work_dir, fidelity)
    if not rows:
        return f"_no completed runs at `{fidelity}`_"
    metric_keys = _metric_order(brief, rows)
    knobs = show_knobs if show_knobs is not None else _varying_knobs(rows)
    head = ["experiment", "seeds", *(f"`{k}`" for k in metric_keys), *(f"`{k}`" for k in knobs), "solved"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [f"`{r['id']}`", str(r["n_seeds"])]
        cells += [_fmt(r["metrics"][k]) if k in r["metrics"] else "—" for k in metric_keys]
        cells += [_fmt(r["config"][k]) if k in r["config"] else "—" for k in knobs]
        cells.append("**yes**" if brief.is_solved(r["metrics"]) else "no")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def grouped_table(adapter: ProblemAdapter, brief: ProblemBrief, work_dir, knob: str,
                  fidelity: str) -> str:
    """One row per value of `knob` — the like-for-like comparison (e.g. algorithm vs algorithm)."""
    rows = _rows(adapter, work_dir, fidelity)
    if not rows:
        return f"_no completed runs at `{fidelity}`_"
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(_fmt(r["config"].get(knob, "(unset)")), []).append(r)
    primary = brief.primary_metric
    best_first = sorted(groups.items(),
                        key=lambda kv: max(adapter.sort_key(r["metrics"]) for r in kv[1]), reverse=True)
    head = [f"`{knob}`", "experiments", "seeds (min-max)", f"best `{primary}`", f"mean `{primary}`",
            "best experiment", "solved"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for value, rs in best_first:
        values = [r["metrics"].get(primary) for r in rs
                  if isinstance(r["metrics"].get(primary), (int, float))]
        best = max(rs, key=lambda r: adapter.sort_key(r["metrics"]))
        seeds = [r["n_seeds"] for r in rs]
        lines.append("| " + " | ".join([
            f"`{value}`", str(len(rs)),
            f"{min(seeds)}-{max(seeds)}" if min(seeds) != max(seeds) else str(min(seeds)),
            _fmt(max(values)) if values else "—",
            _fmt(mean(values)) if values else "—",
            f"`{best['id']}`",
            "**yes**" if any(brief.is_solved(r["metrics"]) for r in rs) else "no",
        ]) + " |")
    return "\n".join(lines)


def _metric_order(brief: ProblemBrief, rows: list[dict]) -> list[str]:
    """Primary metric first, then the tie-break, then everything else alphabetically."""
    seen = {k for r in rows for k in r["metrics"]}
    order = [brief.primary_metric] if brief.primary_metric in seen else []
    if brief.tie_break_metric in seen and brief.tie_break_metric not in order:
        order.append(brief.tie_break_metric)
    return order + sorted(seen - set(order))


def _varying_knobs(rows: list[dict], limit: int = 6) -> list[str]:
    """Only the knobs that actually differ between rows — the columns that explain the ranking. A table
    repeating the same value in every row hides the comparison instead of showing it."""
    keys = {k for r in rows for k in r["config"]}
    varying = [k for k in sorted(keys)
               if len({_fmt(r["config"].get(k, "(unset)")) for r in rows}) > 1]
    return varying[:limit]


def render(adapter: ProblemAdapter, brief: ProblemBrief, work_dir, fidelity: str | None = None,
           by: str | None = None) -> str:
    """The full report: the deliverable table at the required fidelity, the comparison if asked for, and
    an explicit statement of what is NOT yet comparable."""
    required = fidelity or brief.success_criterion.required_fidelity
    rungs = [f.name for f in adapter.fidelity_levels()]
    out = [f"# {brief.problem_id} — performance comparison", "",
           f"**Goal:** {brief.goal}", "",
           f"**Solved when:** {brief.success_criterion.describe()}", ""]

    if by:
        if by not in adapter.declared_knobs():
            return "\n".join(out + [f"_`{by}` is not a declared knob; nothing to group by._"])
        out += [f"## By `{by}`, at `{required}`", "",
                grouped_table(adapter, brief, work_dir, by, required), "",
                "A row is only comparable with another when both were run at the same rung with the "
                "same seed count — check the seeds column before reading the ranking as a verdict.", ""]

    out += [f"## All experiments at `{required}` (the required fidelity)", "",
            flat_table(adapter, brief, work_dir, required), ""]

    cheaper = [r for r in rungs if r != required]
    unconfirmed = [r for r in cheaper if ladder.rank(adapter, work_dir, r)]
    if not ladder.rank(adapter, work_dir, required) and unconfirmed:
        out += [f"> Nothing has reached `{required}` yet, so **no row above is confirmed**. The cheaper "
                f"rungs below rank candidates but their ordering need not transfer.", ""]
    for rung in unconfirmed:
        out += [f"## `{rung}` — unconfirmed (cheaper rung; ranking may not transfer)", "",
                flat_table(adapter, brief, work_dir, rung), ""]
    return "\n".join(out)
