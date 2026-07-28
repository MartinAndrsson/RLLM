"""Coarse-to-fine multi-fidelity driver.

Enqueue candidate experiments at the cheapest rung (screen), then promote the best few up the ladder
(refine -> confirm). The LLM controller (later) chooses which experiments to enqueue and reads results
semantically; this module is the mechanical promotion logic it drives. Kept deliberately simple:
promotion is between waves (runs take hours), not a tight loop.

PROXY-VALIDITY CAVEAT (see DESIGN.md): a cheap rung is only useful if its ranking transfers to full
fidelity. Confirm periodically; if the ranking flips, the screen fidelity is too coarse.
"""
from __future__ import annotations

from statistics import mean

from .interfaces import ProblemAdapter, ExperimentSpec
from .registry import Registry
from .dispatcher import Queue
from .validate import validate_spec


def _fidelity(adapter: ProblemAdapter, name: str):
    for f in adapter.fidelity_levels():
        if f.name == name:
            return f
    raise ValueError(f"unknown fidelity {name!r}")


def enqueue_experiments(adapter: ProblemAdapter, work_dir, specs: list[ExperimentSpec], fidelity: str,
                        llm_proposed: bool = True, permitted_task_changes: tuple[str, ...] = ()) -> int:
    """Register specs (with fidelity overrides merged into config) and queue one job per seed.

    This is the single authorization choke point: every spec is validated against the adapter's knob
    whitelist FIRST, and if any spec is invalid nothing is enqueued (an all-or-nothing gate keeps the
    queue consistent with what was reviewed). Callers that build specs in code — not from a model —
    pass llm_proposed=False; the LLM controller filters/relays rejections before it gets here."""
    problems = [e for spec in specs
                for e in validate_spec(adapter, spec, llm_proposed=llm_proposed,
                                       permitted_task_changes=permitted_task_changes)]
    if problems:
        raise ValueError("refusing to enqueue — invalid experiment spec(s):\n" +
                         "\n".join(f"  - {p}" for p in problems))
    registry, queue = Registry(work_dir), Queue(work_dir)
    fid = _fidelity(adapter, fidelity)
    n = 0
    for spec in specs:
        merged = ExperimentSpec(
            id=spec.id, hypothesis=spec.hypothesis,
            config={**spec.config, **fid.overrides}, fidelity=fidelity, parent=spec.parent,
        )
        registry.put_spec(merged)
        for seed in fid.seeds:
            queue.enqueue(merged.id, seed, fidelity)
            n += 1
    return n


def _mean_metrics(adapter: ProblemAdapter, registry: Registry, exp_id: str) -> dict | None:
    """Seed-average every numeric metric a completed experiment reported. Problem-agnostic: the core
    never names a metric, so a new problem needs no changes here (§8)."""
    results = [r for r in registry.results_for(exp_id) if r.status == "done" and "error" not in r.metrics]
    if not results:
        return None
    agg: dict[str, float] = {}
    for key in {k for r in results for k in r.metrics}:
        vals = [r.metrics[key] for r in results
                if isinstance(r.metrics.get(key), (int, float)) and not isinstance(r.metrics[key], bool)]
        if vals:
            agg[key] = mean(vals)
    agg["n_seeds"] = float(len(results))
    return agg or None


def rank(adapter: ProblemAdapter, work_dir, fidelity: str) -> list[tuple[str, dict]]:
    """Rank completed experiments at a given fidelity, best first (adapter.better)."""
    registry = Registry(work_dir)
    exp_ids = {s.id for s in registry.all_specs() if s.fidelity == fidelity}
    scored = [(eid, _mean_metrics(adapter, registry, eid)) for eid in exp_ids]
    scored = [(eid, m) for eid, m in scored if m is not None]
    scored.sort(key=lambda em: adapter.sort_key(em[1]), reverse=True)
    return scored


def promote(adapter: ProblemAdapter, work_dir, from_fidelity: str, to_fidelity: str,
            top_k: int | None = None, exp_ids: list[str] | None = None) -> list[str]:
    """Enqueue experiments from `from_fidelity` at `to_fidelity`.

    Either the top_k of the ranking, or exactly `exp_ids` when the caller has already decided (the
    session planner does, because it must exclude candidates that were promoted in an earlier wave —
    passing top_k there would silently re-promote the current leader and pay for the same runs twice).
    Anything already present at the target rung is skipped either way.
    """
    registry = Registry(work_dir)
    ranked = rank(adapter, work_dir, from_fidelity)
    if exp_ids is not None:
        wanted = list(exp_ids)
        ranked = [(eid, m) for eid, m in ranked if eid in wanted]
    if top_k is not None:
        ranked = ranked[:top_k]
    existing = {s.id for s in registry.all_specs()}
    ranked = [(eid, m) for eid, m in ranked if f"{_slug(eid)}@{to_fidelity}" not in existing]
    # Strip every knob any rung owns, so the new rung's overrides apply cleanly and a cheap rung's
    # cost-cutting (e.g. screen's BC_EPOCHS=0) is never silently inherited by an expensive one.
    rung_keys = {k for f in adapter.fidelity_levels() for k in f.overrides}
    specs = []
    for eid, _ in ranked:
        base = registry.get_spec(eid)
        base_config = {k: v for k, v in base.config.items() if k not in rung_keys}
        specs.append(ExperimentSpec(id=f"{_slug(eid)}@{to_fidelity}", hypothesis=base.hypothesis,
                                    config=base_config, fidelity=to_fidelity, parent=eid))
    if not specs:
        return []
    # Already-authorized config being re-run at a higher rung, so not re-gated as a fresh LLM proposal.
    enqueue_experiments(adapter, work_dir, specs, to_fidelity, llm_proposed=False)
    return [s.id for s in specs]


def _slug(exp_id: str) -> str:
    """Base slug of an experiment id ("m1@screen" -> "m1"), so promoting twice can't stack suffixes."""
    return exp_id.split("@", 1)[0]
