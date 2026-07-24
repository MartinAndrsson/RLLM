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


def _fidelity(adapter: ProblemAdapter, name: str):
    for f in adapter.fidelity_levels():
        if f.name == name:
            return f
    raise ValueError(f"unknown fidelity {name!r}")


def enqueue_experiments(adapter: ProblemAdapter, work_dir, specs: list[ExperimentSpec], fidelity: str) -> int:
    """Register specs (with fidelity overrides merged into config) and queue one job per seed."""
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
    results = [r for r in registry.results_for(exp_id) if r.status == "done" and "error" not in r.metrics]
    if not results:
        return None
    keys = ("success_fraction", "mean_shortage_days")
    agg = {}
    for k in keys:
        vals = [r.metrics.get(k) for r in results if isinstance(r.metrics.get(k), (int, float))]
        if vals:
            agg[k] = mean(vals)
    return agg or None


def rank(adapter: ProblemAdapter, work_dir, fidelity: str) -> list[tuple[str, dict]]:
    """Rank completed experiments at a given fidelity, best first (adapter.better)."""
    registry = Registry(work_dir)
    exp_ids = {s.id for s in registry.all_specs() if s.fidelity == fidelity}
    scored = [(eid, _mean_metrics(adapter, registry, eid)) for eid in exp_ids]
    scored = [(eid, m) for eid, m in scored if m is not None]
    scored.sort(key=lambda em: _key(adapter, em[1]), reverse=True)
    return scored


def _key(adapter: ProblemAdapter, m: dict):
    return (m.get("success_fraction", -1.0), -m.get("mean_shortage_days", 1e9))


def promote(adapter: ProblemAdapter, work_dir, from_fidelity: str, to_fidelity: str, top_k: int) -> list[str]:
    """Take the top_k experiments completed at from_fidelity and enqueue them at to_fidelity."""
    registry = Registry(work_dir)
    ranked = rank(adapter, work_dir, from_fidelity)[:top_k]
    specs = []
    for eid, _ in ranked:
        base = registry.get_spec(eid)
        # strip the old fidelity overrides so the new rung's overrides apply cleanly
        specs.append(ExperimentSpec(id=f"{eid}@{to_fidelity}", hypothesis=base.hypothesis,
                                    config=base.config, fidelity=to_fidelity, parent=eid))
    enqueue_experiments(adapter, work_dir, specs, to_fidelity)
    return [s.id for s in specs]
