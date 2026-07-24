"""LLM control loop (Phase 2): actor proposes experiments, reviewer critiques, approved ones are
enqueued at the cheap 'screen' rung. NOTHING trains here — a human/worker still runs `work` — so the
review gate always precedes real compute. Every proposal + verdict + action is logged (reproducibility).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..interfaces import ProblemAdapter, ExperimentSpec
from ..registry import Registry
from .. import ladder
from .backend import LLMBackend, parse_json
from . import prompts


def load_memory(work_dir) -> str:
    """problem.md + journal.md — the per-problem memory read FIRST (resume, never restart)."""
    wd = Path(work_dir)
    out = []
    for name in ("problem.md", "journal.md"):
        p = wd / name
        if p.exists():
            out.append(f"===== {name} =====\n{p.read_text()}")
    return "\n\n".join(out) if out else "(no memory files found)"


def registry_summary(adapter: ProblemAdapter, work_dir, max_rows: int = 40) -> str:
    reg = Registry(work_dir)
    specs = reg.all_specs()
    lines = [f"{len(specs)} specs, {len(reg.all_results())} results. Ranked by fidelity:"]
    for fid in ("screen", "refine", "confirm"):
        ranked = ladder.rank(adapter, work_dir, fid)
        if ranked:
            lines.append(f"[{fid}]")
            for eid, m in ranked[:max_rows]:
                lines.append(f"  {eid}: success={m.get('success_fraction', 0):.2f} "
                             f"short_days={m.get('mean_shortage_days', float('nan')):.2f}")
    queued = {s.id for s in specs}
    if queued:
        lines.append("already-registered ids: " + ", ".join(sorted(queued)))
    return "\n".join(lines)


def _log(work_dir, record: dict) -> None:
    with open(Path(work_dir) / "decisions.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **record}) + "\n")


def propose_and_review(actor: LLMBackend, reviewer: LLMBackend, adapter: ProblemAdapter, work_dir,
                       n: int = 4, max_revisions: int = 1) -> dict:
    memory = load_memory(work_dir)
    summary = registry_summary(adapter, work_dir)

    reasons: list[str] | None = None
    proposal = None
    for attempt in range(max_revisions + 1):
        raw = actor.ask(prompts.PROPOSE_SYSTEM, prompts.propose_user(memory, summary, n, reasons))
        proposal = parse_json(raw)
        verdict = parse_json(reviewer.ask(prompts.REVIEW_SYSTEM, prompts.review_user(proposal, memory, summary)))
        _log(work_dir, {"stage": "propose", "attempt": attempt, "proposal": proposal, "verdict": verdict})
        v = str(verdict.get("verdict", "block")).lower()
        if v == "approve":
            keep = set(verdict.get("keep_ids") or [e["id"] for e in proposal.get("experiments", [])])
            break
        if v == "revise" and attempt < max_revisions:
            reasons = verdict.get("reasons") or ["(no reasons given)"]
            continue
        # block, or revise with no attempts left
        _log(work_dir, {"stage": "final", "action": "no_enqueue", "verdict_final": v})
        return {"action": "blocked", "verdict": verdict, "proposal": proposal}

    experiments = [e for e in proposal.get("experiments", []) if e.get("id") in keep]
    specs = [ExperimentSpec(id=e["id"], hypothesis=e.get("hypothesis", ""), config=dict(e.get("config", {})))
             for e in experiments]
    n_jobs = ladder.enqueue_experiments(adapter, work_dir, specs, "screen") if specs else 0
    _log(work_dir, {"stage": "final", "action": "enqueued", "ids": [s.id for s in specs], "jobs": n_jobs})
    return {"action": "enqueued", "ids": [s.id for s in specs], "jobs": n_jobs}
