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
from ..validate import validate_spec, knob_table
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
    """What has been tried and what it measured — including the KNOB VALUES behind each result.

    The configs matter as much as the scores: without them the actor knows a run called
    `mid_smooth_high_thresh` scored 0.80 but not what "high" was, so it cannot search around it. A live
    session flagged exactly that gap. Metric names are never hardcoded here, so a new problem needs no
    change (§8).
    """
    reg = Registry(work_dir)
    specs = {s.id: s for s in reg.all_specs()}
    rung_keys = {k for f in adapter.fidelity_levels() for k in f.overrides}
    lines = [f"{len(specs)} specs, {len(reg.all_results())} results. Ranked best-first per fidelity "
             f"(config shows only the experiment's own knobs; base recipe and rung knobs omitted):"]
    for fid in [f.name for f in adapter.fidelity_levels()]:
        ranked = ladder.rank(adapter, work_dir, fid)
        if not ranked:
            continue
        lines.append(f"[{fid}]")
        for eid, m in ranked[:max_rows]:
            metrics = " ".join(f"{k}={m[k]:.4g}" for k in sorted(m) if k != "n_seeds")
            spec = specs.get(eid)
            config = {k: v for k, v in (spec.config if spec else {}).items() if k not in rung_keys}
            lines.append(f"  {eid}: n_seeds={int(m.get('n_seeds', 0))} {metrics} "
                         f"config={json.dumps(config, sort_keys=True)}")
    if specs:
        lines.append("already-registered ids (proposing one of these, or its config under a new name, "
                     "is rejected as a duplicate): " + ", ".join(sorted(specs)))
    return "\n".join(lines)


def _log(work_dir, record: dict) -> None:
    with open(Path(work_dir) / "decisions.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **record}) + "\n")


def _to_specs(proposal: dict) -> list[ExperimentSpec]:
    """Model JSON -> ExperimentSpecs, tolerating a malformed experiments list (validation rejects the
    bad ones by name rather than crashing the session)."""
    out = []
    for e in proposal.get("experiments") or []:
        if not isinstance(e, dict):
            continue
        out.append(ExperimentSpec(id=str(e.get("id", "")), hypothesis=str(e.get("hypothesis", "")),
                                  config=dict(e.get("config") or {}) if isinstance(e.get("config"), dict)
                                  else e.get("config")))
    return out


def _screen(adapter: ProblemAdapter, specs: list[ExperimentSpec], work_dir,
            permitted_task_changes: tuple[str, ...] = ()) -> tuple[list[ExperimentSpec], list[str]]:
    """Split proposed specs into (authorized, rejection reasons) using ONLY deterministic validation."""
    ok, reasons = [], []
    # Duplicate detection needs the specs already registered PLUS the ones approved earlier in this same
    # batch (an actor can propose the same idea twice in one breath).
    seen = list(Registry(work_dir).all_specs())
    for spec in specs:
        errs = validate_spec(adapter, spec, permitted_task_changes=permitted_task_changes,
                             existing=seen)
        if errs:
            reasons.extend(f"experiment {spec.id!r} rejected: {e}" for e in errs)
        else:
            ok.append(spec)
            seen.append(spec)
    return ok, reasons


def _approved_ids(verdict: dict) -> list[str]:
    """The ids the reviewer explicitly approved. `keep_ids` is accepted as an alias of the schema's
    `approved_item_ids`; anything else (absent, null, not a list) approves nothing."""
    ids = verdict.get("approved_item_ids", verdict.get("keep_ids"))
    return [str(i) for i in ids] if isinstance(ids, list) else []


def propose_and_review(actor: LLMBackend, reviewer: LLMBackend, adapter: ProblemAdapter, work_dir,
                       n: int = 4, max_revisions: int = 1,
                       permitted_task_changes: tuple[str, ...] = ()) -> dict:
    memory = load_memory(work_dir)
    summary = registry_summary(adapter, work_dir)
    knobs = knob_table(adapter)
    rungs = prompts.ladder_text(adapter.fidelity_levels())
    base = prompts.base_config_text(adapter.base_config())
    ident = {"actor": actor.name, "actor_model": getattr(actor, "model", None),
             "reviewer": reviewer.name, "reviewer_model": getattr(reviewer, "model", None)}

    reasons: list[str] | None = None
    proposal, specs = None, []
    for attempt in range(max_revisions + 1):
        raw = actor.ask(prompts.PROPOSE_SYSTEM,
                        prompts.propose_user(memory, summary, knobs, rungs, base, n, reasons))
        proposal = parse_json(raw)

        # Gate 1 (deterministic, non-negotiable): the knob whitelist. Invalid experiments never reach
        # the reviewer's attention as runnable work, and their reasons become revision feedback.
        specs, invalid = _screen(adapter, _to_specs(proposal), work_dir, permitted_task_changes)
        verdict = parse_json(reviewer.ask(
            prompts.REVIEW_SYSTEM, prompts.review_user(proposal, memory, summary, rungs, base, invalid)))
        _log(work_dir, {"stage": "propose", "attempt": attempt, **ident, "proposal": proposal,
                        "validation_rejected": invalid, "valid_ids": [s.id for s in specs],
                        "verdict": verdict})
        v = str(verdict.get("verdict", "block")).lower()
        if v == "approve":
            # Gate 2: the reviewer may only ever SHRINK the authorized set, never extend it — and an
            # absent/empty approval list approves NOTHING (§7.2: it must never default to all items).
            keep = set(_approved_ids(verdict))
            specs = [s for s in specs if s.id in keep]
            break
        if attempt < max_revisions:
            reasons = (verdict.get("required_changes") or verdict.get("reasons") or []) + invalid \
                      or ["(no reasons given)"]
            continue
        _log(work_dir, {"stage": "final", "action": "no_enqueue", **ident, "verdict_final": v,
                        "validation_rejected": invalid})
        return {"action": "blocked", "verdict": verdict, "proposal": proposal, "rejected": invalid,
                "revisions": attempt}

    cheapest = adapter.fidelity_levels()[0].name
    n_jobs = ladder.enqueue_experiments(adapter, work_dir, specs, cheapest,
                                        permitted_task_changes=permitted_task_changes) if specs else 0
    _log(work_dir, {"stage": "final", "action": "enqueued", **ident,
                    "ids": [s.id for s in specs], "jobs": n_jobs})
    return {"action": "enqueued", "ids": [s.id for s in specs], "jobs": n_jobs, "revisions": attempt}
