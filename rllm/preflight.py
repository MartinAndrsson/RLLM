"""Preflight checks — fail before the session starts, not four hours in.

Everything here is deterministic and cheap: does the problem repo exist, is there something runnable, does
the interpreter exist, are the model CLIs on PATH. A session that cannot possibly work should say so in
two seconds, with a specific message, rather than burning a night's window discovering it one failed run
at a time.

This is the *before* half of "stop and tell the user". The *during* half lives in rllm/session.py: runs
that keep failing end the session as `blocked` rather than proposing into a void.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .brief import ProblemBrief
from .interfaces import ProblemAdapter
from .validate import validate_adapter


def check(brief: ProblemBrief, adapter: ProblemAdapter, work_dir, need_llm: bool = True) -> list[str]:
    """Everything that would stop this session from producing a result. Empty list = safe to start."""
    problems: list[str] = []
    problems += [f"brief: {p}" for p in brief.problems()]
    problems += [f"adapter: {p}" for p in validate_adapter(adapter)]
    problems += _check_runnable(brief, adapter)
    problems += _check_criterion(brief, adapter)
    problems += _check_work_dir(work_dir)
    if need_llm:
        problems += _check_model_clis()
    return problems


def _check_runnable(brief: ProblemBrief, adapter: ProblemAdapter) -> list[str]:
    """Is there actually something to execute, and can it start?"""
    problems: list[str] = []
    repo = Path(brief.problem_repo)
    if not repo.is_dir():
        return [f"problem repo {brief.problem_repo!r} does not exist or is not a directory — "
                f"nothing can run"]
    if not os.access(repo, os.R_OK):
        problems.append(f"problem repo {brief.problem_repo!r} is not readable")

    variants = getattr(adapter, "variants", lambda: [])()
    variants_dir = getattr(adapter, "variants_dir", None)
    if variants:
        entry_name = getattr(adapter, "_variant_entry", "train.py")
        for name in variants:
            entry = Path(variants_dir) / name / entry_name
            if not entry.is_file():
                problems.append(f"variant {name!r}: no {entry_name} at {entry}")
            elif not os.access(entry, os.R_OK):
                problems.append(f"variant {name!r}: {entry} is not readable")
    elif not brief.test_command.strip():
        where = f" (looked in {variants_dir})" if variants_dir else ""
        problems.append(
            f"nothing to run: no variants on disk{where} and the brief declares no test_command. "
            f"Author a design under problems/<problem_id>/variants/ (see problems/README.md), or give "
            f"the brief a test_command.")

    interpreter = getattr(adapter, "_python", None)
    if interpreter and not (shutil.which(interpreter) or Path(interpreter).is_file()):
        problems.append(f"interpreter {interpreter!r} not found — every run would fail to start. "
                        f"Set adapter.config.python in the brief to a real interpreter.")

    if brief.metrics_file and Path(brief.metrics_file).is_absolute():
        problems.append(f"metrics_file {brief.metrics_file!r} must be relative to the run directory, "
                        f"not absolute — otherwise concurrent runs overwrite each other's results")
    return problems


def _check_criterion(brief: ProblemBrief, adapter: ProblemAdapter) -> list[str]:
    """A criterion that can never be evaluated is a session that can never succeed."""
    problems: list[str] = []
    rungs = {f.name for f in adapter.fidelity_levels()}
    required = brief.success_criterion.required_fidelity
    if required not in rungs:
        problems.append(f"success_criterion.required_fidelity {required!r} is not one of the adapter's "
                        f"rungs {sorted(rungs)} — 'solved' could never be evaluated")
    seeds = {f.name: len(f.seeds) for f in adapter.fidelity_levels()}
    needed = brief.success_criterion.minimum_training_seeds
    if required in seeds and seeds[required] < needed:
        problems.append(f"rung {required!r} runs {seeds[required]} seed(s) but the criterion needs "
                        f">={needed} — 'solved' is unreachable by construction. Raise the rung's seeds "
                        f"or lower minimum_training_seeds.")
    knobs = adapter.declared_knobs()
    for name in [*brief.permitted_task_changes, *brief.forbidden_task_changes]:
        if name not in knobs:
            problems.append(f"task-change knob {name!r} in the brief is not declared by the adapter — "
                            f"the guard rail refers to something that does not exist")
    if not any(k.llm_may_change and k.category != "task_definition" for k in knobs.values()):
        problems.append("no knob is proposable (all are harness-only or task_definition) — every "
                        "proposal would be rejected and the session could only re-run the base config")
    return problems


def _check_work_dir(work_dir) -> list[str]:
    path = Path(work_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".rllm_write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return [f"work dir {work_dir} is not writable ({exc}) — results could not be recorded"]
    return []


def _check_model_clis() -> list[str]:
    missing = [name for name in ("claude", "codex") if not shutil.which(name)]
    if missing:
        return [f"model CLI(s) not on PATH: {', '.join(missing)} — proposals and the handoff would "
                f"fail. Install/login, or run with --no-llm to execute only queued work."]
    return []
