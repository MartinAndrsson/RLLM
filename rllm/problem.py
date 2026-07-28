"""Resolving a work directory into (brief, adapter) — the one place that maps a brief to an adapter.

Keeping this separate keeps the core problem-agnostic: `rllm.session` and the CLI never import a
specific adapter, they ask here.
"""
from __future__ import annotations

from pathlib import Path

from .brief import ProblemBrief
from .interfaces import ProblemAdapter

BRIEF_NAME = "brief.json"


def brief_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / BRIEF_NAME


def load_brief(work_dir: str | Path) -> ProblemBrief:
    path = brief_path(work_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no {BRIEF_NAME} in {work_dir}. Run the onboarding interview first:\n"
            f"  ./scripts/rllm-onboard.sh")
    brief = ProblemBrief.load(path)
    problems = brief.problems()
    if problems:
        raise ValueError(f"{path} is not usable:\n" + "\n".join(f"  - {p}" for p in problems))
    return brief


def load_adapter(brief: ProblemBrief) -> ProblemAdapter:
    """`adapter.name == "rx"` uses the hand-written Rx adapter; anything else uses the generic
    brief-driven adapter (test_command + metrics_file + declared knobs)."""
    if brief.adapter.name == "rx":
        from adapters.rx.adapter import RxAdapter
        cfg = brief.adapter.config or {}
        kwargs = {}
        if brief.problem_repo:
            kwargs["rx_repo"] = brief.problem_repo
        if cfg.get("python"):
            kwargs["py"] = cfg["python"]
        return RxAdapter(**kwargs)
    from adapters.brief_adapter import BriefAdapter
    return BriefAdapter(brief)


def load(work_dir: str | Path) -> tuple[ProblemBrief, ProblemAdapter]:
    brief = load_brief(work_dir)
    return brief, load_adapter(brief)
