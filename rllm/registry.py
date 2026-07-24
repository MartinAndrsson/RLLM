"""Experiment registry — the durable record of every spec + result + rationale for a problem.

Backed by plain JSON files on the (shared) filesystem so it doubles as the multi-machine source of
truth and survives relaunch (the 'resume, never restart' principle). One directory per problem work
area, e.g. <problem>/rllm_work/registry/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .interfaces import ExperimentSpec, RunResult


class Registry:
    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.specs_dir = self.work_dir / "registry" / "specs"
        self.results_dir = self.work_dir / "registry" / "results"
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # --- specs ---
    def put_spec(self, spec: ExperimentSpec) -> None:
        _atomic_write_json(self.specs_dir / f"{spec.id}.json", spec.as_dict())

    def get_spec(self, exp_id: str) -> ExperimentSpec | None:
        p = self.specs_dir / f"{exp_id}.json"
        if not p.exists():
            return None
        return ExperimentSpec(**json.loads(p.read_text()))

    def all_specs(self) -> list[ExperimentSpec]:
        return [ExperimentSpec(**json.loads(p.read_text())) for p in sorted(self.specs_dir.glob("*.json"))]

    # --- results (append-only; keyed by exp+seed) ---
    def put_result(self, result: RunResult) -> None:
        _atomic_write_json(self.results_dir / f"{result.exp_id}__seed{result.seed}.json", result.as_dict())

    def results_for(self, exp_id: str) -> list[RunResult]:
        out = []
        for p in sorted(self.results_dir.glob(f"{exp_id}__seed*.json")):
            out.append(RunResult(**json.loads(p.read_text())))
        return out

    def all_results(self) -> list[RunResult]:
        return [RunResult(**json.loads(p.read_text())) for p in sorted(self.results_dir.glob("*.json"))]


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)   # atomic on the same filesystem
