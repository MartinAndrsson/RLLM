"""Multi-machine dispatch via a queue on the shared filesystem — no central server.

Each machine runs `worker(...)`, which atomically claims the next pending job (an (experiment, seed)
unit), runs the adapter's command, parses the result into the registry, and marks the job done. Atomic
claim = os.rename (atomic within one filesystem), so two machines never grab the same job.

Queue layout under <work_dir>/queue/: pending/ , claimed/ , done/ , failed/  (each holds job JSONs).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from .interfaces import ProblemAdapter, RunResult
from .registry import Registry, _atomic_write_json


class Queue:
    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.root = self.work_dir / "queue"
        for sub in ("pending", "claimed", "done", "failed"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def enqueue(self, exp_id: str, seed: int, fidelity: str) -> str:
        job_id = f"{exp_id}__seed{seed}"
        _atomic_write_json(self.root / "pending" / f"{job_id}.json",
                           {"exp_id": exp_id, "seed": seed, "fidelity": fidelity, "job_id": job_id})
        return job_id

    def _claim_one(self) -> dict | None:
        """Atomically move one pending job to claimed/. Returns the job dict, or None if queue empty."""
        tag = f"{socket.gethostname()}.{os.getpid()}"
        for p in sorted((self.root / "pending").glob("*.json")):
            dest = self.root / "claimed" / f"{p.stem}.{tag}.json"
            try:
                os.rename(p, dest)          # atomic; loser raises FileNotFoundError
            except (FileNotFoundError, OSError):
                continue                    # another worker took it; try next
            return {**json.loads(dest.read_text()), "_claimed_path": str(dest)}
        return None

    def _finish(self, job: dict, ok: bool) -> None:
        src = Path(job["_claimed_path"])
        dest = self.root / ("done" if ok else "failed") / f"{job['job_id']}.json"
        if src.exists():
            os.replace(src, dest)

    def pending_count(self) -> int:
        return len(list((self.root / "pending").glob("*.json")))


def worker(adapter: ProblemAdapter, work_dir: str | Path, device: str = "0",
           dry_run: bool = False, poll_seconds: float = 10.0, max_idle_polls: int = 1) -> None:
    """Claim-and-run loop for ONE machine. Exits after `max_idle_polls` empty polls."""
    queue = Queue(work_dir)
    registry = Registry(work_dir)
    runs_dir = Path(work_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    idle = 0
    while True:
        job = queue._claim_one()
        if job is None:
            idle += 1
            if idle >= max_idle_polls:
                return
            time.sleep(poll_seconds)
            continue
        idle = 0
        spec = registry.get_spec(job["exp_id"])
        run_dir = str(runs_dir / f"{job['job_id']}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        cmd = adapter.build_command(spec, seed=job["seed"], run_dir=run_dir, device=device)
        t0 = time.perf_counter()
        ok = True
        if dry_run:
            (Path(run_dir) / "DRYRUN_CMD.txt").write_text(" ".join(cmd) + "\n")
            metrics = {"_dry_run": True}
        else:
            try:
                subprocess.run(cmd, check=True)
                metrics = adapter.parse_result(run_dir)
            except Exception as exc:                                  # noqa: BLE001
                ok = False
                metrics = {"error": str(exc)}
        registry.put_result(RunResult(
            exp_id=job["exp_id"], seed=job["seed"], fidelity=job["fidelity"],
            run_dir=run_dir, status="done" if ok else "failed", metrics=metrics,
            wall_seconds=time.perf_counter() - t0,
        ))
        queue._finish(job, ok)
