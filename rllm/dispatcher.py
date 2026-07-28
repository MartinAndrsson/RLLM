"""Multi-machine dispatch via a queue on the shared filesystem — no central server.

Each machine runs `worker(...)`, which atomically claims the next pending job (an (experiment, seed)
unit), runs the adapter's command, parses the result into the registry, and marks the job done. Atomic
claim = os.rename (atomic within one filesystem), so two machines never grab the same job.

Queue layout under <work_dir>/queue/: pending/ , claimed/ , done/ , failed/  (each holds job JSONs).
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable

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

    def _release(self, job: dict) -> None:
        """Put a claimed job back in pending/ (used by dry runs, which must not consume work)."""
        src = Path(job["_claimed_path"])
        if src.exists():
            os.replace(src, self.root / "pending" / f"{job['job_id']}.json")

    def pending_count(self) -> int:
        return len(list((self.root / "pending").glob("*.json")))


def run_process_group(cmd: list[str], timeout: float | None) -> tuple[int, bool]:
    """Run `cmd` in its own process group and enforce a HARD timeout on it.

    The training command is `bash -c "... python train.py ..."`, so killing just the direct child would
    orphan the trainer and let it keep burning the GPU past the session deadline. We therefore start a
    new session (new process group) and signal the whole group: SIGTERM (a chance to checkpoint), then
    SIGKILL. Returns (returncode, timed_out)."""
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        return proc.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)
            proc.wait(timeout=30)
        return (proc.returncode if proc.returncode is not None else -9), True
    except BaseException:                                   # Ctrl-C / SystemExit must not orphan a run
        _signal_group(proc, signal.SIGTERM)
        raise


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def worker(adapter: ProblemAdapter, work_dir: str | Path, device: str = "0",
           dry_run: bool = False, poll_seconds: float = 10.0, max_idle_polls: int = 1,
           latest_start: float | None = None, fit_seconds: Callable[[str], float] | None = None,
           per_run_timeout: Callable[[str], float] | None = None,
           on_result: Callable[[RunResult], None] | None = None) -> str:
    """Claim-and-run loop for ONE machine. Exits after `max_idle_polls` empty polls.

    Deadline discipline (§10.2), enforced here rather than by any model:
      * `latest_start`   -- a monotonic clock reading after which no new job may be CLAIMED;
      * `fit_seconds`    -- conservative estimate for a fidelity: a job is not claimed unless it is
                            expected to finish by `latest_start`;
      * `per_run_timeout`-- hard wall-clock cap per run, enforced on the whole process group.

    Returns why it stopped: "drained" | "deadline" | "rendered".
    """
    queue = Queue(work_dir)
    registry = Registry(work_dir)
    runs_dir = Path(work_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    held: list[dict] = []           # dry-run: claims to hand back once the queue is exhausted
    try:
        return _loop(adapter, queue, registry, runs_dir, device, dry_run, poll_seconds, max_idle_polls,
                     held, latest_start, fit_seconds, per_run_timeout, on_result)
    finally:
        for job in held:
            queue._release(job)


def _loop(adapter, queue, registry, runs_dir, device, dry_run, poll_seconds, max_idle_polls, held,
          latest_start, fit_seconds, per_run_timeout, on_result) -> str:
    idle = 0
    while True:
        if latest_start is not None and time.monotonic() >= latest_start:
            return "deadline"
        job = queue._claim_one()
        if job is None:
            if dry_run:
                return "rendered"   # every job rendered once; `held` is released by the caller
            idle += 1
            if idle >= max_idle_polls:
                return "drained"
            time.sleep(poll_seconds)
            continue
        idle = 0
        if latest_start is not None and fit_seconds is not None:
            # Would this run overrun the deadline? Hand it back for a later session rather than start
            # something we would have to kill (§10.2).
            if time.monotonic() + fit_seconds(job["fidelity"]) > latest_start:
                queue._release(job)
                return "deadline"
        spec = registry.get_spec(job["exp_id"])
        run_dir = str(runs_dir / f"{job['job_id']}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        cmd = adapter.build_command(spec, seed=job["seed"], run_dir=run_dir, device=device)
        if dry_run:
            # Render and validate the command only. A dry run must consume NOTHING: the claim is held
            # (so each job renders exactly once) and handed back to pending/ when the sweep ends, and no
            # result is recorded, so it can never be mistaken for evidence (§12).
            (Path(run_dir) / "DRYRUN_CMD.txt").write_text(" ".join(cmd) + "\n")
            print(f"[dry-run] {job['job_id']}: {' '.join(cmd)}")
            held.append(job)
            continue
        t0 = time.perf_counter()
        ok = True
        timeout = per_run_timeout(job["fidelity"]) if per_run_timeout else None
        try:
            rc, timed_out = run_process_group(cmd, timeout)
            if timed_out:
                ok, metrics = False, {"error": f"run exceeded its {timeout:.0f}s hard timeout; "
                                               f"process group terminated"}
            elif rc != 0:
                ok, metrics = False, {"error": f"command exited {rc}"}
            else:
                metrics = adapter.parse_result(run_dir)
                ok = "error" not in metrics
        except Exception as exc:                                  # noqa: BLE001
            ok = False
            metrics = {"error": str(exc)}
        result = RunResult(
            exp_id=job["exp_id"], seed=job["seed"], fidelity=job["fidelity"],
            run_dir=run_dir, status="done" if ok else "failed", metrics=metrics,
            wall_seconds=time.perf_counter() - t0,
        )
        registry.put_result(result)
        queue._finish(job, ok)
        if on_result:
            on_result(result)
