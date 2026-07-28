"""Proof that the problem repository was not modified by a run.

The models write their own environment, reward and action code — but in THIS repository, under
`problems/<problem_id>/variants/<variant>/`, importing the user's repo as a read-only simulator. That
separation is only meaningful if it is checked, so every run is bracketed by a fingerprint of the problem
repo: if the fingerprint changes, the run is failed and the session stops, because from that point on
every number is suspect (the thing being measured may have been altered).

This is a guard, not a sandbox. It detects modification after the fact rather than preventing it; real
prevention needs a container or a read-only bind mount, which is the next step up. What it does buy is
that a silently-edited simulator can never masquerade as a good result.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

MAX_FILES = 20000            # cap the non-git walk so fingerprinting cannot itself become the bottleneck
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".venv", "node_modules"}


class ProblemRepoModified(RuntimeError):
    """Raised when a run changed a path that was declared read-only."""


def fingerprint(path: str | Path) -> str:
    """A cheap, stable digest of a directory tree.

    Prefers git (exact, and it notices deletions and content edits regardless of timestamps); falls back
    to a walk over (relative path, size, mtime_ns) when the tree is not a git repository.
    """
    path = Path(path)
    if not path.is_dir():
        return "missing"
    git = _git_fingerprint(path)
    if git is not None:
        return f"git:{git}"
    h = hashlib.sha256()
    count = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            full = Path(root) / name
            try:
                st = full.stat()
            except OSError:
                continue
            h.update(f"{full.relative_to(path)}\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
            count += 1
            if count >= MAX_FILES:
                return f"walk:truncated:{h.hexdigest()[:16]}"
    return f"walk:{count}:{h.hexdigest()[:16]}"


def _git_fingerprint(path: Path) -> str | None:
    """HEAD plus the hash of the working-tree status, so both commits and dirty edits are captured.
    Returns None when `path` is not inside a git work tree, or git is unavailable."""
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        if head.returncode != 0:
            return None
        status = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=120)
        if status.returncode != 0:
            return None
        digest = hashlib.sha256(status.stdout.encode()).hexdigest()[:16]
        return f"{head.stdout.strip()[:12]}:{digest}"
    except (OSError, subprocess.SubprocessError):
        return None


def snapshot(paths: list[str]) -> dict[str, str]:
    return {p: fingerprint(p) for p in paths}


def verify(before: dict[str, str], context: str = "") -> None:
    """Re-fingerprint and raise if anything moved."""
    changed = [f"{p}: {was} -> {now}" for p, was in before.items()
               if (now := fingerprint(p)) != was]
    if changed:
        raise ProblemRepoModified(
            f"a read-only path changed{' during ' + context if context else ''} — refusing to trust any "
            f"further results:\n  " + "\n  ".join(changed) +
            "\n  The problem repository must not be modified. Variant code belongs under "
            "problems/<problem_id>/variants/.")
