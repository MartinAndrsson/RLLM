"""Prompt builders for the actor (propose) and reviewer (critique) roles.

Kept as plain functions returning (system, user) so they're easy to inspect/version. The actor and
reviewer are DIFFERENT roles (and ideally different models) — the reviewer's job is to catch waste,
duplication, and unsafe/over-eager decisions before any compute is spent.
"""
from __future__ import annotations

import json

_JSON_ONLY = "Respond with ONLY a single JSON object, no prose, no code fences."

PROPOSE_SYSTEM = (
    "You are an RL research planner for an automated experimentation harness. "
    "You propose the next batch of experiments to try on a problem, given its living memory "
    "(problem.md = understanding + realism constraints; journal.md = what's been tried, results, and "
    "the idea backlog) and the current experiment registry. Prefer high-value untried ideas from the "
    "backlog; avoid duplicating experiments already run; respect the modelling-realism constraints in "
    "problem.md (e.g. do not shorten the episode horizon to fake speed). Each experiment's `config` is "
    "a dict of recipe knob overrides (e.g. SHORTAGE_SCALE, SOFT_SCALE, TD_N_STEPS, MAX_SEND_FRAC). "
    + _JSON_ONLY
)

REVIEW_SYSTEM = (
    "You are an adversarial reviewer for an automated RL harness. Independently critique a proposed "
    "batch of experiments BEFORE any compute is spent. Block clearly wasteful, duplicate, unsafe, or "
    "realism-violating proposals; ask to revise when a fix is concrete; approve only what is worth the "
    "GPU time. Remember hard-won cautions: seed variance is high (single seed only screens), "
    "convergence is late (do not assume early plateau), cheap proxies can mislead. " + _JSON_ONLY
)


def propose_user(memory_text: str, registry_summary: str, n: int, revise_reasons: list[str] | None = None) -> str:
    schema = '{"experiments":[{"id":"short_slug","hypothesis":"one line","config":{"KNOB":value}}]}'
    parts = [
        f"## Problem memory\n{memory_text}",
        f"## Current registry (already tried / queued)\n{registry_summary}",
        f"## Task\nPropose up to {n} experiments to run NEXT at the cheap 'screen' fidelity.",
        f"## Output schema\n{schema}",
    ]
    if revise_reasons:
        parts.insert(0, "## Reviewer asked you to revise your previous proposal for these reasons:\n"
                     + "\n".join(f"- {r}" for r in revise_reasons))
    return "\n\n".join(parts)


def review_user(proposed: dict, memory_text: str, registry_summary: str) -> str:
    schema = '{"verdict":"approve|revise|block","reasons":["..."],"keep_ids":["..."]}'
    return "\n\n".join([
        f"## Problem memory\n{memory_text}",
        f"## Registry (already tried / queued)\n{registry_summary}",
        f"## Proposed batch to review\n{json.dumps(proposed, indent=2)}",
        "## Task\nReview the batch. 'approve' = all worth running; 'revise' = fixable (give reasons + "
        "which ids to keep); 'block' = none worth running (give reasons).",
        f"## Output schema\n{schema}",
    ])
