"""Prompt builders for the actor (propose) and reviewer (critique) roles.

Kept as plain functions returning (system, user) so they're easy to inspect/version. The actor and
reviewer are DIFFERENT roles (and ideally different models) — the reviewer's job is to catch waste,
duplication, and unsafe/over-eager decisions before any compute is spent.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

_JSON_ONLY = "Respond with ONLY a single JSON object, no prose, no code fences."

METHOD_PATH = Path(__file__).resolve().parent.parent.parent / "METHOD.md"
PROMPT_REVISION = 3          # bump when a prompt template changes; recorded in the decision log (§7.5)


@lru_cache(maxsize=1)
def method_text() -> str:
    """The standing research instructions (METHOD.md), injected into every actor/reviewer system prompt.

    Method lives in a file rather than in these strings so it can be edited, reviewed and diffed without
    touching code — and so the exact wording behind any decision is recoverable from its hash."""
    try:
        return METHOD_PATH.read_text().strip()
    except OSError:
        return ""


@lru_cache(maxsize=1)
def method_sha() -> str:
    return hashlib.sha256(method_text().encode()).hexdigest()[:12] if method_text() else "none"


def _with_method(system: str) -> str:
    """Append the standing method to a role's system prompt. The role comes first: METHOD.md is how to
    do the work, the role is what this particular call must return."""
    method = method_text()
    return f"{system}\n\n===== STANDING RESEARCH METHOD (METHOD.md) =====\n{method}" if method else system


PROPOSE_SYSTEM_ROLE = (
    "You are an RL research planner for an automated experimentation harness. "
    "You propose the next batch of experiments to try on a problem, given its living memory "
    "(problem.md = understanding + realism constraints; journal.md = what's been tried, results, and "
    "the idea backlog) and the current experiment registry. Prefer high-value untried ideas from the "
    "backlog; avoid duplicating experiments already run; respect the modelling-realism constraints in "
    "problem.md (e.g. do not shorten the episode horizon to fake speed). Each experiment's `config` is "
    "a dict of recipe knob overrides (e.g. SHORTAGE_SCALE, SOFT_SCALE, TD_N_STEPS, MAX_SEND_FRAC). "
    + _JSON_ONLY
)

REVIEW_SYSTEM_ROLE = (
    "You are an adversarial reviewer for an automated RL harness. Independently critique a proposed "
    "batch of experiments BEFORE any compute is spent. Block clearly wasteful, duplicate, unsafe, or "
    "realism-violating proposals; ask to revise ONLY when the fix is something the proposal itself can "
    "express; approve what is worth the GPU time. Remember hard-won cautions: seed variance is high "
    "(single seed only screens), convergence is late (do not assume early plateau), cheap proxies can "
    "mislead.\n"
    "SCOPE: training budget, seed counts, evaluation cadence, checkpoint selection and promotion are "
    "owned by the harness's fidelity ladder, NOT by the proposal — the actor cannot change them. Never "
    "block or ask to revise because a screening wave is short or single-seed; that is by design and the "
    "harness re-runs survivors at higher fidelity. Record such concerns in `risk_flags` instead. "
    "You cannot add experiments: new ideas must go back through the actor. " + _JSON_ONLY
)


HANDOFF_SYSTEM_ROLE = (
    "You are writing the end-of-session handoff for an automated RL experimentation harness. The "
    "exploration window the human granted has ended. Your job is to tell that human, honestly and "
    "concisely, what was learned and what should happen next.\n"
    "Ground every claim in the evidence given. Distinguish what the numbers show from what you infer. "
    "Do NOT claim success unless the stated success criterion is met at the required fidelity and seed "
    "count — a good cheap-rung result is a lead, not a result. If the evidence is thin, say so; "
    "recommending 'needs_input' or 'needs_help' is a better answer than a confident guess. You cannot "
    "grant yourself more time: 'more_time' is a request for the human to approve. " + _JSON_ONLY
)

HANDOFF_REVIEW_SYSTEM_ROLE = (
    "You are the adversarial reviewer of an end-of-session handoff written by another model. Check it "
    "against the evidence: is any claim overstated, is success claimed without full-fidelity multi-seed "
    "evidence, is a known caution (seed variance, late convergence, cheap-proxy transfer) ignored, is "
    "the recommendation consistent with the numbers, and are the proposed next experiments actually "
    "informative? Approve only if a human could act on it safely. " + _JSON_ONLY
)


# Exported system prompts = role + standing method. Every model call in the harness uses one of these.
PROPOSE_SYSTEM = _with_method(PROPOSE_SYSTEM_ROLE)
REVIEW_SYSTEM = _with_method(REVIEW_SYSTEM_ROLE)
HANDOFF_SYSTEM = _with_method(HANDOFF_SYSTEM_ROLE)
HANDOFF_REVIEW_SYSTEM = _with_method(HANDOFF_REVIEW_SYSTEM_ROLE)


def handoff_user(brief, evidence: dict, terminal_reason: str, memory_text: str) -> str:
    schema = ('{"recommendation":"more_time|needs_help|needs_input|ready_to_confirm|solved|stalled|'
              'abandon","confidence":"high|medium|low","reasoning":"a few sentences",'
              '"estimated_additional_time":"e.g. 12h","next_experiments":[{"id":"slug",'
              '"hypothesis":"one line","config":{"KNOB":value}}],"questions_for_human":["..."],'
              '"requests_requiring_authority":["..."]}')
    return "\n\n".join([
        f"## The problem\n{brief.description}\n\nGoal (user's words): {brief.goal}",
        f"## Success criterion (unchanged, human-owned)\n{brief.success_criterion.describe()}",
        f"## Why the session ended\n{terminal_reason}",
        f"## Evidence from this session (facts from the registry)\n{json.dumps(evidence, indent=2)}",
        f"## Problem memory\n{memory_text}",
        "## Task\nRecommend how the human should continue.\n"
        "- 'more_time': the current direction is working and just needs more wall-clock — say how much "
        "and what it will be spent on.\n"
        "- 'ready_to_confirm': a candidate looks good but lacks full-fidelity multi-seed evidence.\n"
        "- 'needs_input': a decision is required that is not yours (realism, what counts as solved, "
        "which trade-off matters).\n"
        "- 'needs_help': progress is blocked on something the harness cannot do (a missing knob, a "
        "code change in the problem repo, a broken runner, a bad reward).\n"
        "- 'stalled': the search has plateaued and repeating it will not help.\n"
        "- 'solved' / 'abandon': only with the evidence to justify it.\n"
        "`requests_requiring_authority` is for anything a human must authorize (changing the solved "
        "criterion, relaxing a realism constraint, editing the problem repo, a big compute ask).",
        f"## Output schema\n{schema}",
    ])


def handoff_review_user(recommendation: dict, evidence: dict, terminal_reason: str) -> str:
    schema = '{"verdict":"approve|revise|block","reasons":["..."],"risk_flags":["..."]}'
    return "\n\n".join([
        f"## Why the session ended\n{terminal_reason}",
        f"## Evidence (facts from the registry)\n{json.dumps(evidence, indent=2)}",
        f"## Handoff to review\n{json.dumps(recommendation, indent=2)}",
        "## Task\nReview it for overstatement and for consistency with the evidence.",
        f"## Output schema\n{schema}",
    ])


def base_config_text(base: dict) -> str:
    """The base recipe, spelled out: without it a model re-states base values (a no-op experiment) or
    proposes 'changing' a knob that is already set that way. Seen live on the first real run."""
    if not base:
        return "  (none — every knob is at the runner's own default)"
    return ("\n".join(f"  {k} = {v!r}" for k, v in sorted(base.items()))
            + "\n  Every run already uses these, so `config` should contain ONLY your deltas: repeating a "
              "base value buys a duplicate of the current base, and omitting a knob keeps the base value.")


def propose_user(memory_text: str, registry_summary: str, knob_table: str, ladder_text: str,
                 base_text: str, n: int, revise_reasons: list[str] | None = None) -> str:
    schema = '{"experiments":[{"id":"short_slug","hypothesis":"one line","config":{"KNOB":value}}]}'
    parts = [
        f"## Problem memory\n{memory_text}",
        f"## Current registry (already tried / queued)\n{registry_summary}",
        f"## Base recipe applied to every run\n{base_text}",
        f"## Fidelity ladder (harness-owned — not yours to set)\n{ladder_text}",
        f"## Knobs you may set (the ONLY ones accepted; anything else is rejected automatically)\n"
        f"{knob_table}",
        f"## Task\nPropose up to {n} experiments to run NEXT at the cheap 'screen' fidelity.\n"
        f"Hard requirements: `id` matches ^[a-z][a-z0-9_-]{{0,63}}$; every `config` key is from the knob "
        f"list above with a value inside its declared range/choices; training-budget and evaluation "
        f"cadence are owned by the fidelity ladder, so do not set them.",
        f"## Output schema\n{schema}",
    ]
    if revise_reasons:
        parts.insert(0, "## Reviewer asked you to revise your previous proposal for these reasons:\n"
                     + "\n".join(f"- {r}" for r in revise_reasons))
    return "\n\n".join(parts)


def review_user(proposed: dict, memory_text: str, registry_summary: str, ladder_text: str,
                base_text: str, validation_rejected: list[str] | None = None) -> str:
    schema = ('{"verdict":"approve|revise|block","approved_item_ids":["..."],"reasons":["..."],'
              '"required_changes":["..."],"risk_flags":["..."]}')
    parts = [
        f"## Problem memory\n{memory_text}",
        f"## Registry (already tried / queued)\n{registry_summary}",
        f"## Base recipe applied to every run\n{base_text}",
        f"## Fidelity ladder (harness-owned; the actor cannot change any of it)\n{ladder_text}",
        f"## Proposed batch to review\n{json.dumps(proposed, indent=2)}",
    ]
    if validation_rejected:
        parts.append("## Deterministic validation already REJECTED these (they cannot run whatever you "
                     "say; judge only the rest)\n"
                     + "\n".join(f"- {r}" for r in validation_rejected))
    parts += [
        "## Task\nReview the batch.\n"
        "- 'approve' = run the experiments listed in `approved_item_ids`. This list is REQUIRED and is "
        "never assumed to be everything: list every id you want run, and only ids from the batch above "
        "(invented ids are ignored). An empty list approves nothing.\n"
        "- 'revise' = the proposal itself must change; put the concrete fix in `required_changes`. Only "
        "use this for something the actor can actually express (which knobs, which values, which ids).\n"
        "- 'block' = none worth running (give reasons).\n"
        "- `risk_flags` = cautions to carry forward (e.g. proxy_transfer, insufficient_seeds, "
        "late_convergence) that must NOT by themselves downgrade the verdict.",
        f"## Output schema\n{schema}",
    ]
    return "\n\n".join(parts)


def ladder_text(fidelities) -> str:
    """Render the harness-owned ladder for both prompts, so neither model argues about compute it does
    not control (the first live run deadlocked exactly there: the reviewer kept demanding more
    generations and seeds, which are not proposable knobs)."""
    rungs = [f"  {f.name}: {len(f.seeds)} seed(s), overrides "
             + (", ".join(f"{k}={v}" for k, v in f.overrides.items()) or "(none)") for f in fidelities]
    return ("\n".join(rungs) + "\n"
            "Only the cheapest rung is being proposed now: it is TRIAGE, deliberately short and "
            "single-seed. The harness promotes survivors up the ladder, where budget and seed count "
            "increase, and 'solved' is only ever claimed from the top rung. Budget, seeds, evaluation "
            "cadence and promotion are therefore not proposable and not grounds for revision.")
