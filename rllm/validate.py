"""Deterministic validation of experiment specs — the authorization layer.

Design rule §2.1: *models advise; deterministic code authorizes*. Everything an LLM produces passes
through here before it can influence a command line, a filename, or a run. Nothing in this module
consults a model, and it never repairs a proposal — it accepts or rejects with reasons (which the
controller feeds back to the actor as revision reasons).

What is checked:
  * the experiment id/slug is a strict slug (it becomes a filename and a shell-visible run PREFIX);
  * every config key is a knob the adapter DECLARED (unknown knobs rejected — no pass-through);
  * values match the knob's declared type / range / choices;
  * the actor may only set knobs marked `llm_may_change`;
  * `task_definition` knobs (they change what "solved" means) need explicit permission, even when
    both models agree;
  * fidelity rungs may only override `fidelity_safe` knobs (a rung must not alter realism).

An adapter that declares no knobs rejects all config, by design.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .interfaces import ExperimentSpec, KnobSpec, ProblemAdapter

# Actor-supplied slug (implementations.md §3.4).
SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
# Ids as stored/queued: a slug, optionally with the "@rung" suffix `promote()` appends.
ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}(@[a-z][a-z0-9_]{0,15})?$")


def check_slug(slug: Any) -> list[str]:
    if not isinstance(slug, str) or not SLUG_RE.match(slug):
        return [f"invalid slug {slug!r}: must match {SLUG_RE.pattern}"]
    return []


def check_id(exp_id: Any) -> list[str]:
    if not isinstance(exp_id, str) or not ID_RE.match(exp_id):
        return [f"invalid experiment id {exp_id!r}: must match {ID_RE.pattern}"]
    return []


def check_value(knob: KnobSpec, value: Any) -> list[str]:
    """Type/range/choice check for one knob value. Rejects bool-for-number (bool is an int in Python)
    and any non-scalar (a list/dict would be stringified into a command line)."""
    t = knob.value_type
    if t == "bool":
        if not isinstance(value, bool):
            return [f"{knob.name}: expected bool, got {type(value).__name__} ({value!r})"]
        return []
    if t in ("int", "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{knob.name}: expected {t}, got {type(value).__name__} ({value!r})"]
        if t == "int" and not float(value).is_integer():
            return [f"{knob.name}: expected int, got {value!r}"]
        errs = []
        if knob.minimum is not None and value < knob.minimum:
            errs.append(f"{knob.name}={value} below minimum {knob.minimum}")
        if knob.maximum is not None and value > knob.maximum:
            errs.append(f"{knob.name}={value} above maximum {knob.maximum}")
        return errs
    if t in ("string", "enum"):
        if not isinstance(value, str):
            return [f"{knob.name}: expected string, got {type(value).__name__} ({value!r})"]
        if knob.choices is not None and value not in knob.choices:
            return [f"{knob.name}={value!r} not in choices {knob.choices}"]
        return []
    return [f"{knob.name}: adapter declared unknown value_type {t!r}"]


def validate_config(adapter: ProblemAdapter, config: dict, *, llm_proposed: bool = True,
                    permitted_task_changes: tuple[str, ...] = ()) -> list[str]:
    knobs = adapter.declared_knobs()
    errs: list[str] = []
    for key, value in config.items():
        knob = knobs.get(key)
        if knob is None:
            errs.append(f"unknown knob {key!r} (not declared by adapter {adapter.name!r}); "
                        f"declared: {', '.join(sorted(knobs)) or '(none)'}")
            continue
        if llm_proposed and not knob.llm_may_change:
            errs.append(f"{key}: harness-only knob (llm_may_change=False)")
            continue
        if knob.category == "task_definition" and key not in permitted_task_changes:
            errs.append(f"{key}: task_definition knob — changes what 'solved' means; needs explicit "
                        f"human permission (permitted_task_changes)")
            continue
        errs += check_value(knob, value)
    return errs


def config_fingerprint(adapter: ProblemAdapter, config: dict) -> str:
    """Canonical hash of a config, ignoring knobs that a fidelity rung owns and values equal to the
    base recipe — so "the same experiment" is recognised however it was spelled (§3.4)."""
    base = adapter.base_config()
    rung_keys = {k for f in adapter.fidelity_levels() for k in f.overrides}
    effective = {k: v for k, v in {**base, **config}.items() if k not in rung_keys}
    return hashlib.sha256(json.dumps(effective, sort_keys=True, default=str).encode()).hexdigest()


def find_duplicate(adapter: ProblemAdapter, spec: ExperimentSpec,
                   existing: list[ExperimentSpec]) -> str | None:
    """The id of an already-registered experiment that is the same as `spec`, if any.

    Two ways to be a duplicate: reusing an id (records are never overwritten), or being the same
    configuration under a new name. Both mean paying twice for one answer, so both are refused here
    rather than left to the reviewer to notice."""
    slug = spec.id.split("@", 1)[0]
    for other in existing:
        if other.id == spec.id or other.id.split("@", 1)[0] == slug:
            return other.id
    mine = config_fingerprint(adapter, spec.config)
    for other in existing:
        if config_fingerprint(adapter, other.config) == mine:
            return other.id
    return None


def validate_spec(adapter: ProblemAdapter, spec: ExperimentSpec, *, llm_proposed: bool = True,
                  permitted_task_changes: tuple[str, ...] = (),
                  existing: list[ExperimentSpec] | None = None) -> list[str]:
    """All reasons `spec` must not run. Empty list = authorized."""
    errs = check_id(spec.id)
    if not isinstance(spec.hypothesis, str) or not spec.hypothesis.strip():
        errs.append(f"{spec.id}: hypothesis is required (say what the experiment tests)")
    if not isinstance(spec.config, dict):
        return errs + [f"{spec.id}: config must be a dict, got {type(spec.config).__name__}"]
    if existing:
        dup = find_duplicate(adapter, spec, existing)
        if dup:
            errs.append(f"{spec.id}: duplicate of already-registered experiment {dup!r} — same id or "
                        f"same effective config; propose something new instead")
    return errs + validate_config(adapter, spec.config, llm_proposed=llm_proposed,
                                  permitted_task_changes=permitted_task_changes)


def validate_adapter(adapter: ProblemAdapter) -> list[str]:
    """Self-check of the adapter's own declarations: rung overrides must be declared and fidelity-safe,
    and declared defaults must satisfy their own type/range. Run this before a session starts."""
    knobs = adapter.declared_knobs()
    errs = [e for k, knob in knobs.items()
            for e in ([f"knob table key {k!r} != KnobSpec.name {knob.name!r}"] if k != knob.name else [])
            + check_value(knob, knob.default)]
    for fid in adapter.fidelity_levels():
        for key, value in fid.overrides.items():
            knob = knobs.get(key)
            if knob is None:
                errs.append(f"fidelity {fid.name!r}: undeclared override {key!r}")
            elif not knob.fidelity_safe:
                errs.append(f"fidelity {fid.name!r}: {key!r} is not fidelity_safe "
                            f"(a rung must not change the problem)")
            else:
                errs += [f"fidelity {fid.name!r}: {e}" for e in check_value(knob, value)]
    return errs


def knob_table(adapter: ProblemAdapter, llm_only: bool = True) -> str:
    """Compact human/LLM-readable rendering of the whitelist, for prompts (so the actor proposes only
    knobs that can actually be authorized) and for `rllm validate`."""
    lines = []
    for name, k in sorted(adapter.declared_knobs().items()):
        if llm_only and (not k.llm_may_change or k.category == "task_definition"):
            continue
        rng = (f" choices={k.choices}" if k.choices is not None else
               "".join([f" min={k.minimum}" if k.minimum is not None else "",
                        f" max={k.maximum}" if k.maximum is not None else ""]))
        lines.append(f"  {name} ({k.value_type}, {k.category}) default={k.default!r}{rng}")
    return "\n".join(lines) or "  (none)"
