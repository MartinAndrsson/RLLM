"""Handoff report — what the human reads when the exploration window closes (implementations.md §11).

The report's one job is to be trustworthy at a glance, so it is written in three clearly separated
registers and never mixes them:

  FACTS        — read straight from the immutable registry (which runs happened, what they measured).
  ESTIMATES    — derived numbers that carry uncertainty (seed means, per-rung durations, ~GPU hours).
  INTERPRETATION — what the models say it means and what they recommend, labelled with which model said
                   it, whether the reviewer agreed, and what the harness itself concluded from the
                   numbers without any model.

It also always states what is NOT established: unmet success constraints, unconfirmed cheap-rung leads,
and failed runs. And it ends with the exact commands to resume, because the normal outcome of a bounded
session is another bounded session.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import ladder
from .brief import format_duration
from .registry import Registry

UNCONFIRMED = "unconfirmed (cheap rung — ranking may not transfer to full fidelity)"


def write_handoff(session, terminal_reason: str) -> Path:
    """Render `sessions/<id>/handoff.md` (plus the machine-readable evidence) and append a factual
    entry to the problem journal."""
    from .session import report_evidence
    evidence = report_evidence(session)
    path = session.dir / "handoff.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(session, terminal_reason, evidence))
    (session.dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    _append_journal(session, terminal_reason, evidence)
    # A blockage must be findable without knowing the session id.
    marker = Path(session.work_dir) / "BLOCKED.md"
    if terminal_reason in ("blocked", "failed"):
        marker.write_text(
            f"# This problem is blocked\n\n{session.state.blocked_reason or 'see the handoff'}\n\n"
            f"Full detail, including the failing runs' output: `{path}`\n\n"
            f"Fix the cause, then re-run:\n\n"
            f"    python -m rllm.cli solve {session.work_dir} --device {session.state.device}\n\n"
            f"Delete this file once resolved.\n")
    elif marker.exists():
        marker.unlink()          # the problem now runs; stale warnings mislead
    return path


def _render(session, terminal_reason: str, evidence: dict) -> str:
    st, brief, led = session.state, session.brief, session.ledger
    rec = st.recommendation or {}
    top_rung = session.rungs[-1]
    L: list[str] = []
    add = L.append

    add(f"# Handoff — {brief.problem_id} — session {st.session_id}")
    add("")
    add(f"**Ended:** `{terminal_reason}`  |  **started** {st.started_at}  |  **ended** "
        f"{st.ended_at or '(running)'}  |  **window** {format_duration(brief.session_budget.explore_seconds)}")
    add(f"**Brief:** `{st.brief_sha256[:12]}` (frozen for this session)  |  **repo:** "
        f"`{brief.problem_repo}`  |  **device:** `{st.device}`")
    add("")

    # ---- if it could not continue, that is the headline ----
    if terminal_reason in ("blocked", "failed"):
        add("## ⚠ STOPPED — I could not continue without you")
        add("")
        add(st.blocked_reason or "The session hit an error it could not work around.")
        add("")
        add("Nothing below is a result: the runs did not produce measurements. Fix the cause and re-run "
            "the same command — the queue and registry are intact, so nothing is lost.")
        failures = evidence.get("recent_failures") or []
        if failures:
            add("")
            add("### What actually failed")
            for f in failures:
                add("")
                add(f"**`{f['experiment']}` seed {f['seed']} (`{f['fidelity']}`)** — {f['error']}")
                if f.get("command") and f["command"] != "(command not recorded)":
                    add("")
                    add(f"Invoked as: `{f['command']}`")
                if f.get("log") and f["log"] not in ("(no log captured)", "(empty log)"):
                    add("")
                    add("```")
                    add(f["log"])
                    add("```")
        add("")

    # ---- the answer the user actually wants, up front ----
    add("## Recommendation")
    add("")
    verdict = rec.get("recommendation", "unknown")
    add(f"### `{verdict}`  (confidence: {rec.get('confidence', 'unknown')})")
    add("")
    add(f"*Source: {rec.get('source', 'unknown')}.* "
        + (f"Reviewer verdict on this handoff: **{rec['reviewer_verdict']}**. "
           if rec.get("reviewer_verdict") else "")
        + (f"The harness's own model-free reading of the numbers: **{rec['deterministic_view']}**."
           if rec.get("deterministic_view") else ""))
    if rec.get("deterministic_view") and rec["deterministic_view"] != verdict:
        add("")
        add(f"> ⚠ The models recommend `{verdict}` while the deterministic check says "
            f"`{rec['deterministic_view']}`. Trust the numbers below over either.")
    if rec.get("reasoning"):
        add("")
        add(rec["reasoning"])
    if rec.get("estimated_additional_time"):
        add("")
        add(f"**Additional time requested:** {rec['estimated_additional_time']} "
            f"(a request — the harness cannot grant it).")
    for title, key in (("Questions for you", "questions_for_human"),
                       ("Needs your authority", "requests_requiring_authority")):
        items = rec.get(key) or []
        if items:
            add("")
            add(f"**{title}:**")
            for item in items:
                add(f"- {item}")
    if rec.get("reviewer_notes"):
        add("")
        add("**Reviewer's objections to this handoff:**")
        for note in rec["reviewer_notes"]:
            add(f"- {note}")
    add("")

    # ---- was it solved? ----
    add("## Was it solved?")
    add("")
    add(f"**Criterion (human-owned, unchanged):** {brief.success_criterion.describe()}")
    add("")
    best_top = session.best_at(top_rung)
    if terminal_reason == "solved":
        add(f"**Yes — at the required fidelity.** {rec.get('reasoning', '')}")
    elif best_top:
        unmet = brief.unmet(best_top[1])
        add(f"**Not yet.** Best at `{top_rung}`: `{best_top[0]}` — still unmet: "
            f"{', '.join(unmet) if unmet else 'nothing (but seed count below the required minimum)'}.")
    else:
        add(f"**Not yet — no results at the required fidelity `{top_rung}` at all.** Everything below is "
            f"{UNCONFIRMED}.")
    add("")
    add("Note the criterion is a finite claim about a fixed number of seeds and evaluation episodes; it "
        "is not a claim that the policy can never fail.")
    add("")

    # ---- FACTS ----
    add("## Results (FACTS from the registry, ESTIMATES where averaged)")
    for fid in session.rungs:
        ranked = ladder.rank(session.adapter, session.work_dir, fid)
        add("")
        marker = "" if fid == top_rung else f"  — {UNCONFIRMED}"
        add(f"### `{fid}`{marker}")
        if not ranked:
            add("")
            add("_no completed runs_")
            continue
        keys = sorted({k for _, m in ranked for k in m} - {"n_seeds"})
        add("")
        add("| experiment | seeds | " + " | ".join(f"`{k}`" for k in keys) + " | meets criterion |")
        add("|---|---|" + "---|" * (len(keys) + 1))
        for eid, m in ranked:
            cells = " | ".join(f"{m[k]:.4g}" if k in m else "—" for k in keys)
            add(f"| `{eid}` | {int(m.get('n_seeds', 0))} | {cells} | "
                f"{'yes' if brief.is_solved(m) else 'no'} |")
    add("")
    add("Per-experiment values are means over that rung's seeds (ESTIMATE; seed variance on this "
        "problem is large enough that a single-seed row only screens).")
    add("")

    # ---- lineage and what the session did ----
    add("## What the session did")
    add("")
    add(f"{st.wave} wave(s); {led.waves} completed. Wave log:")
    add("")
    for a in st.actions:
        detail = {k: v for k, v in a.items() if k not in ("wave", "action")}
        add(f"- wave {a['wave']}: **{a.get('action')}** {json.dumps(detail)}")
    specs = Registry(session.work_dir).all_specs()
    lineage = [s for s in specs if s.parent]
    if lineage:
        add("")
        add("Promotions (lineage):")
        for s in lineage:
            add(f"- `{s.parent}` → `{s.id}` (`{s.fidelity}`)")
    add("")

    # ---- failures and what is missing ----
    failed = [r for r in Registry(session.work_dir).all_results() if r.status != "done"]
    add("## Failures, gaps and cautions")
    add("")
    if failed:
        add(f"{len(failed)} run(s) failed:")
        for r in failed[:20]:
            add(f"- `{r.exp_id}` seed {r.seed} (`{r.fidelity}`): {r.metrics.get('error', 'unknown')}")
    else:
        add("No failed runs.")
    add("")
    unobserved = [f for f, seen in evidence["estimates_are_observed"].items() if not seen]
    if unobserved:
        add(f"- Duration estimates for {', '.join(f'`{f}`' for f in unobserved)} are still the "
            f"user-supplied guess, not observation — fit-before-deadline decisions for those rungs are "
            f"only as good as that guess.")
    if len(session.rungs) > 1 and not ladder.rank(session.adapter, session.work_dir, top_rung):
        add(f"- Proxy validity is UNTESTED: nothing reached `{top_rung}`, so we do not know whether the "
            f"cheap-rung ranking transfers.")
    if brief.realism_constraints:
        add("- Realism constraints in force (unchanged this session): "
            + "; ".join(brief.realism_constraints))
    add("")

    # ---- budget ----
    add("## Budget consumed")
    add("")
    add(f"- runs: **{led.runs_launched}** launched ({led.runs_failed} failed) of a "
        f"{brief.session_budget.maximum_runs} cap")
    add(f"- run wall-clock: **{format_duration(led.run_seconds)}** (~{led.gpu_hours:.2f} device-hours; "
        f"ESTIMATE — one run per device, not measured utilisation)")
    add(f"- LLM calls: **{led.llm_calls}** of a {brief.session_budget.maximum_llm_calls} cap")
    cap = brief.session_budget.maximum_llm_tokens
    add(f"- LLM tokens: **{led.llm_tokens:,}**"
        + (f" of a {cap:,} cap" if cap else " (no token cap set)")
        + f" (+{led.llm_cache_read_tokens:,} cached reads, not counted against the cap)")
    add(f"- LLM spend: **${led.llm_cost_usd:.2f}**" if led.llm_cost_known else
        f"- LLM spend: **${led.llm_cost_usd:.2f} plus an unreported amount** (one backend reports tokens "
        f"but no price, so this is a lower bound)")
    for note in led.stopped_because:
        add(f"- limit hit: {note}")
    add("")

    # ---- next experiments proposed by the actor ----
    nxt = rec.get("next_experiments") or []
    if nxt:
        add("## Recommended next experiments (INTERPRETATION — not yet validated or reviewed as a wave)")
        add("")
        for e in nxt:
            if isinstance(e, dict):
                add(f"- `{e.get('id', '?')}`: {e.get('hypothesis', '')} → "
                    f"`{json.dumps(e.get('config', {}))}`")
        add("")
        add("These are ideas only. They pass through the knob whitelist and the reviewer like any other "
            "proposal when the next session plans its first wave.")
        add("")

    # ---- resume ----
    add("## Resume")
    add("")
    add("```bash")
    add(f"cd {Path(__file__).resolve().parent.parent}")
    add(f"python -m rllm.cli status {session.work_dir}")
    add(f"# continue exploring (fresh window, same brief):")
    add(f"python -m rllm.cli solve {session.work_dir} --device {st.device} --until "
        f"{format_duration(brief.session_budget.explore_seconds)}")
    add(f"# resume THIS session's state instead (same id, same ledger):")
    add(f"python -m rllm.cli solve {session.work_dir} --resume {st.session_id}")
    add("```")
    add("")
    add(f"To change the time window, the run cap, or the success criterion, edit "
        f"`{session.work_dir}/brief.json` (a human decision — the models may only request it).")
    add("")
    return "\n".join(L)


def _append_journal(session, terminal_reason: str, evidence: dict) -> None:
    """Append a strictly factual, machine-generated block to the problem journal so the next session
    resumes with this session's outcome in its memory.

    Deliberately facts-only (counts, rankings, terminal reason) with the models' interpretation left in
    handoff.md: the journal is read into every future prompt, and one session's speculation must not
    become the next session's premise.
    """
    journal = Path(session.work_dir) / "journal.md"
    top = session.rungs[-1]
    best = session.best_at(top)
    lines = [
        "",
        f"## Session {session.state.session_id} (auto-generated, facts only)",
        f"Ended `{terminal_reason}` after {session.state.wave} wave(s); "
        f"{session.ledger.runs_launched} run(s) launched, {session.ledger.runs_failed} failed, "
        f"~{session.ledger.gpu_hours:.2f} device-hours, {session.ledger.llm_calls} LLM calls.",
    ]
    for fid in session.rungs:
        ranked = ladder.rank(session.adapter, session.work_dir, fid)
        if ranked:
            top3 = "; ".join(f"{eid} ({session.brief.primary_metric}="
                             f"{m.get(session.brief.primary_metric, float('nan')):.3g}, "
                             f"n={int(m.get('n_seeds', 0))})" for eid, m in ranked[:3])
            lines.append(f"- `{fid}`: {top3}")
    lines.append(f"- required fidelity `{top}`: "
                 + (f"best `{best[0]}`, unmet: {', '.join(session.brief.unmet(best[1])) or 'nothing'}"
                    if best else "no results"))
    lines.append(f"- recommendation recorded in `sessions/{session.state.session_id}/handoff.md`: "
                 f"`{(session.state.recommendation or {}).get('recommendation', 'none')}`")
    with open(journal, "a") as f:
        f.write("\n".join(lines) + "\n")
