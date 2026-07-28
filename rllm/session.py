"""Bounded session controller — "explore for N hours, then hand back to me".

One session = plan a wave, run it, read the results, decide whether to go broader (more cheap ideas) or
deeper (promote the survivors to a costlier rung), until the deadline's wind-down point. Then it stops
launching, writes the handoff, and asks the models — within a fixed schema — how the user should
continue.

Two properties are worth stating because they are what make it safe to leave running:

  * The deadline, the run cap and the LLM-call cap are enforced by this module, not by the models
    (§2.3). A model cannot request more time; it can only recommend that the human grant it.
  * Compute escalates. Every wave prefers the cheapest useful work: a new screening wave is cheap and
    broad, and a promotion is only started when its conservative estimate fits in the time left. So
    early wall-clock goes on many short runs, and late wall-clock on the few candidates that earned it.

State is persisted after every transition, so a killed session resumes instead of restarting.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import integrity, ladder
from .brief import ProblemBrief, format_duration
from .budget import STARTUP_OVERHEAD_SECONDS, Estimator, Ledger
from .dispatcher import Queue, worker
from .interfaces import ProblemAdapter
from .llm import controller as llm_controller
from .llm.backend import LLMBackend, Usage
from .registry import Registry

STATES = ("initializing", "planning", "executing", "evaluating", "confirming", "winding_down")
TERMINAL = ("solved", "stalled", "budget_exhausted", "deadline_reached", "no_work_fits", "failed")

STALL_WAVES = 3                  # waves with no improvement in the best result before declaring a stall
PROMOTION_TOP_K = {1: 3, 2: 2}   # rung index -> how many candidates get promoted into it
HANDOFF_LLM_RESERVE = 2          # calls kept back so the write-up always gets to happen
CALLS_PER_PROPOSAL = 2           # one actor call + one reviewer call


@dataclass
class SessionState:
    session_id: str
    problem_id: str
    brief_sha256: str
    work_dir: str
    device: str
    state: str = "initializing"
    terminal_reason: str = ""
    started_at: str = ""
    finish_at: str = ""              # the deadline the user asked for
    ended_at: str = ""
    wave: int = 0
    waves_without_improvement: int = 0
    best_key: list[float] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)      # audit trail of what each wave did
    ledger: dict = field(default_factory=lambda: Ledger().as_dict())
    recommendation: dict = field(default_factory=dict)      # the models' continuation advice

    def as_dict(self) -> dict:
        return asdict(self)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def session_dir(work_dir, session_id: str) -> Path:
    return Path(work_dir) / "sessions" / session_id


class Session:
    def __init__(self, adapter: ProblemAdapter, brief: ProblemBrief, work_dir, device: str = "0",
                 actor: LLMBackend | None = None, reviewer: LLMBackend | None = None,
                 session_id: str | None = None, log: Callable[[str], None] = print,
                 cheap_run_seconds: float | None = None, startup_overhead: float | None = None,
                 handoff_actor: LLMBackend | None = None, handoff_reviewer: LLMBackend | None = None):
        self.adapter, self.brief, self.work_dir, self.device = adapter, brief, work_dir, device
        self.actor, self.reviewer = actor, reviewer
        # Model tiering: `actor` proposes (cheap tier is fine — short structured output, then adversarially
        # reviewed), while review and the handoff write-up stay on the strong tier. Falls back to the same
        # backend when no separate one is given, so untiered use is unchanged.
        self.handoff_actor = handoff_actor or actor
        self.handoff_reviewer = handoff_reviewer or reviewer
        self.log = log
        self.budget = brief.session_budget
        self.registry, self.queue = Registry(work_dir), Queue(work_dir)
        cfg = brief.adapter.config or {}
        overhead = (startup_overhead if startup_overhead is not None
                    else cfg.get("startup_overhead_seconds", STARTUP_OVERHEAD_SECONDS))
        self.estimator = Estimator(adapter, work_dir,
                                   cheap_run_seconds if cheap_run_seconds is not None
                                   else cfg.get("cheap_run_seconds", 1800.0),
                                   startup_overhead=overhead)
        self.state = self._load_or_new(session_id)
        self.ledger = Ledger(**self.state.ledger)
        self.rungs = [f.name for f in adapter.fidelity_levels()]

    def _backends(self) -> list[LLMBackend]:
        seen, out = set(), []
        for backend in (self.actor, self.reviewer, self.handoff_actor, self.handoff_reviewer):
            if backend is not None and id(backend) not in seen:
                seen.add(id(backend))
                out.append(backend)
        return out

    def _spent(self) -> Usage:
        total = Usage()
        for backend in self._backends():
            total.add(backend.usage)
        return total

    def _account(self, before: Usage) -> None:
        """Fold the tokens/cost of the calls just made into the ledger. Measured from what the CLIs
        report, so the cap that protects the subscription is enforced on real spend."""
        after = self._spent()
        delta = Usage(calls=after.calls - before.calls,
                      input_tokens=after.input_tokens - before.input_tokens,
                      output_tokens=after.output_tokens - before.output_tokens,
                      cache_read_tokens=after.cache_read_tokens - before.cache_read_tokens,
                      cost_usd=after.cost_usd - before.cost_usd,
                      cost_known=after.cost_known)
        self.ledger.record_usage(delta)

    # ---------------- persistence ----------------

    def _load_or_new(self, session_id: str | None) -> SessionState:
        if session_id:
            path = session_dir(self.work_dir, session_id) / "session.json"
            if path.exists():
                data = json.loads(path.read_text())
                st = SessionState(**data)
                st.ledger = data.get("ledger", Ledger().as_dict())
                return st
        sid = session_id or _now().strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
        started = _now()
        return SessionState(
            session_id=sid, problem_id=self.brief.problem_id, brief_sha256=self.brief.sha256(),
            work_dir=str(self.work_dir), device=self.device, started_at=_iso(started),
            finish_at=_iso(self.budget.deadline_from(started)))

    @property
    def dir(self) -> Path:
        return session_dir(self.work_dir, self.state.session_id)

    def _persist(self, state: str | None = None) -> None:
        if state:
            self.state.state = state
        self.state.ledger = self.ledger.as_dict()
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "session.json").write_text(json.dumps(self.state.as_dict(), indent=2) + "\n")

    # ---------------- clocks ----------------

    def _deadline(self) -> datetime:
        return datetime.fromisoformat(self.state.finish_at)

    def seconds_left(self) -> float:
        return (self._deadline() - _now()).total_seconds()

    def seconds_until_wind_down(self) -> float:
        return self.seconds_left() - self.budget.wind_down_seconds

    def _latest_start_monotonic(self) -> float:
        return time.monotonic() + max(0.0, self.seconds_until_wind_down())

    # ---------------- the loop ----------------

    def run(self) -> SessionState:
        if self.state.terminal_reason:
            self.log(f"session {self.state.session_id} already ended: {self.state.terminal_reason}")
            return self.state
        self.log(f"session {self.state.session_id}: {format_duration(self.seconds_left())} until "
                 f"review (wind-down at -{format_duration(self.budget.wind_down_seconds)}), "
                 f"caps: {self.budget.maximum_runs} runs / {self.budget.maximum_llm_calls} LLM calls")
        self._persist("planning")
        try:
            while True:
                stop = self._stop_reason()
                if stop:
                    self._finish(stop)
                    break
                self.state.wave += 1
                acted = self._wave()
                if not acted:
                    self._finish("no_work_fits" if self.seconds_until_wind_down() > 0
                                 else "deadline_reached")
                    break
                if self._stalled():
                    self._finish("stalled")
                    break
        except integrity.ProblemRepoModified as exc:
            # Every later result would be suspect, so this ends the session rather than the wave.
            self.log(f"STOPPING: {exc}")
            self.ledger.stopped_because.append(str(exc))
            self._finish("failed")
        except KeyboardInterrupt:
            self.log("interrupted — winding down")
            self._finish("deadline_reached")
        return self.state

    def _stalled(self) -> bool:
        """Has the search really plateaued?

        Three conditions, all required (§10.4: a stall must rest on full-fidelity results, never on cheap
        screens or a model's intuition):
          1. several consecutive waves that tried something NEW produced no improvement;
          2. there is nothing left to promote that would fit — because the right answer to "new ideas
             stopped helping" is to spend what remains confirming the best thing found, not to quit;
          3. the required fidelity actually has results, so the claim is about confirmed performance.

        Without (2) and (3) a session can quit while its leader has never been run at full fidelity,
        which is exactly the outcome the ladder exists to prevent.
        """
        if self.state.waves_without_improvement < STALL_WAVES:
            return False
        for idx in range(1, len(self.rungs)):
            if self._fits(self.rungs[idx]) and self._promotable(self.rungs[idx - 1], self.rungs[idx]):
                return False
        return bool(ladder.rank(self.adapter, self.work_dir,
                                self.brief.success_criterion.required_fidelity))

    def _stop_reason(self) -> str | None:
        """Deterministic stop conditions (§10.4), checked before every wave."""
        if self.seconds_until_wind_down() <= 0:
            return "deadline_reached"
        exhausted = self.ledger.exhausted(self.budget)
        if exhausted:
            return "budget_exhausted"
        solved = self._solved_evidence()
        if solved:
            return "solved"
        return None

    def _wave(self) -> bool:
        """One planning->executing->evaluating cycle. Returns False when nothing useful fits."""
        self._persist("planning")
        plan = self._plan()
        if not plan:
            return False
        for action in plan:
            self.state.actions.append({"wave": self.state.wave, **action})
            self.log(f"[wave {self.state.wave}] {action}")
        self._persist("executing")

        explored = any(a.get("explored") for a in plan)
        before = self._best_key()
        outcome = worker(
            self.adapter, self.work_dir, device=self.device,
            latest_start=self._latest_start_monotonic(),
            fit_seconds=self.estimator.seconds,
            per_run_timeout=lambda fid: (self.budget.per_run_timeout_seconds
                                         or self.estimator.hard_timeout(fid)),
            on_result=self._on_result,
        )
        self._persist("evaluating")
        after = self._best_key()
        improved = after > before
        # A stall means "trying new things stopped helping". Only waves that actually tried something
        # NEW count toward it: promoting the current leader is expected not to beat itself, and counting
        # those waves made the first live session declare a stall after four waves and four ideas.
        if improved:
            self.state.waves_without_improvement = 0
        elif explored:
            self.state.waves_without_improvement += 1
        self.state.best_key = list(after)
        self.ledger.waves += 1
        self.log(f"[wave {self.state.wave}] worker stopped: {outcome}; "
                 f"best={_fmt_key(after)}{' (improved)' if improved else ''}; "
                 f"{format_duration(max(0, self.seconds_until_wind_down()))} of exploration left")
        self._persist()
        return True

    def _on_result(self, result) -> None:
        self.ledger.record_run(result)
        self._persist()

    # ---------------- planning: cheap and broad first, deep only when it fits ----------------

    def _plan(self) -> list[dict]:
        """Choose this wave's work.

        Queued work always comes first — finish what was already authorized. After that the choice is
        between EXPLORING (a new cheap wave of ideas) and ESCALATING (promoting proven candidates to a
        costlier rung), and which one leads depends on how much of the window is left:

            plenty of time left  -> explore first, escalate only if there is nothing new to try
            past the halfway mark -> escalate first, explore only if nothing is promotable

        That is the "start fast, then increase compute" policy made explicit. Getting the order wrong is
        not a small matter: the first live session promoted from wave 2 onward, never screened a fifth
        configuration, and then declared a stall while the best knob was still sitting at the edge of the
        range it had tried.
        """
        pending = self.queue.pending_count()
        if pending:
            return [{"action": "run_queued", "jobs": pending}]

        early = self.seconds_until_wind_down() > 0.5 * self.budget.explore_seconds
        order = (self._explore, self._escalate) if early else (self._escalate, self._explore)
        for step in order:
            plan = step()
            if plan:
                return plan
        return []

    def _escalate(self) -> list[dict]:
        """Promote proven candidates one rung up, deepest rung first, if the estimate fits."""
        for idx in range(len(self.rungs) - 1, 0, -1):
            lower, upper = self.rungs[idx - 1], self.rungs[idx]
            if not self._fits(upper):
                continue
            candidates = self._promotable(lower, upper)
            if not candidates:
                continue
            k = min(PROMOTION_TOP_K.get(idx, 1), len(candidates),
                    self._max_jobs_that_fit(upper, reserve=self._promotion_reserve(idx)))
            if k < 1:
                continue
            # Promote exactly these ids: passing a top_k here would re-promote the leader every wave.
            ids = ladder.promote(self.adapter, self.work_dir, lower, upper, exp_ids=candidates[:k])
            if ids:
                return [{"action": "promote", "from": lower, "to": upper, "ids": ids,
                         "est_per_run": round(self.estimator.seconds(upper)), "explored": False}]
        return []

    def _explore(self) -> list[dict]:
        """Ask the actor for a fresh batch of cheap experiments, reviewer-gated as always."""
        cheapest = self.rungs[0]
        if not (self.actor and self.reviewer) or not self._fits(cheapest):
            return []
        n = self._screen_batch_size()
        if n < 1:
            return []
        # Keep calls back for the handoff. The write-up is the single most useful model call of the
        # session, and a run that spends its last call on one more screening wave hands the user a
        # deterministic stub instead of an explanation (seen live).
        note = self._llm_headroom()
        if note:
            if note not in self.ledger.stopped_because:
                self.ledger.stopped_because.append(note)
            return []
        before = self._spent()
        try:
            res = llm_controller.propose_and_review(
                self.actor, self.reviewer, self.adapter, self.work_dir, n=n,
                max_revisions=self.budget.maximum_actor_reviewer_revisions,
                permitted_task_changes=tuple(self.brief.permitted_task_changes))
        finally:
            self._account(before)
        if res.get("action") != "enqueued" or not res.get("ids"):
            # Nothing new was authorized (blocked, or every idea was a duplicate). Record the attempt
            # for the audit trail, but return no plan so the wave can still escalate instead of running
            # an empty queue.
            self.state.actions.append({"wave": self.state.wave, "action": "propose",
                                       "result": res.get("action"), "ids": [], "explored": False,
                                       "note": "nothing new was approved",
                                       "rejected": (res.get("rejected") or [])[:5]})
            return []
        return [{"action": "propose", "fidelity": cheapest, "ids": res["ids"],
                 "jobs": res.get("jobs", 0), "explored": True}]

    def _llm_headroom(self) -> str | None:
        """Why another proposal must not be made, or None. Reserves headroom for the handoff on every
        axis: a session that spends its last tokens on one more screening wave hands back a stub.

        The token reserve is estimated from what proposals have actually cost this session, because a
        handoff prompt is about as large as a proposal prompt (both carry the memory and the registry)."""
        budget = self.budget
        if self.ledger.llm_calls + CALLS_PER_PROPOSAL > budget.maximum_llm_calls - HANDOFF_LLM_RESERVE:
            return (f"stopped proposing at {self.ledger.llm_calls}/{budget.maximum_llm_calls} LLM calls, "
                    f"keeping {HANDOFF_LLM_RESERVE} back for the handoff")
        per_call = (self.ledger.llm_tokens / self.ledger.llm_calls) if self.ledger.llm_calls else 0.0
        if budget.maximum_llm_tokens is not None and per_call:
            need = per_call * (CALLS_PER_PROPOSAL + HANDOFF_LLM_RESERVE)
            if self.ledger.llm_tokens + need > budget.maximum_llm_tokens:
                return (f"stopped proposing at {self.ledger.llm_tokens:,}/"
                        f"{budget.maximum_llm_tokens:,} tokens, keeping enough for the handoff "
                        f"(~{per_call:,.0f} tokens/call)")
        per_cost = (self.ledger.llm_cost_usd / self.ledger.llm_calls) if self.ledger.llm_calls else 0.0
        if budget.maximum_llm_cost_usd is not None and per_cost:
            need = per_cost * (CALLS_PER_PROPOSAL + HANDOFF_LLM_RESERVE)
            if self.ledger.llm_cost_usd + need > budget.maximum_llm_cost_usd:
                return (f"stopped proposing at ${self.ledger.llm_cost_usd:.2f}/"
                        f"${budget.maximum_llm_cost_usd:.2f}, keeping enough for the handoff")
        return None

    def _promotable(self, lower: str, upper: str) -> list[str]:
        """Experiments completed at `lower` whose promotion to `upper` has not been registered yet."""
        existing = {s.id for s in self.registry.all_specs()}
        return [eid for eid, _ in ladder.rank(self.adapter, self.work_dir, lower)
                if f"{eid.split('@', 1)[0]}@{upper}" not in existing]

    def _fits(self, fidelity: str) -> bool:
        return self.estimator.seconds(fidelity) <= self.seconds_until_wind_down()

    def _max_jobs_that_fit(self, fidelity: str, reserve: int = 0) -> int:
        """How many experiments can be run at this rung within the time and run budget left, after
        holding back `reserve` runs for the rungs above it."""
        per_run = self.estimator.seconds(fidelity)
        seeds = max(1, len(next(f for f in self.adapter.fidelity_levels()
                                if f.name == fidelity).seeds))
        by_time = int(max(0.0, self.seconds_until_wind_down()) // (per_run * seeds))
        by_runs = max(0, self.ledger.runs_remaining(self.budget) - reserve) // seeds
        return max(0, min(by_time, by_runs))

    def _promotion_reserve(self, from_index: int = 0) -> int:
        """Runs to keep back so ONE candidate sitting at rung `from_index` can still be carried all the
        way to the top of the ladder.

        Applied at every rung, not just to screening: a live session spent its whole 12-run cap on
        screening (no promotions at all), and once that was fixed it spent the runs promoting three
        candidates to the middle rung and never confirmed any of them. Each rung has to leave the rungs
        above it enough room to finish the job."""
        return sum(len(f.seeds) for f in self.adapter.fidelity_levels()[from_index + 1:])

    def _screen_batch_size(self) -> int:
        """How many cheap experiments to propose: as many as comfortably fit in the time AND the run
        budget left, holding back the promotion reserve, and capped so one wave never eats the session."""
        by_time = self._max_jobs_that_fit(self.rungs[0])
        remaining = self.ledger.runs_remaining(self.budget)
        if not self.estimator.is_observed(self.rungs[0]):
            # Nothing has been timed yet, so every "does this fit before the deadline?" answer rests on
            # the user's guess. Spend ONE run calibrating before committing a full wave to it: if the
            # guess was 30m and the truth is 4h, this is the difference between one wasted run and four.
            return min(1, by_time, remaining)
        # Hold back the promotion reserve, but never so much that no screening can start at all: with a
        # run cap below the reserve the session would otherwise do nothing whatsoever.
        reserve = min(self._promotion_reserve(), max(0, remaining - 1))
        room = min(by_time, max(0, remaining - reserve))
        return max(0, min(4, room // 2 if room > 2 else room))

    # ---------------- evidence ----------------

    def _best_key(self) -> tuple:
        keys = [self.adapter.sort_key(m) for fid in self.rungs
                for _, m in ladder.rank(self.adapter, self.work_dir, fid)]
        return max(keys) if keys else ()

    def best_at(self, fidelity: str) -> tuple[str, dict] | None:
        ranked = ladder.rank(self.adapter, self.work_dir, fidelity)
        return ranked[0] if ranked else None

    def _solved_evidence(self) -> str | None:
        """Independent, full-fidelity evidence only (§2.5): the criterion must hold at the required
        fidelity over at least the required number of seeds. Cheap-rung wins never end a session."""
        need = self.brief.success_criterion
        for eid, metrics in ladder.rank(self.adapter, self.work_dir, need.required_fidelity):
            if metrics.get("n_seeds", 0) < need.minimum_training_seeds:
                continue
            if self.brief.is_solved(metrics):
                return f"{eid} meets [{need.describe()}]"
        return None

    # ---------------- wind-down ----------------

    def _finish(self, reason: str) -> None:
        self._persist("winding_down")
        self.state.terminal_reason = reason
        self.state.ended_at = _iso(_now())
        self.log(f"winding down: {reason}")
        if reason == "solved":
            self.state.recommendation = {"recommendation": "solved",
                                         "reasoning": self._solved_evidence() or "",
                                         "source": "deterministic"}
        else:
            self.state.recommendation = self._ask_for_continuation(reason)
        self._persist(reason)
        from . import report                       # local import: report reads session state
        path = report.write_handoff(self, reason)
        self.log(f"handoff: {path}")

    def _ask_for_continuation(self, reason: str) -> dict:
        """Ask the actor how to continue and have the reviewer check it. Bounded to a fixed schema, and
        purely advisory: it cannot extend the session. Falls back to a deterministic recommendation when
        no backend is available or the models fail."""
        fallback = _deterministic_recommendation(self, reason)
        actor, reviewer = self.handoff_actor, self.handoff_reviewer
        if not (actor and reviewer):
            return fallback
        if self.ledger.llm_calls + 2 > self.budget.maximum_llm_calls:
            fallback["note"] = "LLM-call cap reached before the continuation question could be asked"
            return fallback
        before = self._spent()
        try:
            from .llm import prompts
            from .llm.backend import parse_json
            evidence = report_evidence(self)
            raw = actor.ask(prompts.HANDOFF_SYSTEM,
                            prompts.handoff_user(self.brief, evidence, reason,
                                                 llm_controller.load_memory(self.work_dir)))
            rec = parse_json(raw)
            crit = parse_json(reviewer.ask(prompts.HANDOFF_REVIEW_SYSTEM,
                                           prompts.handoff_review_user(rec, evidence, reason)))
            self._account(before)
            rec = _clean_recommendation(rec)
            rec["reviewer_verdict"] = str(crit.get("verdict", "")).lower()
            rec["reviewer_notes"] = crit.get("reasons") or []
            rec["source"] = (f"actor={actor.name}({actor.model or 'default'}), "
                             f"reviewer={reviewer.name}({reviewer.model or 'default'})")
            rec["deterministic_view"] = fallback["recommendation"]
            llm_controller._log(self.work_dir, {"stage": "handoff", "session": self.state.session_id,
                                               "terminal_reason": reason, "recommendation": rec,
                                               "prompt_revision": prompts.PROMPT_REVISION,
                                               "method_sha": prompts.method_sha()})
            return rec
        except Exception as exc:                                    # noqa: BLE001
            self._account(before)
            fallback["note"] = f"models could not be consulted: {exc}"
            return fallback


VALID_RECOMMENDATIONS = ("more_time", "needs_help", "needs_input", "ready_to_confirm", "solved",
                         "stalled", "abandon")


def _clean_recommendation(rec: dict) -> dict:
    """Keep only the fields of the agreed schema, and force the verdict into the enum — a free-text
    recommendation would be unactionable and unauditable."""
    value = str(rec.get("recommendation", "")).strip().lower().replace(" ", "_")
    return {
        "recommendation": value if value in VALID_RECOMMENDATIONS else "needs_input",
        "recommendation_raw": rec.get("recommendation"),
        "confidence": str(rec.get("confidence", "unknown")).lower(),
        "reasoning": str(rec.get("reasoning", ""))[:4000],
        "estimated_additional_time": str(rec.get("estimated_additional_time", ""))[:32],
        "next_experiments": rec.get("next_experiments") or [],
        "questions_for_human": rec.get("questions_for_human") or [],
        "requests_requiring_authority": rec.get("requests_requiring_authority") or [],
    }


def _deterministic_recommendation(session: "Session", reason: str) -> dict:
    """What the harness itself concludes from the numbers, with no model involved. Always recorded
    alongside the models' view so the two can be compared in the handoff."""
    top = session.rungs[-1]
    best_top = session.best_at(top)
    if reason == "solved":
        rec = "solved"
    elif reason == "stalled":
        rec = "needs_help"
    elif best_top and session.brief.unmet(best_top[1]) == []:
        rec = "solved"
    elif reason in ("deadline_reached", "budget_exhausted"):
        rec = "more_time"
    elif reason == "no_work_fits":
        rec = "more_time"
    else:
        rec = "needs_input"
    unmet = session.brief.unmet(best_top[1]) if best_top else [c.describe() for c
                                                               in session.brief.success_criterion
                                                               .all_constraints()]
    return {"recommendation": rec, "confidence": "unknown", "source": "deterministic",
            "reasoning": (f"terminal reason {reason}; unmet at '{top}': "
                          f"{', '.join(unmet) or 'nothing'}"),
            "estimated_additional_time": "", "next_experiments": [],
            "questions_for_human": [], "requests_requiring_authority": []}


def report_evidence(session: "Session") -> dict[str, Any]:
    """Compact, factual view of the session for prompts and the handoff: rankings per rung, budget
    consumption, and what the success criterion still needs."""
    specs = {s.id: s for s in session.registry.all_specs()}
    rung_keys = {k for f in session.adapter.fidelity_levels() for k in f.overrides}
    rungs = {}
    for fid in session.rungs:
        # The configs go in alongside the metrics: a recommendation like "search around the leader" is
        # unactionable without the leader's actual knob values (a live session called this out).
        rungs[fid] = [{"id": eid, "metrics": {k: round(v, 4) for k, v in m.items()},
                       "config": {k: v for k, v in (specs[eid].config if eid in specs else {}).items()
                                  if k not in rung_keys},
                       "solved": session.brief.is_solved(m)}
                      for eid, m in ladder.rank(session.adapter, session.work_dir, fid)[:10]]
    return {
        "problem_id": session.brief.problem_id,
        "goal": session.brief.goal,
        "success_criterion": session.brief.success_criterion.describe(),
        "primary_metric": f"{session.brief.primary_metric} ({session.brief.primary_direction})",
        "rankings_by_fidelity": rungs,
        "waves": session.state.wave,
        "waves_without_improvement": session.state.waves_without_improvement,
        "runs_launched": session.ledger.runs_launched,
        "runs_failed": session.ledger.runs_failed,
        "approx_gpu_hours": round(session.ledger.gpu_hours, 2),
        "llm_calls": session.ledger.llm_calls,
        "llm_tokens": session.ledger.llm_tokens,
        "llm_cost_usd": (round(session.ledger.llm_cost_usd, 4)
                         if session.ledger.llm_cost_known else "partly unreported"),
        "llm_token_cap": session.budget.maximum_llm_tokens,
        "run_cap": session.budget.maximum_runs,
        "explored_for": format_duration(session.budget.explore_seconds),
        "observed_run_seconds": {fid: round(session.estimator.seconds(fid))
                                 for fid in session.rungs},
        "estimates_are_observed": {fid: session.estimator.is_observed(fid) for fid in session.rungs},
        "realism_constraints": session.brief.realism_constraints,
        "forbidden_task_changes": session.brief.forbidden_task_changes,
    }


def _fmt_key(key: tuple) -> str:
    return "(" + ", ".join(f"{v:.3g}" for v in key) + ")" if key else "(no results yet)"
