"""RLLM CLI.

Start here (once per problem):
  ./scripts/rllm-onboard.sh                        # the interview -> writes <work_dir>/brief.json
  python -m rllm.cli validate  <work_dir>          # brief + adapter self-check, knob whitelist
  python -m rllm.cli solve     <work_dir> --device 0 --until 8h    # bounded session -> handoff.md

Individual steps (what `solve` orchestrates, runnable by hand):
  python -m rllm.cli propose <work_dir> [--actor claude] [--reviewer codex] [-n 4]  # enqueue, no compute
  python -m rllm.cli seed    <work_dir>            # enqueue a hardcoded wave instead (Rx only)
  python -m rllm.cli work    <work_dir> --device 0 [--dry-run]     # claim + run jobs
  python -m rllm.cli promote <work_dir> --from screen --to refine --top-k 3
  python -m rllm.cli status  <work_dir>            # brief, queue, rankings, sessions
  python -m rllm.cli handoff <work_dir>            # last session's handoff report
  python -m rllm.cli ask --backend codex           # smoke-test an LLM CLI backend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sibling packages (adapters/) importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm import ladder, problem as problem_mod
from rllm.brief import ProblemBrief, format_duration, parse_duration
from rllm.dispatcher import Queue, worker
from rllm.interfaces import ExperimentSpec
from rllm.registry import Registry

# Initial screen candidates for Rx, kept for `seed` (the LLM proposes these in a real session).
SEED_EXPERIMENTS = [
    ExperimentSpec("m1_bootstrap", "Bootstrap fix only (winner baseline).", {}),
    ExperimentSpec("m2_scale",     "+ sharper shortage scale 15000.",       {"SHORTAGE_SCALE": 15000}),
    ExperimentSpec("m3_soft",      "+ smooth survival bonus.",              {"SHORTAGE_SCALE": 15000, "SOFT_SCALE": 500}),
    ExperimentSpec("draincap",     "+ daily drain cap 0.5.",                {"MAX_SEND_FRAC": 0.5}),
]


def _resolve(work_dir, require_brief: bool = False):
    """(brief, adapter) for a work dir. Without a brief.json we fall back to the built-in Rx adapter so
    the pre-brief commands keep working; anything session-shaped requires a brief."""
    if problem_mod.brief_path(work_dir).exists():
        return problem_mod.load(work_dir)
    if require_brief:
        raise SystemExit(f"no brief.json in {work_dir} — run the interview first:\n"
                         f"  ./scripts/rllm-onboard.sh")
    from adapters.rx.adapter import RxAdapter
    return None, RxAdapter()


def _backends(args):
    """(actor, reviewer) or (None, None) with --no-llm. Different backends are required by default."""
    if getattr(args, "no_llm", False):
        return None, None
    from rllm.llm.backend import CLIBackend
    if args.actor == args.reviewer and not args.allow_same_model_review:
        raise SystemExit("actor and reviewer must be different backends (pass "
                         "--allow-same-model-review to override; independent review is the point)")
    mk = {"claude": CLIBackend.claude, "codex": CLIBackend.codex}
    return (mk[args.actor](timeout=args.timeout, model=args.actor_model),
            mk[args.reviewer](timeout=args.timeout, model=args.reviewer_model))


def _add_llm_flags(p):
    p.add_argument("--actor", default="claude", choices=["claude", "codex"])
    p.add_argument("--reviewer", default="codex", choices=["claude", "codex"])
    p.add_argument("--actor-model", default=None)
    p.add_argument("--reviewer-model", default=None)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--allow-same-model-review", action="store_true")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rllm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("seed", "status", "validate"):
        sub.add_parser(name).add_argument("work_dir")

    ob = sub.add_parser("onboard", help="write+validate a brief.json from interview answers")
    ob.add_argument("work_dir"); ob.add_argument("--answers", required=True)
    ob.add_argument("--force", action="store_true", help="overwrite an existing brief.json")

    sv = sub.add_parser("solve", help="run one bounded exploration session, then hand off")
    sv.add_argument("work_dir"); sv.add_argument("--device", default="0")
    sv.add_argument("--until", default=None, help="override the brief's window, e.g. 90m / 8h / 2d")
    sv.add_argument("--resume", default=None, metavar="SESSION_ID")
    sv.add_argument("--no-llm", action="store_true",
                    help="run only already-queued work and promotions (no proposals)")
    _add_llm_flags(sv)

    w = sub.add_parser("work"); w.add_argument("work_dir"); w.add_argument("--device", default="0")
    w.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("promote"); p.add_argument("work_dir")
    p.add_argument("--from", dest="frm", required=True); p.add_argument("--to", required=True)
    p.add_argument("--top-k", type=int, default=3)

    pr = sub.add_parser("propose"); pr.add_argument("work_dir"); pr.add_argument("-n", type=int, default=4)
    _add_llm_flags(pr)

    cp = sub.add_parser("compare", help="the deliverable table; --by VARIANT compares designs")
    cp.add_argument("work_dir"); cp.add_argument("--by", default=None, metavar="KNOB")
    cp.add_argument("--fidelity", default=None)
    cp.add_argument("--out", default=None, help="write the markdown here instead of stdout")

    hf = sub.add_parser("handoff"); hf.add_argument("work_dir")
    hf.add_argument("--session", default=None); hf.add_argument("--path-only", action="store_true")

    a = sub.add_parser("ask"); a.add_argument("--backend", default="claude", choices=["claude", "codex"])
    a.add_argument("--prompt", default='Reply with only this JSON: {"ok": true, "who": "<your model>"}')
    a.add_argument("--timeout", type=float, default=300.0)

    args = ap.parse_args(argv)

    if args.cmd == "onboard":
        _onboard(args)
    elif args.cmd == "solve":
        _solve(args)
    elif args.cmd == "seed":
        _, adapter = _resolve(args.work_dir)
        n = ladder.enqueue_experiments(adapter, args.work_dir, SEED_EXPERIMENTS, "screen",
                                       llm_proposed=False)
        print(f"enqueued {n} screen jobs for {len(SEED_EXPERIMENTS)} experiments -> {args.work_dir}")
    elif args.cmd == "work":
        _, adapter = _resolve(args.work_dir)
        outcome = worker(adapter, args.work_dir, device=args.device, dry_run=args.dry_run)
        print("worker: commands rendered; queue untouched (dry run)." if args.dry_run
              else f"worker: {outcome}.")
    elif args.cmd == "promote":
        _, adapter = _resolve(args.work_dir)
        ids = ladder.promote(adapter, args.work_dir, args.frm, args.to, args.top_k)
        print(f"promoted to {args.to}: {ids}")
    elif args.cmd == "propose":
        brief, adapter = _resolve(args.work_dir)
        from rllm.llm import controller
        actor, reviewer = _backends(args)
        res = controller.propose_and_review(
            actor, reviewer, adapter, args.work_dir, n=args.n,
            permitted_task_changes=tuple(brief.permitted_task_changes) if brief else ())
        print(f"actor={args.actor} reviewer={args.reviewer} -> {res.get('action')}: "
              f"{res.get('ids', res.get('verdict'))}")
        for r in res.get("rejected") or []:
            print(f"  rejected by validation: {r}")
        print("(review-gated; nothing trains until you run `work` or `solve`)")
    elif args.cmd == "validate":
        _validate(args.work_dir)
    elif args.cmd == "compare":
        from rllm import compare
        brief, adapter = _resolve(args.work_dir, require_brief=True)
        text = compare.render(adapter, brief, args.work_dir, fidelity=args.fidelity, by=args.by)
        if args.out:
            Path(args.out).write_text(text + "\n")
            print(f"wrote {args.out}")
        else:
            print(text)
    elif args.cmd == "handoff":
        _handoff(args)
    elif args.cmd == "ask":
        from rllm.llm.backend import CLIBackend
        mk = {"claude": CLIBackend.claude, "codex": CLIBackend.codex}
        txt = mk[args.backend](timeout=args.timeout).ask("You are terse. Answer with JSON only.",
                                                         args.prompt)
        print(f"[{args.backend}] {txt.strip()[:2000]}")
    elif args.cmd == "status":
        _status(args.work_dir)


# ---------------------------------------------------------------- onboarding

def _onboard(args):
    """Turn the interview's answers into a validated, frozen brief. The interview lives in bash
    (scripts/rllm-onboard.sh); all validation lives here."""
    answers = json.loads(Path(args.answers).read_text())
    dest = problem_mod.brief_path(args.work_dir)
    if dest.exists() and not args.force:
        raise SystemExit(f"{dest} already exists (pass --force to overwrite). A brief is meant to be "
                         f"frozen; edit it deliberately.")
    brief = ProblemBrief.from_dict(answers)
    problems = brief.problems()
    if problems:
        print("brief is not usable yet:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(2)
    brief.save(dest)
    print(f"wrote {dest}  (sha256 {brief.sha256()[:12]})")
    print(f"  problem   : {brief.problem_id} in {brief.problem_repo}")
    print(f"  metric    : {brief.primary_metric} ({brief.primary_direction})")
    print(f"  solved    : {brief.success_criterion.describe()}")
    print(f"  window    : {format_duration(brief.session_budget.explore_seconds)} "
          f"(wind-down {format_duration(brief.session_budget.wind_down_seconds)})")
    print(f"  adapter   : {brief.adapter.name}")
    _validate(args.work_dir)


# ---------------------------------------------------------------- session

def _solve(args):
    from rllm.session import Session
    brief, adapter = _resolve(args.work_dir, require_brief=True)
    if args.until:
        brief.session_budget.explore_seconds = parse_duration(args.until)
        brief.session_budget.wind_down_seconds = min(brief.session_budget.wind_down_seconds,
                                                    brief.session_budget.explore_seconds / 4)
    problems = _adapter_problems(adapter)
    if problems:
        print("refusing to start a session — adapter/brief inconsistent:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(2)
    actor, reviewer = _backends(args)
    session = Session(adapter, brief, args.work_dir, device=args.device, actor=actor,
                      reviewer=reviewer, session_id=args.resume)
    state = session.run()
    rec = state.recommendation or {}
    print()
    print(f"session {state.session_id} ended: {state.terminal_reason}")
    print(f"recommendation: {rec.get('recommendation', 'none')} "
          f"({rec.get('confidence', 'unknown')} confidence, {rec.get('source', 'n/a')})")
    if rec.get("reasoning"):
        print(f"  {rec['reasoning'][:600]}")
    for q in (rec.get("questions_for_human") or [])[:5]:
        print(f"  ? {q}")
    print(f"handoff: {session.dir / 'handoff.md'}")


def _handoff(args):
    sessions = sorted((Path(args.work_dir) / "sessions").glob("*/handoff.md"))
    if args.session:
        sessions = [p for p in sessions if p.parent.name == args.session]
    if not sessions:
        raise SystemExit(f"no handoff found under {args.work_dir}/sessions")
    path = sessions[-1]
    print(path if args.path_only else path.read_text())


# ---------------------------------------------------------------- checks / status

def _adapter_problems(adapter) -> list[str]:
    from rllm.validate import validate_adapter
    return validate_adapter(adapter)


def _validate(work_dir):
    from rllm.validate import knob_table, validate_spec
    brief, adapter = _resolve(work_dir)
    problems = list(_adapter_problems(adapter))
    if brief:
        problems += [f"brief: {p}" for p in brief.problems()]
        rungs = {f.name for f in adapter.fidelity_levels()}
        if brief.success_criterion.required_fidelity not in rungs:
            problems.append(f"brief: success_criterion.required_fidelity "
                            f"{brief.success_criterion.required_fidelity!r} is not one of the adapter's "
                            f"rungs {sorted(rungs)}")
        variants = getattr(adapter, "variants", lambda: [])()
        if not variants and not brief.test_command.strip() and brief.adapter.name == "brief":
            problems.append("nothing to run: no variants on disk and no test_command in the brief. "
                            "Author a design under problems/<problem_id>/variants/ (see "
                            "problems/README.md) or give the brief a test_command.")
        knobs = adapter.declared_knobs()
        for name in [*brief.permitted_task_changes, *brief.forbidden_task_changes]:
            if name not in knobs:
                problems.append(f"brief: task-change knob {name!r} is not declared by the adapter")
        print(f"brief    : {brief.problem_id} (sha256 {brief.sha256()[:12]})")
        print(f"  goal   : {brief.goal}")
        print(f"  solved : {brief.success_criterion.describe()}")
        print(f"  test   : {brief.test_command or '(adapter-defined)'}")
    else:
        problems += [f"seed spec: {e}" for spec in SEED_EXPERIMENTS
                     for e in validate_spec(adapter, spec, llm_proposed=False)]
        print(f"brief    : (none — using the built-in {adapter.name!r} adapter)")
    print(f"adapter  : {adapter.name}, {len(adapter.declared_knobs())} knobs declared, "
          f"rungs {[f.name for f in adapter.fidelity_levels()]}")
    designs = getattr(adapter, "variants", lambda: [])()
    if designs:
        print(f"designs  : {len(designs)} variant(s) — {', '.join(designs)}")
    print("knobs the actor may propose:")
    print(knob_table(adapter))
    if problems:
        print(f"\nFAIL ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nOK: declarations consistent, fidelity overrides fidelity-safe, brief usable.")


def _status(work_dir):
    brief, adapter = _resolve(work_dir)
    q, reg = Queue(work_dir), Registry(work_dir)
    if brief:
        print(f"problem: {brief.problem_id} — {brief.goal}")
        print(f"solved when: {brief.success_criterion.describe()}")
        print(f"window per session: {format_duration(brief.session_budget.explore_seconds)}")
    else:
        print(f"goal: {adapter.success_goal}")
    print(f"pending jobs: {q.pending_count()} | specs: {len(reg.all_specs())} | "
          f"results: {len(reg.all_results())}")
    for fid in [f.name for f in adapter.fidelity_levels()]:
        ranked = ladder.rank(adapter, work_dir, fid)
        if ranked:
            print(f"\n[{fid}] ranked best-first:")
            for eid, m in ranked:
                solved = " SOLVED" if adapter.is_solved(m) else ""
                extra = " ".join(f"{k}={m[k]:.4g}" for k in sorted(m) if k != "n_seeds")
                print(f"  {eid:28s} n={int(m.get('n_seeds', 0))} {extra}{solved}")
    sessions = sorted((Path(work_dir) / "sessions").glob("*/session.json"))
    if sessions:
        print("\nsessions:")
        for p in sessions[-5:]:
            s = json.loads(p.read_text())
            rec = (s.get("recommendation") or {}).get("recommendation", "-")
            print(f"  {s['session_id']}  {s.get('state'):12s} "
                  f"{s.get('terminal_reason') or '(running)':18s} waves={s.get('wave')} "
                  f"runs={s.get('ledger', {}).get('runs_launched')} rec={rec}")


if __name__ == "__main__":
    main()
