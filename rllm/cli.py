"""RLLM CLI (Phase-1 MVP, no LLM yet).

  python -m rllm.cli seed    <work_dir>            # enqueue the initial screen wave (Rx experiments)
  python -m rllm.cli work    <work_dir> [--device 0] [--dry-run]   # run this machine's worker loop
  python -m rllm.cli status  <work_dir>            # queue + ranking summary
  python -m rllm.cli promote <work_dir> --from screen --to refine --top-k 3

Adapter is hardcoded to Rx for now; onboarding/LLM layers come later (see DESIGN.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sibling packages importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm.interfaces import ExperimentSpec
from rllm.dispatcher import Queue, worker
from rllm.registry import Registry
from rllm import ladder
from adapters.rx.adapter import RxAdapter

ADAPTER = RxAdapter()

# Initial screen candidates for Rx — a few experiments drawn from the journal/idea catalog.
# (Later the LLM proposes these; hardcoded here to prove the loop.)
SEED_EXPERIMENTS = [
    ExperimentSpec("m1_bootstrap", "Bootstrap fix only (winner baseline).", {}),
    ExperimentSpec("m2_scale",     "+ sharper shortage scale 15000.",       {"SHORTAGE_SCALE": 15000}),
    ExperimentSpec("m3_soft",      "+ smooth survival bonus.",              {"SHORTAGE_SCALE": 15000, "SOFT_SCALE": 500}),
    ExperimentSpec("draincap",     "+ daily drain cap 0.5.",                {"MAX_SEND_FRAC": 0.5}),
]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rllm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("seed", "status"):
        sub.add_parser(name).add_argument("work_dir")
    w = sub.add_parser("work"); w.add_argument("work_dir"); w.add_argument("--device", default="0")
    w.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("promote"); p.add_argument("work_dir")
    p.add_argument("--from", dest="frm", required=True); p.add_argument("--to", required=True)
    p.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args(argv)

    if args.cmd == "seed":
        n = ladder.enqueue_experiments(ADAPTER, args.work_dir, SEED_EXPERIMENTS, "screen")
        print(f"enqueued {n} screen jobs for {len(SEED_EXPERIMENTS)} experiments -> {args.work_dir}")
    elif args.cmd == "work":
        worker(ADAPTER, args.work_dir, device=args.device, dry_run=args.dry_run)
        print("worker: queue drained.")
    elif args.cmd == "promote":
        ids = ladder.promote(ADAPTER, args.work_dir, args.frm, args.to, args.top_k)
        print(f"promoted to {args.to}: {ids}")
    elif args.cmd == "status":
        _status(args.work_dir)


def _status(work_dir):
    q, reg = Queue(work_dir), Registry(work_dir)
    print(f"goal: {ADAPTER.success_goal}")
    print(f"pending jobs: {q.pending_count()} | specs: {len(reg.all_specs())} | results: {len(reg.all_results())}")
    for fid in ("screen", "refine", "confirm"):
        ranked = ladder.rank(ADAPTER, work_dir, fid)
        if ranked:
            print(f"\n[{fid}] ranked best-first:")
            for eid, m in ranked:
                solved = " SOLVED" if ADAPTER.is_solved(m) else ""
                print(f"  {eid:28s} success={m.get('success_fraction', 0):.2f} "
                      f"short_days={m.get('mean_shortage_days', float('nan')):.2f}{solved}")


if __name__ == "__main__":
    main()
