"""Tests for the piece that makes this more than hyperparameter search: the models design the
observation, action space and reward themselves, as code in THIS repository, against the user's repo as a
read-only simulator.

What must hold:
  * a variant directory is discovered and becomes an enum choice of the `VARIANT` knob, so a design is
    searched, compared and gated exactly like any other knob;
  * a variant is run with the harness contract (RUN_DIR, SEED, PROBLEM_REPO, knobs) and its metrics.json
    is the only thing believed;
  * a run that modifies the problem repo fails and stops the session — this is the property that makes
    letting a model write code acceptable at all;
  * `compare --by VARIANT` produces the design-against-design table.
"""
from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm import compare, integrity, ladder, problem as problem_mod
from rllm.brief import ProblemBrief
from rllm.dispatcher import worker
from rllm.interfaces import ExperimentSpec
from rllm.llm.backend import MockBackend
from rllm.registry import Registry
from rllm.session import Session
from rllm.validate import validate_spec
from adapters.brief_adapter import BriefAdapter

# A variant: reads knobs from the environment, imports the "simulator" from PROBLEM_REPO, writes only
# into RUN_DIR. `BONUS` is the design's own reward weight; `sim.score` is the read-only simulator.
VARIANT = """#!/usr/bin/env python3
import json, os, sys
sys.path.insert(0, os.environ["PROBLEM_REPO"])
import sim                                   # the user's read-only simulator
weight = float(os.environ.get("BONUS", "0"))
seed = int(os.environ.get("SEED", "1"))
score = sim.score(seed) + {shape} * weight
json.dump({{"score": round(score, 4)}}, open(os.path.join(os.environ["RUN_DIR"], "metrics.json"), "w"))
"""

# A hostile variant: writes into the problem repo. Must be caught.
SABOTEUR = """#!/usr/bin/env python3
import json, os
open(os.path.join(os.environ["PROBLEM_REPO"], "sim.py"), "a").write("\\n# tampered\\n")
json.dump({"score": 1.0}, open(os.path.join(os.environ["RUN_DIR"], "metrics.json"), "w"))
"""

SIM = "def score(seed):\n    return 0.5 + 0.01 * (seed % 3)\n"

KNOBS = [{"name": "BONUS", "value_type": "float", "default": 0.0, "category": "reward",
          "minimum": 0.0, "maximum": 1.0}]
FIDELITIES = [
    {"name": "screen", "overrides": {}, "seeds": [1], "cost_multiplier": 1.0},
    {"name": "confirm", "overrides": {}, "seeds": [1, 2, 3], "cost_multiplier": 2.0},
]


def build(tmp_path, variants: dict[str, str], git: bool = False) -> tuple[ProblemBrief, BriefAdapter, Path]:
    repo = tmp_path / "problem_repo"
    repo.mkdir(exist_ok=True)
    (repo / "sim.py").write_text(SIM)
    if git:
        for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t",
                                                    "commit", "-qm", "init"]):
            subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)
    vdir = tmp_path / "variants"
    for name, body in variants.items():
        d = vdir / name
        d.mkdir(parents=True, exist_ok=True)
        entry = d / "train.py"
        entry.write_text(body)
        entry.chmod(entry.stat().st_mode | stat.S_IXUSR)
        (d / "manifest.json").write_text(json.dumps({"id": name, "hypothesis": f"design {name}"}))
    brief = ProblemBrief.from_dict({
        "problem_id": "designed", "description": "a problem whose designs are code",
        "goal": "score >= 0.9", "problem_repo": str(repo),
        "primary_metric": "score", "primary_direction": "maximize", "metrics_file": "metrics.json",
        "success_criterion": {"primary": {"metric": "score", "operator": ">=", "value": 0.9},
                              "required_fidelity": "confirm", "minimum_training_seeds": 3},
        "adapter": {"name": "brief", "config": {
            "knobs": KNOBS, "fidelities": FIDELITIES, "variants_dir": str(vdir),
            "python": sys.executable, "cheap_run_seconds": 1.0, "startup_overhead_seconds": 0.5}},
        "session_budget": {"explore_seconds": 120.0, "wind_down_seconds": 1.0,
                           "maximum_runs": 30, "maximum_llm_calls": 30},
    })
    wd = tmp_path / "work"
    wd.mkdir(exist_ok=True)
    brief.save(problem_mod.brief_path(wd))
    return brief, BriefAdapter(brief), wd


def two_designs(tmp_path, git: bool = False):
    return build(tmp_path, {"plain": VARIANT.format(shape="0.0"),
                            "shaped": VARIANT.format(shape="1.0")}, git=git)


# ---------------------------------------------------------------- discovery

def test_variants_become_choices_of_a_declared_knob(tmp_path):
    _, adapter, _ = two_designs(tmp_path)
    knobs = adapter.declared_knobs()
    assert knobs["VARIANT"].value_type == "enum"
    assert knobs["VARIANT"].choices == ["plain", "shaped"]
    assert adapter.variants() == ["plain", "shaped"]


def test_a_variant_that_does_not_exist_is_rejected(tmp_path):
    _, adapter, _ = two_designs(tmp_path)
    errs = validate_spec(adapter, ExperimentSpec("try_it", "use a design nobody wrote",
                                                 {"VARIANT": "imaginary"}))
    assert any("not in choices" in e for e in errs)
    assert validate_spec(adapter, ExperimentSpec("real", "use a real design",
                                                 {"VARIANT": "shaped"})) == []


def test_directories_without_an_entry_point_are_not_variants(tmp_path):
    _, adapter, _ = two_designs(tmp_path)
    (adapter.variants_dir / "notes").mkdir()
    (adapter.variants_dir / "notes" / "README.md").write_text("just notes")
    (adapter.variants_dir / "_wip").mkdir()
    (adapter.variants_dir / "_wip" / "train.py").write_text("x")
    assert adapter.variants() == ["plain", "shaped"]        # no entry point / underscore-prefixed


def test_no_variants_dir_falls_back_to_command_mode(tmp_path):
    brief, adapter, _ = build(tmp_path, {})
    brief.test_command = "echo hi"
    assert adapter.variants() == [] and "VARIANT" not in adapter.declared_knobs()
    cmd = BriefAdapter(brief).build_command(ExperimentSpec("x", "h", {}), 1, str(tmp_path), "0")
    assert "echo hi" in cmd[2] and str(brief.problem_repo) in cmd[2]


# ---------------------------------------------------------------- running

def test_a_variant_runs_under_the_harness_contract(tmp_path):
    _, adapter, wd = two_designs(tmp_path)
    ladder.enqueue_experiments(adapter, wd, [
        ExperimentSpec("plain_design", "no shaping", {"VARIANT": "plain", "BONUS": 0.5}),
        ExperimentSpec("shaped_design", "shaping helps", {"VARIANT": "shaped", "BONUS": 0.5}),
    ], "screen")
    worker(adapter, wd, device="0")
    results = {r.exp_id: r for r in Registry(wd).all_results()}
    assert all(r.status == "done" for r in results.values()), results
    # plain ignores BONUS; shaped adds it -> the DESIGN changed the outcome, not just a knob.
    assert results["plain_design"].metrics["score"] == pytest.approx(0.51)
    assert results["shaped_design"].metrics["score"] == pytest.approx(1.01)


def test_the_variant_gets_the_knobs_and_writes_only_to_run_dir(tmp_path):
    _, adapter, wd = two_designs(tmp_path)
    cmd = adapter.build_command(ExperimentSpec("x", "h", {"VARIANT": "shaped", "BONUS": 0.25}),
                                seed=7, run_dir=str(tmp_path / "rd"), device="1")
    script = cmd[2]
    for expected in ("BONUS=0.25", "SEED=7", "CUDA_VISIBLE_DEVICES=1", "PROBLEM_REPO=", "RUN_DIR="):
        assert expected in script
    assert str(adapter.variants_dir / "shaped") in script      # runs from the variant's own directory
    before = integrity.fingerprint(adapter.brief.problem_repo)
    worker(adapter, wd, device="0")                            # nothing queued; just proves no side effects
    assert integrity.fingerprint(adapter.brief.problem_repo) == before


# ---------------------------------------------------------------- the read-only guarantee

@pytest.mark.parametrize("git", [False, True])
def test_a_run_that_touches_the_problem_repo_fails_and_stops_everything(tmp_path, git):
    """The property that makes model-written code acceptable: the simulator and the test cannot move."""
    _, adapter, wd = build(tmp_path, {"saboteur": SABOTEUR}, git=git)
    ladder.enqueue_experiments(adapter, wd, [
        ExperimentSpec("tamper", "writes into the problem repo", {"VARIANT": "saboteur"})], "screen")
    with pytest.raises(integrity.ProblemRepoModified, match="read-only path changed"):
        worker(adapter, wd, device="0")
    result = Registry(wd).all_results()[0]
    assert result.status == "failed" and "read-only path changed" in result.metrics["error"]
    assert ladder.rank(adapter, wd, "screen") == []            # never ranked, whatever it "scored"


def test_a_tampering_run_ends_the_session_as_failed(tmp_path):
    _, adapter, wd = build(tmp_path, {"saboteur": SABOTEUR})
    brief = problem_mod.load_brief(wd)
    ladder.enqueue_experiments(adapter, wd, [
        ExperimentSpec("tamper", "writes into the problem repo", {"VARIANT": "saboteur"})], "screen")
    session = Session(adapter, brief, wd, log=lambda *_: None, cheap_run_seconds=1.0,
                      startup_overhead=0.5)
    state = session.run()
    assert state.terminal_reason == "failed"
    assert any("read-only path changed" in n for n in session.ledger.stopped_because)
    assert (session.dir / "handoff.md").exists()               # a failure still hands off


def test_fingerprint_notices_content_edits_and_deletions(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1")
    first = integrity.fingerprint(repo)
    (repo / "a.py").write_text("x = 2")                        # same size, different content
    assert integrity.fingerprint(repo) != first
    (repo / "a.py").unlink()
    assert integrity.fingerprint(repo) != first


def test_git_fingerprint_notices_a_dirty_edit(tmp_path):
    _, adapter, _ = two_designs(tmp_path, git=True)
    repo = adapter.brief.problem_repo
    before = integrity.fingerprint(repo)
    assert before.startswith("git:")
    (Path(repo) / "sim.py").write_text(SIM + "# edited\n")
    assert integrity.fingerprint(repo) != before


def test_missing_path_is_reported_not_silently_ok(tmp_path):
    assert integrity.fingerprint(tmp_path / "nope") == "missing"


# ---------------------------------------------------------------- the deliverable table

def test_compare_by_variant_tables_designs_against_each_other(tmp_path):
    _, adapter, wd = two_designs(tmp_path)
    ladder.enqueue_experiments(adapter, wd, [
        ExperimentSpec("plain_a", "plain, no bonus", {"VARIANT": "plain", "BONUS": 0.0}),
        ExperimentSpec("shaped_a", "shaped, small bonus", {"VARIANT": "shaped", "BONUS": 0.2}),
        ExperimentSpec("shaped_b", "shaped, larger bonus", {"VARIANT": "shaped", "BONUS": 0.4}),
    ], "screen")
    worker(adapter, wd, device="0")
    brief = problem_mod.load_brief(wd)
    table = compare.grouped_table(adapter, brief, wd, "VARIANT", "screen")
    assert "`shaped`" in table and "`plain`" in table
    assert table.index("`shaped`") < table.index("`plain`")     # better design ranks first
    assert "2" in table                                        # shaped has two experiments

    full = compare.render(adapter, brief, wd, by="VARIANT")
    assert "By `VARIANT`" in full
    assert "no row above is confirmed" in full                 # nothing reached `confirm` yet


def test_compare_shows_only_the_knobs_that_differ(tmp_path):
    _, adapter, wd = two_designs(tmp_path)
    ladder.enqueue_experiments(adapter, wd, [
        ExperimentSpec("a", "bonus 0.1", {"VARIANT": "shaped", "BONUS": 0.1}),
        ExperimentSpec("b", "bonus 0.9", {"VARIANT": "shaped", "BONUS": 0.9}),
    ], "screen")
    worker(adapter, wd, device="0")
    table = compare.flat_table(adapter, problem_mod.load_brief(wd), wd, "screen")
    assert "`BONUS`" in table          # differs between rows -> shown
    assert "`VARIANT`" not in table    # identical in every row -> hidden, it explains nothing


def test_compare_rejects_grouping_by_an_undeclared_knob(tmp_path):
    _, adapter, wd = two_designs(tmp_path)
    out = compare.render(adapter, problem_mod.load_brief(wd), wd, by="NOPE")
    assert "not a declared knob" in out


# ---------------------------------------------------------------- designs in a real session

def test_a_session_can_compare_two_designs_end_to_end(tmp_path):
    _, adapter, wd = two_designs(tmp_path)
    brief = problem_mod.load_brief(wd)
    actor = MockBackend([json.dumps({"experiments": [
        {"id": "use_plain", "hypothesis": "plain design is enough", "config": {"VARIANT": "plain"}}]}),
        json.dumps({"experiments": [
            {"id": "use_shaped", "hypothesis": "shaped reward should win",
             "config": {"VARIANT": "shaped", "BONUS": 0.5}}]})] * 5)
    reviewer = MockBackend([json.dumps({"verdict": "approve",
                                        "approved_item_ids": ["use_plain", "use_shaped"]})] * 20)
    session = Session(adapter, brief, wd, actor=actor, reviewer=reviewer, log=lambda *_: None,
                      cheap_run_seconds=1.0, startup_overhead=0.5)
    state = session.run()
    assert state.terminal_reason == "solved"          # shaped design clears 0.9 at confirm over 3 seeds
    winner = ladder.rank(adapter, wd, "confirm")[0]
    spec = Registry(wd).get_spec(winner[0])
    assert spec.config["VARIANT"] == "shaped"


# ---------------------------------------------------------------- diagnosability of failures

def test_a_failed_run_records_its_output_not_just_an_exit_code(tmp_path):
    """A live session lost 7 runs to `python: command not found` and recorded only "exited 127", so the
    handoff could only ask the user for the traceback. The output has to be in the record."""
    _, adapter, wd = build(tmp_path, {"broken": "import sys\nsys.exit('deliberate explosion')\n"})
    ladder.enqueue_experiments(adapter, wd, [
        ExperimentSpec("boom", "this variant crashes", {"VARIANT": "broken"})], "screen")
    worker(adapter, wd, device="0")
    result = Registry(wd).all_results()[0]
    assert result.status == "failed"
    assert "deliberate explosion" in result.metrics["error"]
    assert (Path(result.run_dir) / "run.log").exists()


def test_the_default_interpreter_is_one_that_exists(tmp_path):
    """`python` is absent on plenty of systems; a variant that cannot start looks exactly like a variant
    that performs badly."""
    brief, _, _ = two_designs(tmp_path)
    del brief.adapter.config["python"]
    adapter = BriefAdapter(brief)
    assert adapter._python == sys.executable
    assert Path(adapter._python).exists()
