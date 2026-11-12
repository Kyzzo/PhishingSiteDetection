"""run_all.py: which experiments run, in what order, and stopping on failure (no real runs)."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("run_all", Path(__file__).resolve().parents[1] / "scripts" / "run_all.py")
run_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_all)


def names(steps):
    return [name for name, _ in steps]


def test_plan_runs_everything_in_order_by_default():
    assert names(run_all.plan()) == ["exp1", "exp2", "exp3", "exp4", "exp5"]


def test_plan_only_skip_and_quick():
    assert names(run_all.plan(only=["exp5", "exp3"])) == ["exp3", "exp5"]
    assert names(run_all.plan(skip=["exp5"])) == ["exp1", "exp2", "exp3", "exp4"]
    exp5 = dict(run_all.plan(only=["exp5"], quick=True))["exp5"]
    assert exp5[-2:] == ["--url-mode", "normalized"]


def test_plan_rejects_unknown_names():
    with pytest.raises(SystemExit):
        run_all.plan(only=["exp9"])


def test_run_stops_at_first_failure_unless_keep_going():
    steps = run_all.plan()
    fail_exp2 = lambda cmd, cwd: SimpleNamespace(returncode=1 if "exp2" in cmd[2] else 0)
    stopped = run_all.run(steps, runner=fail_exp2)
    assert [s for _, s, _ in stopped] == ["ok", "FAILED", "not run", "not run", "not run"]
    kept = run_all.run(steps, keep_going=True, runner=fail_exp2)
    assert [s for _, s, _ in kept] == ["ok", "FAILED", "ok", "ok", "ok"]
