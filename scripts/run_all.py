"""Run every experiment in order and regenerate everything in results/.

Run from the repo root:
    python scripts/run_all.py                  # all experiments, stop at the first failure
    python scripts/run_all.py --quick          # fewer tuning settings, normalized URLs only
    python scripts/run_all.py --only exp3 exp5 # just these
    python scripts/run_all.py --skip exp5      # all but these
    python scripts/run_all.py --keep-going     # run the rest even if one fails
    python scripts/run_all.py --tests          # run pytest first

Downloads the dataset first if data/raw/ doesn't have it. Each experiment runs as its own
process (python -m experiments.<name>), so one crashing can't leave state behind for the next.
Approximate CPU times: exp3 ~10 min, exp5 ~25 min per URL mode for a CNN that isn't saved yet.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phishing_detection import config  # noqa: E402

# (short name, module, args, args with --quick)
EXPERIMENTS = [
    ("exp1", "experiments.exp1_full_features", [], []),
    ("exp2", "experiments.exp2_feature_selection", ["--k", "10", "15", "20"], ["--k", "10", "15", "20"]),
    ("exp3", "experiments.exp3_tuning", ["--n-iter", "20"], ["--n-iter", "5"]),
    ("exp4", "experiments.exp4_feature_importance", [], []),
    ("exp5", "experiments.exp5_deep_learning", [], ["--url-mode", "normalized"]),
]


def plan(only=None, skip=None, quick=False) -> list[tuple[str, list[str]]]:
    """The (name, command) pairs to run, in order."""
    names = [name for name, *_ in EXPERIMENTS]
    for name in (only or []) + (skip or []):
        if name not in names:
            raise SystemExit(f"Unknown experiment '{name}'. Choose from {names}.")
    steps = []
    for name, module, args, quick_args in EXPERIMENTS:
        if (only and name not in only) or (skip and name in skip):
            continue
        steps.append((name, [sys.executable, "-m", module, *(quick_args if quick else args)]))
    return steps


def run(steps, keep_going=False, runner=subprocess.run) -> list[tuple[str, str, float]]:
    """Run each step from the repo root. Returns (name, status, seconds) per step."""
    results = []
    for i, (name, command) in enumerate(steps):
        print(f"\n=== {name}: {' '.join(command[1:])}", flush=True)
        start = time.perf_counter()
        status = "ok" if runner(command, cwd=ROOT).returncode == 0 else "FAILED"
        results.append((name, status, time.perf_counter() - start))
        if status == "FAILED" and not keep_going:
            results += [(later, "not run", 0.0) for later, _ in steps[i + 1:]]
            break
    return results


def main():
    parser = argparse.ArgumentParser(description="Run every experiment in order.")
    parser.add_argument("--only", nargs="+", metavar="EXP", help="run just these (exp1 ... exp5)")
    parser.add_argument("--skip", nargs="+", metavar="EXP", help="run all but these")
    parser.add_argument("--quick", action="store_true", help="fewer tuning settings, normalized URLs only")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failure")
    parser.add_argument("--tests", action="store_true", help="run pytest before the experiments")
    args = parser.parse_args()

    if not config.RAW_CSV.exists():
        print("Dataset missing; downloading it first.")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")], cwd=ROOT, check=True)

    if args.tests and subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode != 0:
        print("Tests failed; not running the experiments (drop --tests to run them anyway).")
        sys.exit(1)

    results = run(plan(args.only, args.skip, args.quick), args.keep_going)

    print("\n=== Summary")
    for name, status, seconds in results:
        print(f"  {name:5s} {status:8s} {seconds / 60:6.1f} min")
    print(f"  total          {sum(s for *_, s in results) / 60:6.1f} min")
    sys.exit(0 if all(status == "ok" for _, status, _ in results) else 1)


if __name__ == "__main__":
    main()
