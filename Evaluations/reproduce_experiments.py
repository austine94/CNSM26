#!/usr/bin/env python3
"""Run the selected paper experiment reproduction subset.

Run this script from the ``paper_experiments`` directory with:

    python reproduce_experiments.py

The script executes only the experiment/post-processing scripts listed in
``EXPERIMENT_SCRIPTS`` and stops at the first failure by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

EXPERIMENT_SCRIPTS: tuple[str, ...] = (
    "varying_prompt_regret_model_sweep.py",
    "varying_prompt_regret_model_sweep_parameter_selection_accuracy.py",
    "scalability_experiment.py",
    "plot_scalability_experiment.py",
    "regret_baseline_comparison.py",
    "tool_calling_reliability.py",
    "tool_calling_reliability_model_sweep_ablated.py",
    "tool_calling_reliability_temperature_sweep.py",
    "parameter_accuracy_model_sweep.py",
    "parameter_accuracy_selection_std_tables.py",
    "conflict_parameter_selection_regret.py",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the reproduction runner."""
    parser = argparse.ArgumentParser(
        description="Run the selected subset of paper experiments in sequence."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later scripts after a script exits with a non-zero status.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the configured experiment scripts from the paper_experiments directory."""
    args = parse_args()
    failures: list[tuple[str, int]] = []

    for script_name in EXPERIMENT_SCRIPTS:
        script_path = SCRIPT_DIR / script_name
        command = [sys.executable, script_name]
        printable_command = " ".join(command)

        if not script_path.exists():
            print(f"Missing script: {script_path}", file=sys.stderr)
            failures.append((script_name, 127))
            if not args.continue_on_error:
                return 127
            continue

        print(f"\n=== Running: {printable_command} ===", flush=True)
        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=SCRIPT_DIR, check=False)
        if completed.returncode != 0:
            print(
                f"Script failed with exit code {completed.returncode}: {script_name}",
                file=sys.stderr,
            )
            failures.append((script_name, completed.returncode))
            if not args.continue_on_error:
                return completed.returncode

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for script_name, return_code in failures:
            print(f"- {script_name}: exit code {return_code}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
