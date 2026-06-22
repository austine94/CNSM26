#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute regret for saved conflict-experiment parameter selections.

This is a post-processing experiment: it reads previously saved parameter
choices, regenerates each recorded world seed, evaluates both the recorded
model-selected parameters and the scenario optimum with ``solve_allocation``,
and computes the same metric-wise regret fields used by the other regret
experiments. It does not call Ollama or make any agentic/model requests.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_selection_accuracy import (  # noqa: E402
    OPTIMAL_PARAMS,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
)
from regret_temperature_sweep import (  # noqa: E402
    average,
    call_solve_allocation,
    compute_regret,
    metric_regret_percent_for_summary,
)
from stress_parameter_accuracy_model_sweep import (  # noqa: E402
    STRESS_OPTIMAL_PARAMS,
    STRESS_SCENARIO_LABELS,
    STRESS_SCENARIO_ORDER,
)

PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
DEFAULT_SELECTIONS_FILE = EXPERIMENT_DATA_DIR / "conflict_parameter_selection_results.json"
DEFAULT_RESULTS_FILE = EXPERIMENT_DATA_DIR / "conflict_parameter_selection_regret_results.json"
DEFAULT_REGRET_TABLE_CSV = EXPERIMENT_DATA_DIR / "conflict_parameter_selection_regret_table.csv"
DEFAULT_REGRET_TABLE_MD = EXPERIMENT_DATA_DIR / "conflict_parameter_selection_regret_table.md"

SERVER_SCRIPT = PROJECT_ROOT / "server" / "mcp_server.py"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_scenario.py"
MAX_DATACENTRES = 9999

KNOWN_OPTIMAL_PARAMS = {
    **OPTIMAL_PARAMS,
    **STRESS_OPTIMAL_PARAMS,
}
KNOWN_SCENARIO_LABELS = {
    **SCENARIO_LABELS,
    **STRESS_SCENARIO_LABELS,
}
KNOWN_SCENARIO_ORDER = [
    *SCENARIO_ORDER,
    *[scenario for scenario in STRESS_SCENARIO_ORDER if scenario not in SCENARIO_ORDER],
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the post-processing script."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute Gurobi-backed regret for saved conflict-experiment parameter "
            "selections without making any agentic/model calls."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SELECTIONS_FILE,
        help=f"Parameter-selection JSON to read (default: {DEFAULT_SELECTIONS_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help=f"Raw regret JSON to write (default: {DEFAULT_RESULTS_FILE})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_REGRET_TABLE_CSV,
        help=f"Average regret CSV to write (default: {DEFAULT_REGRET_TABLE_CSV})",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_REGRET_TABLE_MD,
        help=f"Average regret Markdown table to write (default: {DEFAULT_REGRET_TABLE_MD})",
    )
    return parser.parse_args()


def load_selection_results(path: Path) -> dict[str, Any]:
    """Load a saved parameter-selection result JSON object."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Pass your conflict parameter-selection file with --input."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}, got {type(data).__name__}")

    return data


def is_scenario_runs_mapping(value: Any) -> bool:
    """Return True when a JSON value looks like ``scenario -> [runs]``."""
    return isinstance(value, dict) and all(isinstance(runs, list) for runs in value.values())


def normalise_selection_results(raw_results: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Normalise supported selection-result schemas to ``group -> scenario -> runs``.

    Supported inputs include:
    - model/group keyed results from paper experiments: ``model -> scenario -> [records]``;
    - older single-model parameter-choice files: ``scenario -> [records]``.
    """
    if all(isinstance(runs, list) for runs in raw_results.values()):
        return {
            "parameter_selections": {
                scenario: [run for run in runs if isinstance(run, dict)]
                for scenario, runs in raw_results.items()
            }
        }

    normalised: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group, scenario_results in raw_results.items():
        if not is_scenario_runs_mapping(scenario_results):
            continue
        normalised[group] = {
            scenario: [run for run in runs if isinstance(run, dict)]
            for scenario, runs in scenario_results.items()
        }

    if not normalised:
        raise ValueError(
            "Could not find parameter-selection runs. Expected either "
            "scenario -> [records] or group/model -> scenario -> [records]."
        )

    return normalised


def ordered_groups(results: dict[str, dict[str, list[dict[str, Any]]]]) -> list[str]:
    """Return result group names in input order."""
    return list(results.keys())


def ordered_scenarios(results: dict[str, dict[str, list[dict[str, Any]]]]) -> list[str]:
    """Return known scenarios first, followed by any extra scenarios in input order."""
    discovered: list[str] = []
    for scenario_results in results.values():
        for scenario in scenario_results:
            if scenario not in discovered:
                discovered.append(scenario)

    known = [scenario for scenario in KNOWN_SCENARIO_ORDER if scenario in discovered]
    extras = [scenario for scenario in discovered if scenario not in known]
    return [*known, *extras]


def scenario_label(scenario: str) -> str:
    """Return a display label for a scenario key."""
    return KNOWN_SCENARIO_LABELS.get(scenario, scenario)


def extract_tool_args(run: dict[str, Any]) -> dict[str, Any] | None:
    """Extract model-selected solve_allocation arguments from a saved record."""
    for key in ("agent_tool_args", "tool_args"):
        value = run.get(key)
        if isinstance(value, dict):
            return value

    if "alpha" in run or "beta" in run:
        return {
            "alpha": run.get("alpha"),
            "beta": run.get("beta"),
            "max_datacentres": run.get("max_datacentres", MAX_DATACENTRES),
        }

    return None


def extract_optimal_args(scenario: str, run: dict[str, Any]) -> dict[str, Any] | None:
    """Extract or infer optimum solve_allocation arguments for a scenario."""
    optimal_args = run.get("optimal_tool_args")
    if isinstance(optimal_args, dict):
        return optimal_args
    return KNOWN_OPTIMAL_PARAMS.get(scenario)


def coerce_float(value: Any, field_name: str) -> float:
    """Coerce a JSON value to a float suitable for solve_allocation."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} is missing or not numeric")
    return float(value)


def normalise_solve_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalise saved solve_allocation arguments before calling the MCP tool."""
    return {
        "alpha": coerce_float(args.get("alpha"), "alpha"),
        "beta": coerce_float(args.get("beta"), "beta"),
        "max_datacentres": args.get("max_datacentres", MAX_DATACENTRES),
    }


def run_seed(run: dict[str, Any]) -> int:
    """Return the world seed for a saved run."""
    seed = run.get("seed", run.get("rep"))
    if seed is None or isinstance(seed, bool):
        raise ValueError("run is missing a numeric seed/rep")
    return int(seed)


def args_cache_key(arguments: dict[str, Any]) -> str:
    """Return a stable cache key for solve_allocation arguments."""
    return json.dumps(arguments, sort_keys=True, default=str)


def empty_results_like(
    selections: dict[str, dict[str, list[dict[str, Any]]]]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Create an empty nested result object matching the selection groups/scenarios."""
    return {
        group: {scenario: [] for scenario in scenario_results}
        for group, scenario_results in selections.items()
    }


def flatten_by_seed(
    selections: dict[str, dict[str, list[dict[str, Any]]]]
) -> dict[int, list[tuple[str, str, dict[str, Any]]]]:
    """Group saved runs by world seed so each scenario is regenerated once."""
    runs_by_seed: dict[int, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)

    for group, scenario_results in selections.items():
        for scenario, runs in scenario_results.items():
            for run in runs:
                try:
                    seed = run_seed(run)
                except (TypeError, ValueError):
                    seed = -1
                runs_by_seed[seed].append((group, scenario, run))

    return dict(sorted(runs_by_seed.items()))


def build_average_regret_table(
    results: dict[str, dict[str, list[dict[str, Any]]]],
    groups: list[str],
    scenarios: list[str],
) -> list[dict[str, str]]:
    """Create rows with scenarios on rows and result groups on columns."""
    table_rows: list[dict[str, str]] = []

    for scenario in scenarios:
        row = {"scenario": scenario_label(scenario)}
        for group in groups:
            runs = results.get(group, {}).get(scenario, [])
            regrets = [
                regret_percent
                for run in runs
                if (regret_percent := metric_regret_percent_for_summary(run)) is not None
            ]
            avg_regret = average(regrets)
            row[group] = "" if avg_regret is None else f"{avg_regret:.4f}"
        table_rows.append(row)

    return table_rows


def write_regret_table_csv(table_rows: list[dict[str, str]], groups: list[str], path: Path) -> None:
    """Write the average regret table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario", *groups]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_regret_table_markdown(table_rows: list[dict[str, str]], groups: list[str], path: Path) -> None:
    """Write the average regret table as Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Scenario", *groups]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in table_rows:
        values = [row["scenario"], *[row[group] for group in groups]]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_regret_table(table_rows: list[dict[str, str]], groups: list[str]) -> None:
    """Print the average regret table to stdout."""
    headers = ["Scenario", *groups]
    rows = [[row["scenario"], *[row[group] for group in groups]] for row in table_rows]
    widths = [len(header) for header in headers]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print("\nAverage metric-wise absolute regret percentage from optimum:")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


async def compute_conflict_parameter_selection_regret(
    selections: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Evaluate saved parameter selections and compute regret using solve_allocation."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER_SCRIPT)],
        env=os.environ.copy(),
    )

    results = empty_results_like(selections)
    runs_by_seed = flatten_by_seed(selections)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for seed, seed_runs in runs_by_seed.items():
                if seed < 0:
                    print("\n📢 --- Runs with missing/invalid seeds ---")
                else:
                    print(f"\n📢 --- World Seed {seed} ({len(seed_runs)} saved selections) ---")
                    subprocess.run(
                        ["python", str(GENERATOR_SCRIPT), str(seed)],
                        check=True,
                        capture_output=True,
                    )

                solve_cache: dict[str, tuple[dict[str, float], float | None]] = {}

                async def cached_solve(arguments: dict[str, Any]) -> tuple[dict[str, float], float | None]:
                    key = args_cache_key(arguments)
                    if key not in solve_cache:
                        solve_cache[key] = await call_solve_allocation(session, arguments)
                    return solve_cache[key]

                for group, scenario, run in seed_runs:
                    print(f"  > {group} / {scenario:<24}...", end="", flush=True)
                    record: dict[str, Any] = {
                        **run,
                        "model": run.get("model", group),
                        "prompt_category": run.get("prompt_category", scenario),
                        "source_error": run.get("error"),
                    }

                    try:
                        if seed < 0:
                            raise ValueError("Cannot compute regret without a valid seed/rep")

                        optimal_args = extract_optimal_args(scenario, run)
                        if optimal_args is None:
                            raise ValueError(f"No optimal_tool_args available for scenario {scenario}")

                        agent_args = extract_tool_args(run)
                        if agent_args is None:
                            raise ValueError("No model-selected tool arguments found")

                        optimal_tool_args = normalise_solve_args(optimal_args)
                        agent_tool_args = normalise_solve_args(agent_args)

                        record["seed"] = seed
                        record["optimal_tool_args"] = optimal_tool_args
                        record["agent_tool_args"] = agent_tool_args

                        optimal_metrics, optimal_objective = await cached_solve(optimal_tool_args)
                        record["optimal_metrics"] = optimal_metrics
                        record["optimal_objective"] = optimal_objective
                        if optimal_objective is None:
                            raise ValueError("Could not parse optimal objective")

                        agent_metrics, agent_objective = await cached_solve(agent_tool_args)
                        record["agent_metrics"] = agent_metrics
                        record["agent_objective"] = agent_objective
                        if agent_objective is None:
                            raise ValueError("Could not parse agent objective")

                        record.update(
                            compute_regret(
                                agent_metrics,
                                optimal_metrics,
                                agent_objective,
                                optimal_objective,
                            )
                        )
                        print(" ✅")
                    except Exception as exc:
                        record["error"] = f"Regret computation error: {exc}"
                        print(f" ❌ {exc}")

                    results[group][scenario].append(record)

    return results


async def main() -> None:
    """Run conflict-parameter regret post-processing."""
    args = parse_args()
    selections = normalise_selection_results(load_selection_results(args.input))
    groups = ordered_groups(selections)
    scenarios = ordered_scenarios(selections)

    print("🚀 Conflict parameter-selection regret post-processing")
    print(f"📥 Reading selections: {args.input}")
    print(f"🤖 Result groups: {groups}")
    print(f"🧪 Scenarios: {scenarios}")
    print("🚫 No Ollama/model calls are made; this only runs solve_allocation via Gurobi.")

    results = await compute_conflict_parameter_selection_regret(selections)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    table_rows = build_average_regret_table(results, groups, scenarios)
    write_regret_table_csv(table_rows, groups, args.csv)
    write_regret_table_markdown(table_rows, groups, args.markdown)
    print_regret_table(table_rows, groups)

    print(f"\n✅ Conflict parameter-selection regret results saved to {args.output}")
    print(f"✅ Average regret table CSV saved to {args.csv}")
    print(f"✅ Average regret table Markdown saved to {args.markdown}")


if __name__ == "__main__":
    asyncio.run(main())
