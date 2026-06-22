#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark tool-call success across scenarios for a temperature sweep.

This mirrors the parameter-accuracy temperature sweep, but it only checks
whether the model selected the expected ``solve_allocation`` tool.  It does not
execute the selected tool or invoke the optimisation/Gurobi backend.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from parameter_accuracy_temperature_sweep import TEMPERATURES
from parameter_selection_accuracy import (
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    SYSTEM_PROMPT,
    TEST_PROMPTS,
    TOOL_DEFINITION,
)
from tool_calling_reliability import (
    CORRECT_TOOL_NAME,
    MAX_AGENT_TRIES,
    average,
    score_tool_call,
    seed_for_attempt,
    selected_tool_names,
)

# --- CONFIGURATION ---
N_REPS = 250
MODEL = "interstellarninja/llama3.1-8b-tools:latest"

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "tool_calling_reliability_temperature_sweep_results.json"
SUCCESS_TABLE_CSV = EXPERIMENT_DATA_DIR / "tool_calling_reliability_temperature_sweep_success_table.csv"
SUCCESS_TABLE_MD = EXPERIMENT_DATA_DIR / "tool_calling_reliability_temperature_sweep_success_table.md"


def build_success_table(results: dict[str, Any], temperatures: list[float]) -> list[dict[str, str]]:
    """Create rows containing correct-tool-call proportions per scenario and temperature."""
    table_rows: list[dict[str, str]] = []
    temperature_columns = [str(temperature) for temperature in temperatures]

    for scenario in SCENARIO_ORDER:
        row = {"scenario": SCENARIO_LABELS[scenario]}
        for temperature_key in temperature_columns:
            runs = results.get(temperature_key, {}).get(scenario, [])
            successes = [
                float(run["correct_tool_called"])
                for run in runs
                if run.get("correct_tool_called") is not None
            ]
            success_rate = average(successes)
            row[temperature_key] = "" if success_rate is None else f"{success_rate:.4f}"
        table_rows.append(row)

    return table_rows


def write_success_table_csv(table_rows: list[dict[str, str]], temperatures: list[float], path: Path) -> None:
    """Write a tool-call success table as CSV."""
    fieldnames = ["scenario", *[str(temperature) for temperature in temperatures]]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_success_table_markdown(table_rows: list[dict[str, str]], temperatures: list[float], path: Path) -> None:
    """Write a tool-call success table as Markdown."""
    temperature_columns = [str(temperature) for temperature in temperatures]
    headers = ["Scenario", *temperature_columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table_rows:
        values = [row["scenario"], *[row[temperature] for temperature in temperature_columns]]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_success_table(table_rows: list[dict[str, str]], temperatures: list[float]) -> None:
    """Print a tool-call success table to stdout."""
    temperature_columns = [str(temperature) for temperature in temperatures]
    headers = ["Scenario", *temperature_columns]
    rows = [[row["scenario"], *[row[temperature] for temperature in temperature_columns]] for row in table_rows]
    widths = [len(header) for header in headers]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print("\nTool-call success proportion (1.0 = always called correct tool):")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


async def run_tool_calling_reliability_temperature_sweep() -> None:
    """Run tool-call selection reliability benchmarking for every scenario and temperature."""
    import ollama

    print(
        f"🚀 Tool-calling reliability temperature sweep: {N_REPS} reps x "
        f"{len(TEMPERATURES)} temps x {len(TEST_PROMPTS)} prompts"
    )
    print(f"🤖 Model: {MODEL} (Local)")
    print(f"🌡️ Temperatures: {TEMPERATURES}")
    print(f"🛠️ Correct tool: {CORRECT_TOOL_NAME}")
    print(f"🔁 Agent tries per prompt: {MAX_AGENT_TRIES}")
    print("🚫 This experiment records the selected tool only; it does not execute tools or call Gurobi.")

    results_db: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(temperature): {scenario: [] for scenario in TEST_PROMPTS}
        for temperature in TEMPERATURES
    }

    for rep in range(N_REPS):
        print(f"\n📢 --- REP {rep + 1}/{N_REPS} ---")
        for temperature in TEMPERATURES:
            temp_key = str(temperature)
            print(f"  🌡️ Temp: {temperature}")
            for scenario in SCENARIO_ORDER:
                prompt = TEST_PROMPTS[scenario]
                print(f"    > Strat: {scenario:<15}...", end="", flush=True)

                record: dict[str, Any] = {
                    "rep": rep,
                    "seed": rep,
                    "base_seed": rep,
                    "model": MODEL,
                    "temperature": temperature,
                    "prompt_category": scenario,
                    "correct_tool_name": CORRECT_TOOL_NAME,
                    "max_agent_tries": MAX_AGENT_TRIES,
                    "attempts": [],
                }

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]

                successful_attempt: dict[str, Any] | None = None
                last_scored_attempt: dict[str, Any] | None = None

                for attempt_index in range(MAX_AGENT_TRIES):
                    attempt_number = attempt_index + 1
                    attempt_seed = seed_for_attempt(rep, attempt_index, n_reps=N_REPS)
                    attempt_record: dict[str, Any] = {
                        "attempt": attempt_number,
                        "seed": attempt_seed,
                    }

                    try:
                        response = ollama.chat(
                            model=MODEL,
                            messages=messages,
                            tools=TOOL_DEFINITION,
                            options={"seed": attempt_seed, "temperature": temperature},
                        )
                        attempt_record.update(score_tool_call(selected_tool_names(response)))
                    except Exception as exc:
                        attempt_record["error"] = f"Agent run error: {exc}"
                        attempt_record.update(score_tool_call([]))

                    record["attempts"].append(attempt_record)
                    last_scored_attempt = attempt_record

                    if attempt_record["correct_tool_called"]:
                        successful_attempt = attempt_record
                        break

                final_attempt = successful_attempt or last_scored_attempt or score_tool_call([])
                record.update(
                    {
                        "seed": final_attempt.get("seed", rep),
                        "successful_seed": (
                            successful_attempt["seed"] if successful_attempt is not None else None
                        ),
                        "selected_tool_names": final_attempt["selected_tool_names"],
                        "first_tool_name": final_attempt["first_tool_name"],
                        "correct_tool_called": successful_attempt is not None,
                        "tool_call_success": successful_attempt is not None,
                        "tool_call_failure": successful_attempt is None,
                        "successful_attempt": (
                            successful_attempt["attempt"] if successful_attempt is not None else None
                        ),
                        "num_attempts": len(record["attempts"]),
                    }
                )

                errors = [attempt["error"] for attempt in record["attempts"] if "error" in attempt]
                if errors:
                    record["errors"] = errors

                results_db[temp_key][scenario].append(record)

                if record["tool_call_success"]:
                    print(f" ✅ correct on try {record['successful_attempt']}")
                else:
                    first_tool_name = record["first_tool_name"] or "No Tool"
                    print(f" ❌ failed after {record['num_attempts']} tries ({first_tool_name})")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    table_rows = build_success_table(results_db, TEMPERATURES)
    write_success_table_csv(table_rows, TEMPERATURES, SUCCESS_TABLE_CSV)
    write_success_table_markdown(table_rows, TEMPERATURES, SUCCESS_TABLE_MD)
    print_success_table(table_rows, TEMPERATURES)

    print(f"\n✅ Tool-calling reliability temperature sweep complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Success table CSV saved to {SUCCESS_TABLE_CSV}")
    print(f"✅ Success table Markdown saved to {SUCCESS_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_tool_calling_reliability_temperature_sweep())
