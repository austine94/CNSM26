#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablated tool-calling reliability model sweep.

This repeats the tool-calling reliability model sweep while removing the main
scaffolding that helps the agent choose the correct tool: no system message and
no retry loop. The single ``solve_allocation`` tool definition is still supplied
so the model can select from the same tool contract as the assisted sweep.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from parameter_accuracy_model_sweep import MODELS, N_REPS, TEMPERATURE
from parameter_selection_accuracy import (
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    TEST_PROMPTS,
    TOOL_DEFINITION,
)
from tool_calling_reliability import (
    CORRECT_TOOL_NAME,
    average,
    score_tool_call,
    selected_tool_names,
)

# --- ABLATED CONFIGURATION ---
MAX_AGENT_TRIES = 1
SYSTEM_INSTRUCTIONS_ENABLED = False
RETRY_ON_TOOL_FAILURE = False

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "tool_calling_reliability_model_sweep_ablated_results.json"
FAILURE_TABLE_CSV = EXPERIMENT_DATA_DIR / "tool_calling_reliability_model_sweep_ablated_failure_table.csv"
FAILURE_TABLE_MD = EXPERIMENT_DATA_DIR / "tool_calling_reliability_model_sweep_ablated_failure_table.md"


def build_ablated_messages(prompt: str) -> list[dict[str, str]]:
    """Return the ablated chat messages: the user prompt only, with no system instructions."""
    return [{"role": "user", "content": prompt}]


def build_failure_table(results: dict[str, Any], models: list[str]) -> list[dict[str, str]]:
    """Create rows containing tool-call failure proportions per model and scenario."""
    table_rows: list[dict[str, str]] = []
    for scenario in SCENARIO_ORDER:
        row = {"scenario": SCENARIO_LABELS[scenario]}
        for model in models:
            runs = results.get(model, {}).get(scenario, [])
            failures = [
                float(run["tool_call_failure"])
                for run in runs
                if run.get("tool_call_failure") is not None
            ]
            failure_rate = average(failures)
            row[model] = "" if failure_rate is None else f"{failure_rate:.4f}"
        table_rows.append(row)
    return table_rows


def write_failure_table_csv(table_rows: list[dict[str, str]], models: list[str], path: Path) -> None:
    """Write a tool-call failure table as CSV."""
    fieldnames = ["scenario", *models]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_failure_table_markdown(table_rows: list[dict[str, str]], models: list[str], path: Path) -> None:
    """Write a tool-call failure table as Markdown."""
    headers = ["Scenario", *models]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table_rows:
        values = [row["scenario"], *[row[model] for model in models]]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_failure_table(table_rows: list[dict[str, str]], models: list[str]) -> None:
    """Print a tool-call failure table to stdout."""
    headers = ["Scenario", *models]
    rows = [[row["scenario"], *[row[model] for model in models]] for row in table_rows]
    widths = [len(header) for header in headers]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print("\nAblated tool-call failure proportion (1.0 = never called correct tool):")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


async def run_tool_calling_reliability_model_sweep_ablated() -> None:
    """Run the ablated one-shot, no-system-instructions tool-call reliability sweep."""
    import ollama

    print(
        f"🚀 Ablated tool-calling reliability model sweep: {N_REPS} reps x "
        f"{len(MODELS)} models x {len(TEST_PROMPTS)} prompts"
    )
    print(f"🤖 Models: {MODELS}")
    print(f"🌡️ Temperature: {TEMPERATURE}")
    print(f"🛠️ Correct tool: {CORRECT_TOOL_NAME}")
    print(f"🔁 Agent tries per prompt: {MAX_AGENT_TRIES}")
    print("🧪 Ablations: no system instructions; no retries after a failed/missing tool call.")
    print("🧰 Tool descriptions remain enabled via the original solve_allocation tool definition.")
    print("🚫 This experiment records the selected tool only; it does not execute tools or call Gurobi.")

    results_db: dict[str, dict[str, list[dict[str, Any]]]] = {
        model: {scenario: [] for scenario in TEST_PROMPTS}
        for model in MODELS
    }

    for rep in range(N_REPS):
        print(f"\n📢 --- REP {rep + 1}/{N_REPS} ---")
        for model in MODELS:
            print(f"  🤖 Model: {model}")
            for scenario in SCENARIO_ORDER:
                prompt = TEST_PROMPTS[scenario]
                print(f"    > Strat: {scenario:<15}...", end="", flush=True)

                record: dict[str, Any] = {
                    "rep": rep,
                    "seed": rep,
                    "base_seed": rep,
                    "model": model,
                    "temperature": TEMPERATURE,
                    "prompt_category": scenario,
                    "correct_tool_name": CORRECT_TOOL_NAME,
                    "max_agent_tries": MAX_AGENT_TRIES,
                    "system_instructions_enabled": SYSTEM_INSTRUCTIONS_ENABLED,
                    "retry_on_tool_failure": RETRY_ON_TOOL_FAILURE,
                    "tool_descriptions_enabled": True,
                    "attempts": [],
                }

                attempt_record: dict[str, Any] = {"attempt": 1, "seed": rep}

                try:
                    response = ollama.chat(
                        model=model,
                        messages=build_ablated_messages(prompt),
                        tools=TOOL_DEFINITION,
                        options={"seed": rep, "temperature": TEMPERATURE},
                    )
                    attempt_record.update(score_tool_call(selected_tool_names(response)))
                except Exception as exc:
                    attempt_record["error"] = f"Agent run error: {exc}"
                    attempt_record.update(score_tool_call([]))

                record["attempts"].append(attempt_record)
                record.update(
                    {
                        "selected_tool_names": attempt_record["selected_tool_names"],
                        "first_tool_name": attempt_record["first_tool_name"],
                        "correct_tool_called": attempt_record["correct_tool_called"],
                        "tool_call_failure": attempt_record["tool_call_failure"],
                        "successful_attempt": 1 if attempt_record["correct_tool_called"] else None,
                        "successful_seed": rep if attempt_record["correct_tool_called"] else None,
                        "num_attempts": 1,
                    }
                )

                if "error" in attempt_record:
                    record["errors"] = [attempt_record["error"]]

                results_db[model][scenario].append(record)

                if record["tool_call_failure"]:
                    first_tool_name = record["first_tool_name"] or "No Tool"
                    print(f" ❌ failed in one try ({first_tool_name})")
                else:
                    print(" ✅ correct on first try")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    table_rows = build_failure_table(results_db, MODELS)
    write_failure_table_csv(table_rows, MODELS, FAILURE_TABLE_CSV)
    write_failure_table_markdown(table_rows, MODELS, FAILURE_TABLE_MD)
    print_failure_table(table_rows, MODELS)

    print(f"\n✅ Ablated tool-calling reliability model sweep complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Failure table CSV saved to {FAILURE_TABLE_CSV}")
    print(f"✅ Failure table Markdown saved to {FAILURE_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_tool_calling_reliability_model_sweep_ablated())
