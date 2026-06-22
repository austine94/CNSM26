#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark tool-calling reliability across fixed parameter-accuracy prompts.

This experiment reuses the same single solve_allocation tool contract as the
parameter-accuracy setting. It only inspects whether a model selected that tool;
it does not execute the selected tool or invoke the optimisation/Gurobi backend.
"""

from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from parameter_accuracy_model_sweep import MODELS, N_REPS, TEMPERATURE
from parameter_selection_accuracy import (
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    SYSTEM_PROMPT,
    TEST_PROMPTS,
    TOOL_DEFINITION,
)

CORRECT_TOOL_NAME = "solve_allocation"
MAX_AGENT_TRIES = 3

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "tool_calling_reliability_results.json"
FAILURE_TABLE_CSV = EXPERIMENT_DATA_DIR / "tool_calling_reliability_failure_table.csv"
FAILURE_TABLE_MD = EXPERIMENT_DATA_DIR / "tool_calling_reliability_failure_table.md"


def get_response_field(value: Any, field_name: str, default: Any = None) -> Any:
    """Return a field from dict-like or attribute-based response objects.

    The Ollama Python client has returned both plain dictionaries and typed
    response objects across versions.  Tool calls can therefore be nested in
    either mapping keys (``response["message"]["tool_calls"]``) or object
    attributes (``response.message.tool_calls``).
    """
    if isinstance(value, Mapping):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def selected_tool_names(response: Any) -> list[str]:
    """Extract selected tool names from an Ollama chat response.

    Supports both legacy dictionary responses and newer typed Ollama response
    objects. Without this compatibility layer, valid tool calls represented as
    objects are misclassified as missing tool calls.
    """
    message = get_response_field(response, "message")
    tool_calls = get_response_field(message, "tool_calls", [])
    if not tool_calls:
        return []

    names: list[str] = []
    for tool_call in tool_calls:
        function = get_response_field(tool_call, "function", {})
        name = get_response_field(function, "name")
        if isinstance(name, str):
            names.append(name)
    return names


def score_tool_call(tool_names: list[str], correct_tool_name: str = CORRECT_TOOL_NAME) -> dict[str, Any]:
    """Return pass/fail metadata for whether the first selected tool is correct."""
    first_tool_name = tool_names[0] if tool_names else None
    correct_tool_called = first_tool_name == correct_tool_name
    return {
        "selected_tool_names": tool_names,
        "first_tool_name": first_tool_name,
        "correct_tool_name": correct_tool_name,
        "correct_tool_called": correct_tool_called,
        "tool_call_failure": not correct_tool_called,
    }


def seed_for_attempt(rep: int, attempt_index: int, n_reps: int = N_REPS) -> int:
    """Return a stable, distinct seed for a retry attempt within a repetition."""
    return rep + (attempt_index * n_reps)


def average(values: list[float]) -> float | None:
    """Return the arithmetic mean, or None when no values are available."""
    if not values:
        return None
    return sum(values) / len(values)


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

    print("\nTool-call failure proportion (1.0 = never called correct tool):")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


async def run_tool_calling_reliability() -> None:
    """Run tool-call selection reliability benchmarking for every scenario and model."""
    import ollama

    print(
        f"🚀 Tool-calling reliability: {N_REPS} reps x "
        f"{len(MODELS)} models x {len(TEST_PROMPTS)} prompts"
    )
    print(f"🤖 Models: {MODELS}")
    print(f"🌡️ Temperature: {TEMPERATURE}")
    print(f"🛠️ Correct tool: {CORRECT_TOOL_NAME}")
    print(f"🔁 Agent tries per prompt: {MAX_AGENT_TRIES}")
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
                    attempt_seed = seed_for_attempt(rep, attempt_index)
                    attempt_record: dict[str, Any] = {
                        "attempt": attempt_number,
                        "seed": attempt_seed,
                    }

                    try:
                        response = ollama.chat(
                            model=model,
                            messages=messages,
                            tools=TOOL_DEFINITION,
                            options={"seed": attempt_seed, "temperature": TEMPERATURE},
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

                results_db[model][scenario].append(record)

                if record["tool_call_failure"]:
                    first_tool_name = record["first_tool_name"] or "No Tool"
                    print(f" ❌ failed after {record['num_attempts']} tries ({first_tool_name})")
                else:
                    print(f" ✅ correct on try {record['successful_attempt']}")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    table_rows = build_failure_table(results_db, MODELS)
    write_failure_table_csv(table_rows, MODELS, FAILURE_TABLE_CSV)
    write_failure_table_markdown(table_rows, MODELS, FAILURE_TABLE_MD)
    print_failure_table(table_rows, MODELS)

    print(f"\n✅ Tool-calling reliability complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Failure table CSV saved to {FAILURE_TABLE_CSV}")
    print(f"✅ Failure table Markdown saved to {FAILURE_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_tool_calling_reliability())
