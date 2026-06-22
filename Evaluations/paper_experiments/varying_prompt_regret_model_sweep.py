#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark metric-wise regret for a model sweep over generated prompt variants.

For each generated world seed and prompt scenario, this script:
1) solves the scenario with the fixed optimal parameters for that scenario;
2) rewrites the canonical scenario prompt with llama3.2:latest while preserving
   the same optimisation requirements;
3) asks each configured local Ollama model to choose solve_allocation parameters
   from the generated prompt variant;
4) computes metric-wise absolute percentage regret for cost, energy, and latency
   relative to the fixed optimum; and
5) writes both raw JSON results and average-regret tables.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from regret_baseline_comparison import (  # noqa: E402
    PROMPT_FUZZ_MODEL,
    PROMPT_FUZZ_TEMPERATURE,
    generate_fuzzed_prompt,
)
from regret_temperature_sweep import (
    OPTIMAL_PARAMS,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    TEST_PROMPTS,
    average,
    call_solve_allocation,
    compute_regret,
    metric_regret_percent_for_summary,
)
from parameter_selection_accuracy import SYSTEM_PROMPT, TOOL_DEFINITION


# --- CONFIGURATION ---
N_REPS = 250
TEMPERATURE = 0.2
MODELS = [
    "interstellarninja/llama3.1-8b-tools:latest",
    "llama3.2:1b",
    "llama3.2:latest",
    "aratan/qwen3-4b-tools:latest",
    "lukaspetrik/gemma3-tools:4b",
]

PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "varying_prompt_regret_model_sweep_results.json"
REGRET_TABLE_CSV = EXPERIMENT_DATA_DIR / "varying_prompt_regret_model_sweep_average_regret_table.csv"
REGRET_TABLE_MD = EXPERIMENT_DATA_DIR / "varying_prompt_regret_model_sweep_average_regret_table.md"

SERVER_SCRIPT = PROJECT_ROOT / "server" / "mcp_server.py"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_scenario.py"


def build_average_regret_table(results: dict[str, Any]) -> list[dict[str, str]]:
    """Create rows with scenarios on rows and model names on columns."""
    table_rows: list[dict[str, str]] = []

    for scenario in SCENARIO_ORDER:
        row = {"scenario": SCENARIO_LABELS[scenario]}
        for model in MODELS:
            runs = results.get(model, {}).get(scenario, [])
            regrets = [
                regret_percent
                for run in runs
                if (regret_percent := metric_regret_percent_for_summary(run)) is not None
            ]
            avg_regret = average(regrets)
            row[model] = "" if avg_regret is None else f"{avg_regret:.4f}"
        table_rows.append(row)

    return table_rows


def write_regret_table_csv(table_rows: list[dict[str, str]]) -> None:
    """Write the average absolute regret percentage table as CSV."""
    fieldnames = ["scenario", *MODELS]
    with REGRET_TABLE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_regret_table_markdown(table_rows: list[dict[str, str]]) -> None:
    """Write the average absolute regret percentage table as Markdown for quick viewing."""
    headers = ["Scenario", *MODELS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for row in table_rows:
        values = [row["scenario"], *[row[model] for model in MODELS]]
        lines.append("| " + " | ".join(values) + " |")

    REGRET_TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_regret_table(table_rows: list[dict[str, str]]) -> None:
    """Print the average absolute regret percentage table to stdout."""
    headers = ["Scenario", *MODELS]
    rows = [[row["scenario"], *[row[model] for model in MODELS]] for row in table_rows]
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


async def run_varying_prompt_regret_model_sweep() -> None:
    """Run regret benchmarking for generated prompt variants and configured Ollama models."""
    import ollama
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    print(
        f"🚀 Varying-prompt regret model sweep: {N_REPS} reps x {len(MODELS)} models x {len(TEST_PROMPTS)} prompts"
    )
    print(f"🤖 Models: {MODELS}")
    print(f"🌡️ Agent temperature: {TEMPERATURE}")
    print(f"🧪 Prompt fuzzing model: {PROMPT_FUZZ_MODEL} temp={PROMPT_FUZZ_TEMPERATURE}")
    print(f"🎯 Optimal parameters: {OPTIMAL_PARAMS}")

    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER_SCRIPT)],
        env=os.environ.copy(),
    )

    results_db: dict[str, dict[str, list[dict[str, Any]]]] = {
        model: {scenario: [] for scenario in TEST_PROMPTS}
        for model in MODELS
    }

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for rep in range(N_REPS):
                print(f"\n📢 --- REP {rep + 1}/{N_REPS} (World Seed {rep}) ---")
                subprocess.run(["python", str(GENERATOR_SCRIPT), str(rep)], check=True, capture_output=True)

                optimal_outputs: dict[str, dict[str, Any]] = {}
                for scenario in SCENARIO_ORDER:
                    try:
                        optimal_metrics, optimal_objective = await call_solve_allocation(
                            session,
                            OPTIMAL_PARAMS[scenario],
                        )
                        optimal_outputs[scenario] = {
                            "optimal_tool_args": OPTIMAL_PARAMS[scenario],
                            "optimal_metrics": optimal_metrics,
                            "optimal_objective": optimal_objective,
                        }
                    except Exception as exc:
                        optimal_outputs[scenario] = {
                            "optimal_tool_args": OPTIMAL_PARAMS[scenario],
                            "error": f"Optimal baseline error: {exc}",
                        }

                prompt_records: dict[str, dict[str, Any]] = {}
                for scenario in SCENARIO_ORDER:
                    canonical_prompt = TEST_PROMPTS[scenario]
                    print(f"  🧪 Prompt variant: {scenario:<15}...", end="", flush=True)
                    prompt_record = generate_fuzzed_prompt(scenario, canonical_prompt, rep)
                    prompt_records[scenario] = prompt_record
                    print(" ✅" if "prompt_generation_error" not in prompt_record else " ⚠️ fallback")

                for model in MODELS:
                    print(f"  🤖 Model: {model}")
                    for scenario in SCENARIO_ORDER:
                        prompt_record = prompt_records[scenario]
                        prompt = prompt_record["prompt"]
                        print(f"    > Strat: {scenario:<15}...", end="", flush=True)

                        record: dict[str, Any] = {
                            "rep": rep,
                            "seed": rep,
                            "model": model,
                            "temperature": TEMPERATURE,
                            "prompt_category": scenario,
                            **prompt_record,
                            **optimal_outputs[scenario],
                        }

                        if record.get("optimal_objective") is None:
                            record["error"] = record.get("error", "Could not parse optimal objective")
                            results_db[model][scenario].append(record)
                            print(" ❌ Baseline failed")
                            continue

                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ]

                        try:
                            response = ollama.chat(
                                model=model,
                                messages=messages,
                                tools=TOOL_DEFINITION,
                                options={"seed": rep, "temperature": TEMPERATURE},
                            )

                            tool_calls = response["message"].get("tool_calls", [])
                            if not tool_calls:
                                record["error"] = "No tool call from agent"
                                results_db[model][scenario].append(record)
                                print(" ❌ Agent failed (No Tool)")
                                continue

                            tool_call = tool_calls[0]
                            agent_fn_name = tool_call["function"]["name"]
                            agent_fn_args = tool_call["function"]["arguments"]
                            record["agent_tool_args"] = agent_fn_args

                            if agent_fn_name != "solve_allocation":
                                record["error"] = f"Agent called unexpected tool: {agent_fn_name}"
                                results_db[model][scenario].append(record)
                                print(f" ⚠️ Agent skipped ({agent_fn_name})")
                                continue

                            agent_metrics, agent_objective = await call_solve_allocation(
                                session,
                                agent_fn_args,
                            )
                            record["agent_metrics"] = agent_metrics
                            record["agent_objective"] = agent_objective

                            if agent_objective is None:
                                record["error"] = "Could not parse agent objective"
                                results_db[model][scenario].append(record)
                                print(" ⚠️ Missing objective")
                                continue

                            record.update(
                                compute_regret(
                                    agent_metrics,
                                    record["optimal_metrics"],
                                    agent_objective,
                                    record["optimal_objective"],
                                )
                            )
                            results_db[model][scenario].append(record)
                            print(" ✅")

                        except Exception as exc:
                            record["error"] = f"Agent run error: {exc}"
                            results_db[model][scenario].append(record)
                            print(f" ❌ Agent error: {exc}")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    table_rows = build_average_regret_table(results_db)
    write_regret_table_csv(table_rows)
    write_regret_table_markdown(table_rows)
    print_regret_table(table_rows)

    print(f"\n✅ Varying-prompt regret model sweep complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Average absolute regret table CSV saved to {REGRET_TABLE_CSV}")
    print(f"✅ Average absolute regret table Markdown saved to {REGRET_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_varying_prompt_regret_model_sweep())
