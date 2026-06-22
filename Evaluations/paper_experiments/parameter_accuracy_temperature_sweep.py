#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark parameter-selection accuracy across scenarios for a temperature sweep."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from parameter_selection_accuracy import (
    OPTIMAL_PARAMS,
    SCENARIO_ORDER,
    SYSTEM_PROMPT,
    TEST_PROMPTS,
    TOLERANCE,
    TOOL_DEFINITION,
    build_accuracy_table,
    print_accuracy_table,
    score_parameter_selection,
    write_accuracy_table_csv,
    write_accuracy_table_markdown,
)

# --- CONFIGURATION ---
N_REPS = 250
MODEL = "interstellarninja/llama3.1-8b-tools:latest"
TEMPERATURES = [0.0, 0.25, 0.5, 0.75, 1.0]

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "parameter_accuracy_temperature_sweep_results.json"
ACCURACY_TABLE_CSV = EXPERIMENT_DATA_DIR / "parameter_accuracy_temperature_sweep_accuracy_table.csv"
ACCURACY_TABLE_MD = EXPERIMENT_DATA_DIR / "parameter_accuracy_temperature_sweep_accuracy_table.md"


async def run_parameter_accuracy_temperature_sweep() -> None:
    """Run parameter-selection accuracy benchmarking for every scenario and temperature."""
    import ollama

    print(
        f"🚀 Parameter accuracy temperature sweep: {N_REPS} reps x "
        f"{len(TEMPERATURES)} temps x {len(TEST_PROMPTS)} prompts"
    )
    print(f"🤖 Model: {MODEL} (Local)")
    print(f"🌡️ Temperatures: {TEMPERATURES}")
    print(f"🎯 Optimal parameters: {OPTIMAL_PARAMS}")
    print(f"📏 Correct when alpha/beta are within ±{TOLERANCE} of target")

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
                    "temperature": temperature,
                    "prompt_category": scenario,
                    "optimal_tool_args": OPTIMAL_PARAMS[scenario],
                    "tolerance": TOLERANCE,
                }

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]

                try:
                    response = ollama.chat(
                        model=MODEL,
                        messages=messages,
                        tools=TOOL_DEFINITION,
                        options={"seed": rep, "temperature": temperature},
                    )

                    tool_calls = response["message"].get("tool_calls", [])
                    if not tool_calls:
                        record["error"] = "No tool call from agent"
                        record.update(score_parameter_selection(scenario, None))
                        results_db[temp_key][scenario].append(record)
                        print(" ❌ Agent failed (No Tool)")
                        continue

                    tool_call = tool_calls[0]
                    agent_fn_name = tool_call["function"]["name"]
                    agent_fn_args = tool_call["function"]["arguments"]
                    record["agent_tool_name"] = agent_fn_name
                    record["agent_tool_args"] = agent_fn_args

                    if agent_fn_name != "solve_allocation":
                        record["error"] = f"Agent called unexpected tool: {agent_fn_name}"
                        record.update(score_parameter_selection(scenario, agent_fn_args))
                        results_db[temp_key][scenario].append(record)
                        print(f" ⚠️ Agent skipped ({agent_fn_name})")
                        continue

                    record.update(score_parameter_selection(scenario, agent_fn_args))
                    results_db[temp_key][scenario].append(record)
                    print(f" ✅ score={record['parameter_selection_score']:.1f}")

                except Exception as exc:
                    record["error"] = f"Agent run error: {exc}"
                    record.update(score_parameter_selection(scenario, None))
                    results_db[temp_key][scenario].append(record)
                    print(f" ❌ Agent error: {exc}")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    columns = [str(temperature) for temperature in TEMPERATURES]
    table_rows = build_accuracy_table(results_db, columns)
    write_accuracy_table_csv(table_rows, columns, ACCURACY_TABLE_CSV)
    write_accuracy_table_markdown(table_rows, columns, ACCURACY_TABLE_MD)
    print_accuracy_table(table_rows, columns)

    print(f"\n✅ Parameter accuracy temperature sweep complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Accuracy table CSV saved to {ACCURACY_TABLE_CSV}")
    print(f"✅ Accuracy table Markdown saved to {ACCURACY_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_parameter_accuracy_temperature_sweep())
