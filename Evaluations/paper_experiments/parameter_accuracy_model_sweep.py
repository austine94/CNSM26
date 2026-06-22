#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark parameter-selection accuracy across scenarios for a local SLM model sweep."""

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
TEMPERATURE = 0.05
MODELS = [
    "interstellarninja/llama3.1-8b-tools:latest",
    "llama3.2:1b",
    "llama3.2:latest",
    "aratan/qwen3-4b-tools:latest",
    "lukaspetrik/gemma3-tools:4b",
]

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "parameter_accuracy_model_sweep_results.json"
ACCURACY_TABLE_CSV = EXPERIMENT_DATA_DIR / "parameter_accuracy_model_sweep_accuracy_table.csv"
ACCURACY_TABLE_MD = EXPERIMENT_DATA_DIR / "parameter_accuracy_model_sweep_accuracy_table.md"


async def run_parameter_accuracy_model_sweep() -> None:
    """Run parameter-selection accuracy benchmarking for every scenario and configured Ollama model."""
    import ollama

    print(
        f"🚀 Parameter accuracy model sweep: {N_REPS} reps x "
        f"{len(MODELS)} models x {len(TEST_PROMPTS)} prompts"
    )
    print(f"🤖 Models: {MODELS}")
    print(f"🌡️ Temperature: {TEMPERATURE}")
    print(f"🎯 Optimal parameters: {OPTIMAL_PARAMS}")
    print(f"📏 Correct when alpha/beta are within ±{TOLERANCE} of target")

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
                    "model": model,
                    "temperature": TEMPERATURE,
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
                        model=model,
                        messages=messages,
                        tools=TOOL_DEFINITION,
                        options={"seed": rep, "temperature": TEMPERATURE},
                    )

                    tool_calls = response["message"].get("tool_calls", [])
                    if not tool_calls:
                        record["error"] = "No tool call from agent"
                        record.update(score_parameter_selection(scenario, None))
                        results_db[model][scenario].append(record)
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
                        results_db[model][scenario].append(record)
                        print(f" ⚠️ Agent skipped ({agent_fn_name})")
                        continue

                    record.update(score_parameter_selection(scenario, agent_fn_args))
                    results_db[model][scenario].append(record)
                    print(f" ✅ score={record['parameter_selection_score']:.1f}")

                except Exception as exc:
                    record["error"] = f"Agent run error: {exc}"
                    record.update(score_parameter_selection(scenario, None))
                    results_db[model][scenario].append(record)
                    print(f" ❌ Agent error: {exc}")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    table_rows = build_accuracy_table(results_db, MODELS)
    write_accuracy_table_csv(table_rows, MODELS, ACCURACY_TABLE_CSV)
    write_accuracy_table_markdown(table_rows, MODELS, ACCURACY_TABLE_MD)
    print_accuracy_table(table_rows, MODELS)

    print(f"\n✅ Parameter accuracy model sweep complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Accuracy table CSV saved to {ACCURACY_TABLE_CSV}")
    print(f"✅ Accuracy table Markdown saved to {ACCURACY_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_parameter_accuracy_model_sweep())
