#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark cost-latency trade-off scalability across workload sizes and local SLMs.

For each workload size and repetition, this script:
1) generates a reproducible scalable workload;
2) solves the cost-latency trade-off objective with fixed target parameters;
3) asks each configured local Ollama model to choose solve_allocation parameters;
4) records metric-wise absolute percentage regret, Ollama count fields / total token count, and timings; and
5) writes raw JSON plus aggregate metric tables for plotting.
"""

from __future__ import annotations

import asyncio
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from parameter_selection_accuracy import SYSTEM_PROMPT, TOOL_DEFINITION
from regret_temperature_sweep import OPTIMAL_PARAMS, compute_regret, metric_regret_percent_for_summary

# --- CONFIGURATION ---
WORKLOAD_SIZES = list(range(50, 1001, 50))
N_REPS = 20
TEMPERATURE = 0.2
SCENARIO = "tradeoff_cost_latency"
PROMPT = "Optimise this using all datacentres. Keep financial cost controlled, but maintain acceptable user experience."
MODELS = [
    "interstellarninja/llama3.1-8b-tools:latest",
    "llama3.2:1b",
    "llama3.2:latest",
    "aratan/qwen3-4b-tools:latest",
    "lukaspetrik/gemma3-tools:4b",
]
N_DATACENTRES = 5
N_HOTSPOTS = 5
DATACENTRE_CAPACITY = 1100

PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "scalability_experiment_results.json"
METRICS_TABLE_CSV = EXPERIMENT_DATA_DIR / "scalability_average_metrics_table.csv"
METRICS_TABLE_MD = EXPERIMENT_DATA_DIR / "scalability_average_metrics_table.md"

GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "scalability_data_generator.py"
DATA_PATH = PROJECT_ROOT / "data" / "current_problem.json"
ENGINE_PATH = PROJECT_ROOT / "optimisation_engine"


def workload_seed(workload_size: int, rep: int) -> int:
    """Return a reproducible seed unique to a workload size and repetition."""
    return workload_size * 10_000 + rep


def average(values: list[float]) -> float | None:
    """Return the arithmetic mean, or None when no values are available."""
    if not values:
        return None
    return sum(values) / len(values)


def response_to_dict(response: Any) -> dict[str, Any]:
    """Convert an Ollama response object to a plain dictionary."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return dict(response)


def extract_ollama_count_fields(response: Any) -> dict[str, int | float]:
    """Extract top-level Ollama response fields whose names include 'count'."""
    response_data = response_to_dict(response)

    return {
        str(key): value
        for key, value in response_data.items()
        if "count" in str(key).lower() and isinstance(value, (int, float))
    }


def total_token_count(count_fields: dict[str, int | float]) -> int | float | None:
    """Compute total token count from Ollama count fields when possible."""
    if "total_token_count" in count_fields:
        return count_fields["total_token_count"]
    if "total_tokens" in count_fields:
        return count_fields["total_tokens"]

    prompt_count = count_fields.get("prompt_eval_count")
    eval_count = count_fields.get("eval_count")
    if prompt_count is not None and eval_count is not None:
        return prompt_count + eval_count
    if eval_count is not None:
        return eval_count
    return None


def load_problem_into_instance() -> Any:
    """Load the current JSON problem using the same normalisation as the MCP server."""
    import numpy as np

    if str(ENGINE_PATH) not in sys.path:
        sys.path.append(str(ENGINE_PATH))

    from flexmedia_delivery_instance import flexmedia_delivery_instance

    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    inst = flexmedia_delivery_instance(
        n_datacentres=data["n_datacentres"],
        n_consumers=data["n_consumers"],
        n_hotspots=1,
    )

    raw_dc_costs = np.array(data["datacentre_costs"])
    raw_consumer_costs = np.array(data["consumer_costs"])
    raw_energy_costs = np.array(data["energy_costs"])

    inst.datacentre_capacities = np.array(data["datacentre_capacities"])
    inst.consumer_latencies = np.array(data["consumer_latencies"])
    inst.datacentre_locations = np.array(data["datacentre_locations"])
    inst.consumer_locations = np.array(data["consumer_locations"])

    max_financial = max(np.max(raw_dc_costs), np.max(raw_consumer_costs))
    if max_financial > 0:
        inst.datacentre_costs = raw_dc_costs / max_financial
        inst.consumer_costs = raw_consumer_costs / max_financial
    else:
        inst.datacentre_costs = raw_dc_costs
        inst.consumer_costs = raw_consumer_costs

    max_energy = np.max(raw_energy_costs)
    if max_energy > 0:
        inst.energy_costs = raw_energy_costs / max_energy
    else:
        inst.energy_costs = raw_energy_costs

    inst.financial_scaler = max_financial
    inst.energy_scaler = max_energy
    return inst


def evaluate_solution_objective(inst: Any, *, alpha: float, beta: float) -> float | None:
    """Evaluate an already-solved assignment under a chosen objective."""
    if inst.assignments is None or inst.open_centres is None:
        return None

    objective = sum(inst.datacentre_costs[i] for i in inst.open_centres)
    for dc_idx, consumers in inst.assignments.items():
        for consumer_idx in consumers:
            objective += (
                beta * inst.consumer_costs[dc_idx, consumer_idx]
                + (1 - beta) * inst.energy_costs[dc_idx, consumer_idx]
                + alpha * inst.consumer_latencies[dc_idx, consumer_idx]
            )
    return float(objective)


def solve_and_evaluate(
    solve_args: dict[str, Any],
    evaluation_args: dict[str, Any],
) -> tuple[dict[str, Any], float | None, float | None]:
    """Solve with one parameter set and evaluate the solution under another."""
    inst = load_problem_into_instance()
    inst.exact_solve(alpha=float(solve_args["alpha"]), beta=float(solve_args["beta"]))
    if inst.exact_obj is None:
        return {}, None, None

    metrics = inst.get_exact_solution_breakdown() or {}
    evaluated_objective = evaluate_solution_objective(
        inst,
        alpha=float(evaluation_args["alpha"]),
        beta=float(evaluation_args["beta"]),
    )
    return metrics, float(inst.exact_obj), evaluated_objective


def build_average_metrics_table(results: dict[str, Any]) -> list[dict[str, str]]:
    """Create aggregate metric rows by workload size and model."""
    table_rows: list[dict[str, str]] = []
    metrics = [
        "metric_regret_percent",
        "total_token_count",
        "agent_end_to_end_seconds",
        "total_case_seconds",
    ]

    for workload_size in WORKLOAD_SIZES:
        size_key = str(workload_size)
        for model in MODELS:
            runs = results.get(size_key, {}).get(model, [])
            row = {"workload_size": size_key, "model": model}
            for metric in metrics:
                if metric == "metric_regret_percent":
                    values = [
                        regret_percent
                        for run in runs
                        if (regret_percent := metric_regret_percent_for_summary(run)) is not None
                    ]
                else:
                    values = [float(run[metric]) for run in runs if run.get(metric) is not None]
                avg_value = average(values)
                row[f"average_{metric}"] = "" if avg_value is None else f"{avg_value:.6f}"
            table_rows.append(row)

    return table_rows


def write_metrics_table_csv(table_rows: list[dict[str, str]]) -> None:
    """Write aggregate scalability metrics as CSV."""
    fieldnames = [
        "workload_size",
        "model",
        "average_metric_regret_percent",
        "average_total_token_count",
        "average_agent_end_to_end_seconds",
        "average_total_case_seconds",
    ]
    with METRICS_TABLE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_metrics_table_markdown(table_rows: list[dict[str, str]]) -> None:
    """Write aggregate scalability metrics as Markdown for quick viewing."""
    headers = [
        "Workload size",
        "Model",
        "Avg metric-wise regret (%)",
        "Avg tokens",
        "Avg agent e2e (s)",
        "Avg total case (s)",
    ]
    keys = [
        "workload_size",
        "model",
        "average_metric_regret_percent",
        "average_total_token_count",
        "average_agent_end_to_end_seconds",
        "average_total_case_seconds",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table_rows:
        lines.append("| " + " | ".join(row[key] for key in keys) + " |")
    METRICS_TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_workload(workload_size: int, seed: int) -> float:
    """Generate one scalable workload and return the generation wall-clock time."""
    start = time.perf_counter()
    subprocess.run(
        [
            "python",
            str(GENERATOR_SCRIPT),
            str(seed),
            "--n-consumers",
            str(workload_size),
            "--n-datacentres",
            str(N_DATACENTRES),
            "--n-hotspots",
            str(N_HOTSPOTS),
            "--datacentre-capacity",
            str(DATACENTRE_CAPACITY),
        ],
        check=True,
        capture_output=True,
    )
    return time.perf_counter() - start


async def run_scalability_experiment() -> None:
    """Run cost-latency trade-off scalability benchmarking for every size/model."""
    import ollama

    print(
        f"🚀 Scalability experiment: {len(WORKLOAD_SIZES)} sizes x {N_REPS} reps x "
        f"{len(MODELS)} models"
    )
    print(f"📏 Workload sizes: {WORKLOAD_SIZES}")
    print(f"🤖 Models: {MODELS}")
    print(f"🌡️ Temperature: {TEMPERATURE}")
    print(f"🎯 Scenario: {SCENARIO}, optimal parameters: {OPTIMAL_PARAMS[SCENARIO]}")

    results_db: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(workload_size): {model: [] for model in MODELS}
        for workload_size in WORKLOAD_SIZES
    }
    evaluation_args = OPTIMAL_PARAMS[SCENARIO]

    for workload_size in WORKLOAD_SIZES:
        size_key = str(workload_size)
        print(f"\n📊 --- Workload size {workload_size} ---")

        for rep in range(N_REPS):
            seed = workload_seed(workload_size, rep)
            print(f"  📢 Rep {rep + 1}/{N_REPS} (seed {seed})")
            generation_seconds = generate_workload(workload_size, seed)

            baseline_start = time.perf_counter()
            try:
                optimal_metrics, optimal_solver_objective, optimal_objective = solve_and_evaluate(
                    evaluation_args,
                    evaluation_args,
                )
                baseline_solve_seconds = time.perf_counter() - baseline_start
                baseline_error = None
            except Exception as exc:
                optimal_metrics = {}
                optimal_solver_objective = None
                optimal_objective = None
                baseline_solve_seconds = time.perf_counter() - baseline_start
                baseline_error = f"Optimal baseline error: {exc}"

            for model in MODELS:
                print(f"    > Model: {model:<12}...", end="", flush=True)
                record: dict[str, Any] = {
                    "workload_size": workload_size,
                    "rep": rep,
                    "seed": seed,
                    "scenario": SCENARIO,
                    "model": model,
                    "temperature": TEMPERATURE,
                    "n_datacentres": N_DATACENTRES,
                    "datacentre_capacity": DATACENTRE_CAPACITY,
                    "generation_seconds": generation_seconds,
                    "baseline_solve_seconds": baseline_solve_seconds,
                    "optimal_tool_args": evaluation_args,
                    "optimal_metrics": optimal_metrics,
                    "optimal_solver_objective": optimal_solver_objective,
                    "optimal_objective": optimal_objective,
                }

                if baseline_error is not None or optimal_objective is None:
                    record["error"] = baseline_error or "Could not evaluate optimal objective"
                    record["total_case_seconds"] = generation_seconds + baseline_solve_seconds
                    results_db[size_key][model].append(record)
                    print(" ❌ Baseline failed")
                    continue

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT},
                ]

                agent_start = time.perf_counter()
                model_seconds = 0.0
                tool_seconds = 0.0
                try:
                    model_start = time.perf_counter()
                    response = ollama.chat(
                        model=model,
                        messages=messages,
                        tools=TOOL_DEFINITION,
                        options={"seed": seed, "temperature": TEMPERATURE},
                    )
                    model_seconds = time.perf_counter() - model_start

                    response_data = response_to_dict(response)
                    count_fields = extract_ollama_count_fields(response_data)
                    record["ollama_count_fields"] = count_fields
                    record["total_token_count"] = total_token_count(count_fields)

                    tool_calls = response_data["message"].get("tool_calls", [])
                    if not tool_calls:
                        agent_end_to_end_seconds = time.perf_counter() - agent_start
                        record.update(
                            {
                                "error": "No tool call from agent",
                                "model_response_seconds": model_seconds,
                                "agent_tool_seconds": tool_seconds,
                                "agent_end_to_end_seconds": agent_end_to_end_seconds,
                                "total_case_seconds": generation_seconds
                                + baseline_solve_seconds
                                + agent_end_to_end_seconds,
                            }
                        )
                        results_db[size_key][model].append(record)
                        print(" ❌ Agent failed (No Tool)")
                        continue

                    tool_call = tool_calls[0]
                    agent_fn_name = tool_call["function"]["name"]
                    agent_fn_args = tool_call["function"]["arguments"]
                    record["agent_tool_name"] = agent_fn_name
                    record["agent_tool_args"] = agent_fn_args

                    if agent_fn_name != "solve_allocation":
                        agent_end_to_end_seconds = time.perf_counter() - agent_start
                        record.update(
                            {
                                "error": f"Agent called unexpected tool: {agent_fn_name}",
                                "model_response_seconds": model_seconds,
                                "agent_tool_seconds": tool_seconds,
                                "agent_end_to_end_seconds": agent_end_to_end_seconds,
                                "total_case_seconds": generation_seconds
                                + baseline_solve_seconds
                                + agent_end_to_end_seconds,
                            }
                        )
                        results_db[size_key][model].append(record)
                        print(f" ⚠️ Agent skipped ({agent_fn_name})")
                        continue

                    tool_start = time.perf_counter()
                    agent_metrics, agent_solver_objective, agent_objective = solve_and_evaluate(
                        agent_fn_args,
                        evaluation_args,
                    )
                    tool_seconds = time.perf_counter() - tool_start
                    agent_end_to_end_seconds = time.perf_counter() - agent_start

                    record.update(
                        {
                            "agent_metrics": agent_metrics,
                            "agent_solver_objective": agent_solver_objective,
                            "agent_objective": agent_objective,
                            "agent_evaluation_objective_args": evaluation_args,
                            "model_response_seconds": model_seconds,
                            "agent_tool_seconds": tool_seconds,
                            "agent_end_to_end_seconds": agent_end_to_end_seconds,
                            "total_case_seconds": generation_seconds
                            + baseline_solve_seconds
                            + agent_end_to_end_seconds,
                        }
                    )
                    if agent_objective is None:
                        record["error"] = "Could not evaluate agent solution objective"
                    else:
                        record.update(
                            compute_regret(
                                agent_metrics,
                                optimal_metrics,
                                agent_objective,
                                optimal_objective,
                            )
                        )

                    results_db[size_key][model].append(record)
                    regret = record.get("metric_regret_percent")
                    tokens = record.get("total_token_count")
                    print(
                        f" ✅ regret={regret if regret is not None else 'NA'}%, "
                        f"tokens={tokens if tokens is not None else 'NA'}"
                    )

                except Exception as exc:
                    agent_end_to_end_seconds = time.perf_counter() - agent_start
                    record.update(
                        {
                            "error": f"Agent run error: {exc}",
                            "model_response_seconds": model_seconds,
                            "agent_tool_seconds": tool_seconds,
                            "agent_end_to_end_seconds": agent_end_to_end_seconds,
                            "total_case_seconds": generation_seconds
                            + baseline_solve_seconds
                            + agent_end_to_end_seconds,
                        }
                    )
                    results_db[size_key][model].append(record)
                    print(f" ❌ Agent error: {exc}")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    table_rows = build_average_metrics_table(results_db)
    write_metrics_table_csv(table_rows)
    write_metrics_table_markdown(table_rows)

    print(f"\n✅ Scalability experiment complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Metrics table CSV saved to {METRICS_TABLE_CSV}")
    print(f"✅ Metrics table Markdown saved to {METRICS_TABLE_MD}")

if __name__ == "__main__":
    asyncio.run(run_scalability_experiment())
