#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute parameter-selection accuracy from varying-prompt regret model sweep results.

This post-processing script reuses the raw records produced by
``varying_prompt_regret_model_sweep.py`` instead of rerunning the models. It
scores each record's selected ``solve_allocation`` alpha/beta values against
the canonical scenario parameters and writes accuracy tables with the same model
columns as the varying-prompt regret model-sweep table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_selection_accuracy import (  # noqa: E402
    SCENARIO_ORDER,
    TOLERANCE,
    build_accuracy_table,
    print_accuracy_table,
    score_parameter_selection,
    write_accuracy_table_csv,
    write_accuracy_table_markdown,
)
from varying_prompt_regret_model_sweep import MODELS  # noqa: E402

EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
REGRET_RESULTS_FILE = EXPERIMENT_DATA_DIR / "varying_prompt_regret_model_sweep_results.json"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "varying_prompt_regret_model_sweep_parameter_selection_accuracy_results.json"
ACCURACY_TABLE_CSV = EXPERIMENT_DATA_DIR / "varying_prompt_regret_model_sweep_parameter_selection_accuracy_table.csv"
ACCURACY_TABLE_MD = EXPERIMENT_DATA_DIR / "varying_prompt_regret_model_sweep_parameter_selection_accuracy_table.md"


def load_regret_results(path: Path = REGRET_RESULTS_FILE) -> dict[str, Any]:
    """Load raw regret model-sweep JSON results."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run paper_experiments/varying_prompt_regret_model_sweep.py first."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}, got {type(data).__name__}")

    return data


def infer_models(results: dict[str, Any]) -> list[str]:
    """Return configured model order, appending any extra models found in the input."""
    configured_models = [model for model in MODELS if model in results]
    extra_models = [model for model in results if model not in MODELS]
    return [*configured_models, *extra_models]


def score_varying_prompt_regret_model_sweep_parameters(
    regret_results: dict[str, Any],
    models: list[str],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Score agent parameter choices already stored in regret model-sweep records."""
    accuracy_results: dict[str, dict[str, list[dict[str, Any]]]] = {
        model: {scenario: [] for scenario in SCENARIO_ORDER}
        for model in models
    }

    for model in models:
        model_results = regret_results.get(model, {})
        if not isinstance(model_results, dict):
            continue

        for scenario in SCENARIO_ORDER:
            runs = model_results.get(scenario, [])
            if not isinstance(runs, list):
                continue

            for run in runs:
                if not isinstance(run, dict):
                    continue

                raw_agent_args = run.get("agent_tool_args")
                agent_args = raw_agent_args if isinstance(raw_agent_args, dict) else None
                scored_record: dict[str, Any] = {
                    "rep": run.get("rep"),
                    "seed": run.get("seed"),
                    "model": run.get("model", model),
                    "temperature": run.get("temperature"),
                    "prompt_category": run.get("prompt_category", scenario),
                    "optimal_tool_args": run.get("optimal_tool_args"),
                    "agent_tool_args": raw_agent_args,
                    "source_error": run.get("error"),
                    "tolerance": TOLERANCE,
                    "source_results_file": str(REGRET_RESULTS_FILE.relative_to(SCRIPT_DIR.parent)),
                }
                scored_record.update(score_parameter_selection(scenario, agent_args))
                accuracy_results[model][scenario].append(scored_record)

    return accuracy_results


def run_varying_prompt_regret_model_sweep_parameter_selection_accuracy() -> None:
    """Compute and persist parameter-selection accuracy from varying-prompt regret model-sweep data."""
    regret_results = load_regret_results()
    models = infer_models(regret_results)
    if not models:
        raise ValueError(f"No model result blocks found in {REGRET_RESULTS_FILE}")

    print("🚀 Varying-prompt regret model-sweep parameter-selection accuracy")
    print(f"📥 Reading: {REGRET_RESULTS_FILE}")
    print(f"🤖 Models: {models}")
    print(f"📏 Correct when alpha/beta are within ±{TOLERANCE} of target")

    accuracy_results = score_varying_prompt_regret_model_sweep_parameters(regret_results, models)

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(accuracy_results, f, indent=2)

    table_rows = build_accuracy_table(accuracy_results, models)
    write_accuracy_table_csv(table_rows, models, ACCURACY_TABLE_CSV)
    write_accuracy_table_markdown(table_rows, models, ACCURACY_TABLE_MD)
    print_accuracy_table(table_rows, models)

    print(f"\n✅ Parameter-selection accuracy results saved to {RESULTS_FILE}")
    print(f"✅ Accuracy table CSV saved to {ACCURACY_TABLE_CSV}")
    print(f"✅ Accuracy table Markdown saved to {ACCURACY_TABLE_MD}")


if __name__ == "__main__":
    run_varying_prompt_regret_model_sweep_parameter_selection_accuracy()
