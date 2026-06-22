#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create standard-deviation tables from parameter-accuracy sweep results.

This post-processing script reads the raw JSON outputs produced by
``parameter_accuracy_model_sweep.py`` and
``parameter_accuracy_temperature_sweep.py``. It does not rerun any models.
For each scenario and sweep column, it computes the population standard
deviation of the existing ``parameter_selection_score`` values and writes CSV
and Markdown tables using the same paper-experiment naming style as the other
summary outputs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_accuracy_model_sweep import MODELS  # noqa: E402
from parameter_accuracy_temperature_sweep import TEMPERATURES  # noqa: E402
from parameter_selection_accuracy import (  # noqa: E402
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    write_accuracy_table_csv,
    write_accuracy_table_markdown,
)

EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"

MODEL_SWEEP_RESULTS_FILE = EXPERIMENT_DATA_DIR / "parameter_accuracy_model_sweep_results.json"
MODEL_SWEEP_STD_TABLE_CSV = (
    EXPERIMENT_DATA_DIR / "parameter_accuracy_model_sweep_parameter_selection_standard_deviation_table.csv"
)
MODEL_SWEEP_STD_TABLE_MD = (
    EXPERIMENT_DATA_DIR / "parameter_accuracy_model_sweep_parameter_selection_standard_deviation_table.md"
)

TEMPERATURE_SWEEP_RESULTS_FILE = (
    EXPERIMENT_DATA_DIR / "parameter_accuracy_temperature_sweep_results.json"
)
TEMPERATURE_SWEEP_STD_TABLE_CSV = (
    EXPERIMENT_DATA_DIR / "parameter_accuracy_temperature_sweep_parameter_selection_standard_deviation_table.csv"
)
TEMPERATURE_SWEEP_STD_TABLE_MD = (
    EXPERIMENT_DATA_DIR / "parameter_accuracy_temperature_sweep_parameter_selection_standard_deviation_table.md"
)


def load_results(path: Path) -> dict[str, Any]:
    """Load a raw parameter-accuracy sweep JSON results file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run the corresponding parameter-accuracy sweep first."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}, got {type(data).__name__}")

    return data


def infer_columns(results: dict[str, Any], configured_columns: list[str]) -> list[str]:
    """Return configured column order, appending any extra columns found in the input."""
    ordered_columns = [column for column in configured_columns if column in results]
    extra_columns = [column for column in results if column not in configured_columns]
    return [*ordered_columns, *extra_columns]


def population_standard_deviation(values: list[float]) -> float | None:
    """Return the population standard deviation, or None when no values exist."""
    if not values:
        return None

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def build_parameter_selection_std_table(
    results: dict[str, Any],
    columns: list[str],
) -> list[dict[str, str]]:
    """Create scenario rows and sweep-value columns with parameter-selection-score std dev."""
    table_rows: list[dict[str, str]] = []

    for scenario in SCENARIO_ORDER:
        row = {"scenario": SCENARIO_LABELS[scenario]}
        for column in columns:
            runs = results.get(column, {}).get(scenario, [])
            if not isinstance(runs, list):
                row[column] = ""
                continue

            scores = [
                float(run["parameter_selection_score"])
                for run in runs
                if isinstance(run, dict) and run.get("parameter_selection_score") is not None
            ]
            std_dev = population_standard_deviation(scores)
            row[column] = "" if std_dev is None else f"{std_dev:.4f}"
        table_rows.append(row)

    return table_rows


def print_std_table(table_rows: list[dict[str, str]], columns: list[str], title: str) -> None:
    """Print a standard-deviation table to stdout."""
    headers = ["Scenario", *columns]
    rows = [[row["scenario"], *[row[column] for column in columns]] for row in table_rows]
    widths = [len(header) for header in headers]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print(f"\n{title}")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def write_std_outputs(
    results_file: Path,
    configured_columns: list[str],
    csv_path: Path,
    markdown_path: Path,
    title: str,
) -> None:
    """Load one sweep result file and write its standard-deviation tables."""
    results = load_results(results_file)
    columns = infer_columns(results, configured_columns)
    if not columns:
        raise ValueError(f"No sweep columns found in {results_file}")

    print(f"📥 Reading: {results_file}")
    print(f"📊 Columns: {columns}")

    table_rows = build_parameter_selection_std_table(results, columns)
    write_accuracy_table_csv(table_rows, columns, csv_path)
    write_accuracy_table_markdown(table_rows, columns, markdown_path)
    print_std_table(table_rows, columns, title)

    print(f"✅ Parameter-selection standard-deviation table CSV saved to {csv_path}")
    print(f"✅ Parameter-selection standard-deviation table Markdown saved to {markdown_path}")


def run_parameter_accuracy_selection_std_tables() -> None:
    """Write standard-deviation tables for model and temperature parameter sweeps."""
    print("🚀 Parameter-accuracy parameter-selection standard-deviation tables")
    print("📏 Statistic: population standard deviation of parameter_selection_score")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    write_std_outputs(
        MODEL_SWEEP_RESULTS_FILE,
        MODELS,
        MODEL_SWEEP_STD_TABLE_CSV,
        MODEL_SWEEP_STD_TABLE_MD,
        "Parameter-selection score standard deviation by model",
    )
    write_std_outputs(
        TEMPERATURE_SWEEP_RESULTS_FILE,
        [str(temperature) for temperature in TEMPERATURES],
        TEMPERATURE_SWEEP_STD_TABLE_CSV,
        TEMPERATURE_SWEEP_STD_TABLE_MD,
        "Parameter-selection score standard deviation by temperature",
    )


if __name__ == "__main__":
    run_parameter_accuracy_selection_std_tables()
