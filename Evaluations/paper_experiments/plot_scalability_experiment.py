#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot scalability experiment aggregate metrics as PDF line charts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from regret_temperature_sweep import metric_regret_percent_for_summary
from scalability_experiment import MODELS, WORKLOAD_SIZES, average

EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
PLOTS_DIR = SCRIPT_DIR / "paper_plots"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "scalability_experiment_results.json"

MODEL_COLOURS: dict[str, tuple[float, float, float]] = {
    "interstellarninja/llama3.1-8b-tools:latest": (0.12, 0.47, 0.71),
    "llama3.2:1b": (1.00, 0.50, 0.05),
    "llama3.2:latest": (0.17, 0.63, 0.17),
    "aratan/qwen3-4b-tools:latest": (0.84, 0.15, 0.16),
    "lukaspetrik/gemma3-tools:4b": (0.58, 0.40, 0.74),
}

MODEL_LABELS: dict[str, str] = {
    "interstellarninja/llama3.1-8b-tools:latest": "Llama3.1:8B",
    "llama3.2:1b": "Llama3.2:1B",
    "llama3.2:latest": "Llama3.2:3B",
    "aratan/qwen3-4b-tools:latest": "Qwen3:4B",
    "lukaspetrik/gemma3-tools:4b": "Gemma3:4B",
}

PLOTS = [
    {
        "metric": "metric_regret_percent",
        "title": " ",
        "ylabel": "Average absolute regret (%)",
        "output": "scalability_average_regret.pdf",
        "y_max": 120.0,
    },
    {
        "metric": "total_token_count",
        "title": "Average total token count by workload size",
        "ylabel": "Average total tokens",
        "output": "scalability_average_tokens.pdf",
    },
    {
        "metric": "agent_end_to_end_seconds",
        "title": "Average agent end-to-end time by workload size",
        "ylabel": "Average agent e2e time (s)",
        "output": "scalability_average_agent_time.pdf",
    },
    {
        "metric": "total_case_seconds",
        "title": "Average total case time by workload size",
        "ylabel": "Average total case time (s)",
        "output": "scalability_average_total_case_time.pdf",
    },
]


class PdfPage:
    """Tiny PDF writer with enough drawing primitives for line charts."""

    def __init__(self, width: float = 820, height: float = 504):
        self.width = width
        self.height = height
        self.commands: list[str] = []

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def text(self, x: float, y: float, text: str, size: int = 10) -> None:
        self.commands.append(f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({self._escape(text)}) Tj ET")

    def rotated_text(self, x: float, y: float, text: str, size: int = 10) -> None:
        """Draw text rotated 90 degrees counter-clockwise around its origin."""
        self.commands.append(f"BT /F1 {size} Tf 0 1 -1 0 {x:.2f} {y:.2f} Tm ({self._escape(text)}) Tj ET")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float = 1.0,
        colour: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        r, g, b = colour
        self.commands.append(
            f"{r:.2f} {g:.2f} {b:.2f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        width: float = 1.25,
        colour: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        if len(points) < 2:
            return
        r, g, b = colour
        commands = [f"{r:.2f} {g:.2f} {b:.2f} RG {width:.2f} w"]
        start_x, start_y = points[0]
        commands.append(f"{start_x:.2f} {start_y:.2f} m")
        for x, y in points[1:]:
            commands.append(f"{x:.2f} {y:.2f} l")
        commands.append("S")
        self.commands.append(" ".join(commands))

    def fill_rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        colour: tuple[float, float, float],
    ) -> None:
        r, g, b = colour
        self.commands.append(
            f"{r:.2f} {g:.2f} {b:.2f} rg {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f"
        )

    def marker(self, x: float, y: float, colour: tuple[float, float, float], size: float = 3.0) -> None:
        self.fill_rect(x - size / 2, y - size / 2, size, size, colour)

    def save(self, output_file: Path) -> None:
        """Write the page to a single-page PDF file."""
        content = "\n".join(self.commands).encode("latin-1")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
            ).encode("latin-1"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(content)).encode("latin-1") + b" >>\nstream\n" + content + b"\nendstream",
        ]

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("wb") as f:
            f.write(b"%PDF-1.4\n")
            offsets = [0]
            for index, obj in enumerate(objects, start=1):
                offsets.append(f.tell())
                f.write(f"{index} 0 obj\n".encode("latin-1"))
                f.write(obj)
                f.write(b"\nendobj\n")
            xref_offset = f.tell()
            f.write(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
            f.write(b"0000000000 65535 f \n")
            for offset in offsets[1:]:
                f.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
            f.write(
                (
                    f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                    f"startxref\n{xref_offset}\n%%EOF\n"
                ).encode("latin-1")
            )


def load_results(results_file: Path | None = None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load scalability experiment results."""
    path = results_file or RESULTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"No scalability results found at {path}. Run scalability_experiment.py first.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_metric(
    results: dict[str, dict[str, list[dict[str, Any]]]],
    metric: str,
) -> dict[str, list[tuple[int, float | None]]]:
    """Average a metric by workload size for every model."""
    series: dict[str, list[tuple[int, float | None]]] = {model: [] for model in MODELS}
    for workload_size in WORKLOAD_SIZES:
        size_key = str(workload_size)
        for model in MODELS:
            runs = results.get(size_key, {}).get(model, [])
            if metric == "metric_regret_percent":
                values = [
                    regret_percent
                    for run in runs
                    if (regret_percent := metric_regret_percent_for_summary(run)) is not None
                ]
            else:
                values = [float(run[metric]) for run in runs if run.get(metric) is not None]
            series[model].append((workload_size, average(values)))
    return series


def nice_axis_max(value: float) -> float:
    """Return a readable upper bound for a plot axis."""
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


def draw_line_plot(
    series: dict[str, list[tuple[int, float | None]]],
    *,
    title: str,
    ylabel: str,
    output_file: Path,
    y_axis_max: float | None = None,
) -> None:
    """Draw a single PDF line plot for one aggregate metric."""
    page = PdfPage()
    page.text(54, 472, title, size=16)

    plot_x = 78
    plot_y = 68
    plot_width = 520
    plot_height = 348
    x_min = min(WORKLOAD_SIZES)
    x_max = max(WORKLOAD_SIZES)
    finite_values = [value for values in series.values() for _, value in values if value is not None]
    y_max = y_axis_max if y_axis_max is not None else nice_axis_max(max(finite_values) * 1.08) if finite_values else 1.0

    # Axes and grid.
    page.line(plot_x, plot_y, plot_x + plot_width, plot_y, width=1.0)
    page.line(plot_x, plot_y, plot_x, plot_y + plot_height, width=1.0)
    page.text(plot_x + plot_width / 2 - 42, 30, "Workload size", size=16)
    page.rotated_text(22, plot_y + plot_height / 2 - (len(ylabel) * 4), ylabel, size=16)

    for tick in range(0, 6):
        value = y_max * tick / 5
        tick_y = plot_y + (value / y_max) * plot_height
        page.line(plot_x - 4, tick_y, plot_x, tick_y, width=0.75)
        page.text(plot_x - 54, tick_y - 3, f"{value:.2f}" if y_max < 10 else f"{value:.0f}", size=16)
        if tick:
            page.line(plot_x, tick_y, plot_x + plot_width, tick_y, width=0.25, colour=(0.82, 0.82, 0.82))

    x_ticks = [50, 250, 500, 750, 1000]
    for tick in x_ticks:
        tick_x = plot_x + ((tick - x_min) / (x_max - x_min)) * plot_width
        page.line(tick_x, plot_y, tick_x, plot_y - 4, width=0.75)
        page.text(tick_x - 10, plot_y - 18, str(tick), size=16)

    def map_point(workload_size: int, value: float) -> tuple[float, float]:
        x = plot_x + ((workload_size - x_min) / (x_max - x_min)) * plot_width
        y = plot_y + (value / y_max) * plot_height
        return x, y

    # Lines.
    for model in MODELS:
        colour = MODEL_COLOURS[model]
        points = [map_point(size, value) for size, value in series[model] if value is not None]
        page.polyline(points, width=1.35, colour=colour)
        for x, y in points:
            page.marker(x, y, colour)

    # Legend.
    legend_x = 632
    legend_y = 394
    page.text(legend_x, legend_y + 20, "Model", size=16)
    for index, model in enumerate(MODELS):
        y = legend_y - index * 20
        colour = MODEL_COLOURS[model]
        page.line(legend_x, y, legend_x + 22, y, width=1.4, colour=colour)
        page.marker(legend_x + 11, y, colour)
        page.text(legend_x + 30, y - 4, MODEL_LABELS.get(model, model), size=16)

    page.save(output_file)
    print(f"Saved plot: {output_file}")


def plot_scalability_experiment(results: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    """Create all scalability line plots."""
    for plot_spec in PLOTS:
        series = aggregate_metric(results, str(plot_spec["metric"]))
        draw_line_plot(
            series,
            title=str(plot_spec["title"]),
            ylabel=str(plot_spec["ylabel"]),
            output_file=PLOTS_DIR / str(plot_spec["output"]),
            y_axis_max=plot_spec.get("y_max"),
        )


if __name__ == "__main__":
    plot_scalability_experiment(load_results())
