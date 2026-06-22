"""Create paper-ready PDF plots for the illustrative benchmark experiment.

The script reads illustrative benchmark results from the experiment_data
folder, creates one PDF per scenario, and creates a combined 2 x 3 PDF
containing all scenarios.
It intentionally uses only the Python standard library so plots can be
regenerated in minimal execution environments.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev


# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "illustrative_experiment_results.json"
PLOTS_DIR = SCRIPT_DIR / "paper_plots"

SCENARIO_LABELS = {
    "focus_cost": "Cost priority",
    "focus_energy": "Energy priority",
    "focus_latency": "Latency priority",
    "tradeoff_cost_energy": "Cost-energy trade-off",
    "tradeoff_cost_latency": "Cost-latency trade-off",
    "tradeoff_energy_latency": "Energy-latency trade-off",
}

METRIC_LABELS = {
    "cost": "Cost",
    "energy": "Energy",
    "latency": "Latency",
}

SCENARIO_ORDER = [
    "focus_cost",
    "focus_energy",
    "focus_latency",
    "tradeoff_cost_energy",
    "tradeoff_cost_latency",
    "tradeoff_energy_latency",
]
METRIC_ORDER = ["cost", "energy", "latency"]
BAR_COLOURS = {
    "cost": (0.18, 0.38, 0.62),
    "energy": (0.18, 0.55, 0.32),
    "latency": (0.72, 0.22, 0.20),
}


def load_results(results_file: Path | None = None) -> dict[str, list[dict]]:
    """Load benchmark-style JSON results from the illustrative experiment."""
    source = results_file or RESULTS_FILE
    if not source.exists():
        raise FileNotFoundError(
            f"No illustrative experiment results found at {source}. "
            "Run paper_experiments/illustrative_experiment.py before plotting."
        )

    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)

    valid_runs = 0
    skipped_runs = 0
    cleaned: dict[str, list[dict]] = {}
    for scenario, runs in data.items():
        cleaned[scenario] = []
        for run in runs:
            metrics = run.get("metrics", {})
            if not all(metric in metrics for metric in METRIC_ORDER):
                skipped_runs += 1
                continue
            cleaned[scenario].append(run)
            valid_runs += 1

    print(f"Processed {valid_runs} valid runs from {source}.")
    if skipped_runs:
        print(f"Skipped {skipped_runs} runs due to incomplete metrics.")

    return cleaned


def calculate_metric_maxima(results: dict[str, list[dict]]) -> dict[str, float]:
    """Calculate maxima used to normalise metrics onto a common y-axis."""
    maxima = {metric: 0.0 for metric in METRIC_ORDER}
    for runs in results.values():
        for run in runs:
            for metric in METRIC_ORDER:
                maxima[metric] = max(maxima[metric], float(run["metrics"][metric]))
    return {metric: value or 1.0 for metric, value in maxima.items()}


def scenario_summary(
    results: dict[str, list[dict]], scenario: str, maxima: dict[str, float]
) -> list[dict[str, float | str]]:
    """Return mean and standard deviation for each normalised metric."""
    runs = results.get(scenario, [])
    summary = []
    for metric in METRIC_ORDER:
        values = [float(run["metrics"][metric]) / maxima[metric] for run in runs]
        metric_mean = mean(values) if values else 0.0
        metric_sd = stdev(values) if len(values) > 1 else 0.0
        summary.append(
            {
                "metric": metric,
                "label": METRIC_LABELS[metric],
                "mean": metric_mean,
                "sd": metric_sd,
            }
        )
    return summary


def pdf_escape(value: str) -> str:
    """Escape text for a PDF string literal."""
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfPage:
    """A small PDF page builder for simple vector bar charts."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = []

    def text(self, x: float, y: float, value: str, size: int = 10) -> None:
        self.commands.append(
            f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({pdf_escape(value)}) Tj ET"
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, width: float = 1) -> None:
        self.commands.append(
            f"{width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"
        )

    def stroke_rect(self, x: float, y: float, width: float, height: float) -> None:
        self.commands.append(f"0 0 0 RG {x:.2f} {y:.2f} {width:.2f} {height:.2f} re S")

    def fill_rect(
        self, x: float, y: float, width: float, height: float, colour: tuple[float, float, float]
    ) -> None:
        r, g, b = colour
        self.commands.append(
            f"{r:.2f} {g:.2f} {b:.2f} rg {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f"
        )

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


def draw_scenario_panel(
    page: PdfPage,
    summary: list[dict[str, float | str]],
    title: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    """Draw a normalised scenario panel on a PDF page."""
    title_y = y + height - 16
    plot_x = x + 42
    plot_y = y + 44
    plot_width = width - 58
    plot_height = height - 84

    page.text(x + 4, title_y, title, size=16)
    page.stroke_rect(plot_x, plot_y, plot_width, plot_height)
    page.text(plot_x, plot_y + plot_height + 8, "Normalised mean +/- SD", size=16)

    for tick in range(0, 6):
        value = tick / 5
        tick_y = plot_y + value * plot_height
        page.line(plot_x - 4, tick_y, plot_x, tick_y)
        page.text(plot_x - 31, tick_y - 3, f"{value:.1f}", size=16)
        if tick:
            page.line(plot_x, tick_y, plot_x + plot_width, tick_y, width=0.25)

    bar_count = len(summary)
    slot_width = plot_width / bar_count
    bar_width = min(36, slot_width * 0.48)

    for index, item in enumerate(summary):
        metric = str(item["metric"])
        metric_mean = float(item["mean"])
        metric_sd = float(item["sd"])
        bar_x = plot_x + slot_width * index + (slot_width - bar_width) / 2
        bar_height = max(0, min(metric_mean, 1.1)) * plot_height
        colour = BAR_COLOURS[metric]

        page.fill_rect(bar_x, plot_y, bar_width, bar_height, colour)
        page.text(bar_x - 2, plot_y - 18, str(item["label"]), size=16)
        page.text(bar_x + 2, plot_y + bar_height + 5, f"{metric_mean:.2f}", size=16)

        error_low = max(0, metric_mean - metric_sd)
        error_high = min(1.1, metric_mean + metric_sd)
        error_x = bar_x + bar_width / 2
        low_y = plot_y + error_low * plot_height
        high_y = plot_y + error_high * plot_height
        cap_width = bar_width * 0.55
        page.line(error_x, low_y, error_x, high_y, width=0.75)
        page.line(error_x - cap_width / 2, low_y, error_x + cap_width / 2, low_y, width=0.75)
        page.line(error_x - cap_width / 2, high_y, error_x + cap_width / 2, high_y, width=0.75)


def save_individual_scenario_plots(results: dict[str, list[dict]], maxima: dict[str, float]) -> None:
    """Save one PDF for each scenario in the illustrative experiment."""
    for scenario in SCENARIO_ORDER:
        if not results.get(scenario):
            print(f"Skipping missing scenario: {scenario}")
            continue

        page = PdfPage(width=612, height=432)
        page.text(36, 402, "Illustrative experiment results", size=16)
        draw_scenario_panel(
            page,
            scenario_summary(results, scenario, maxima),
            SCENARIO_LABELS.get(scenario, scenario),
            x=48,
            y=36,
            width=516,
            height=348,
        )

        output_file = PLOTS_DIR / f"illustrative_experiment_{scenario}.pdf"
        page.save(output_file)
        print(f"Saved plot: {output_file}")


def save_combined_plot(results: dict[str, list[dict]], maxima: dict[str, float]) -> None:
    """Save all six scenario plots in a combined 2 x 3 PDF."""
    page = PdfPage(width=1080, height=720)
    page.text(36, 684, "Illustrative experiment results", size=18)

    panel_positions = [
        (36, 386, 324, 284),
        (390, 386, 324, 284),
        (744, 386, 324, 284),
        (36, 54, 324, 284),
        (390, 54, 324, 284),
        (744, 54, 324, 284),
    ]

    for scenario, (x, y, width, height) in zip(SCENARIO_ORDER, panel_positions):
        draw_scenario_panel(
            page,
            scenario_summary(results, scenario, maxima),
            SCENARIO_LABELS.get(scenario, scenario),
            x=x,
            y=y,
            width=width,
            height=height,
        )

    output_file = PLOTS_DIR / "illustrative_experiment_all.pdf"
    page.save(output_file)
    print(f"Saved combined plot: {output_file}")


def plot_illustrative_experiment(results: dict[str, list[dict]]) -> None:
    """Create all individual and combined illustrative experiment plots."""
    maxima = calculate_metric_maxima(results)
    save_individual_scenario_plots(results, maxima)
    save_combined_plot(results, maxima)


if __name__ == "__main__":
    plot_illustrative_experiment(load_results())
