from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_EXPERIMENTS_DIR = PROJECT_ROOT / "paper_experiments"
sys.path.insert(0, str(PAPER_EXPERIMENTS_DIR))

from plot_parameter_std_and_tool_reliability_temperature_sweep import (
    aggregate_tool_success_by_scenario_temperature,
)


class ParameterStdAndToolReliabilityPlotTests(unittest.TestCase):
    def test_aggregate_tool_success_by_scenario_temperature(self):
        results = {
            "0.25": {
                "focus_cost": [
                    {"correct_tool_called": True},
                    {"correct_tool_called": False},
                ],
                "focus_energy": [{"correct_tool_called": True}],
            },
            "0.0": {
                "focus_cost": [{"correct_tool_called": True}],
                "focus_energy": [
                    {"correct_tool_called": False},
                    {"correct_tool_called": False},
                ],
            },
        }

        series = aggregate_tool_success_by_scenario_temperature(results)

        self.assertEqual(series["Cost"], [(0.0, 1.0), (0.25, 0.5)])
        self.assertEqual(series["Energy"], [(0.0, 0.0), (0.25, 1.0)])


if __name__ == "__main__":
    unittest.main()
