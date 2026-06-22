#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate scalable FlexMedia workload instances for paper experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
ENGINE_PATH = PROJECT_ROOT / "optimisation_engine"
if str(ENGINE_PATH) not in sys.path:
    sys.path.append(str(ENGINE_PATH))

from flexmedia_delivery_instance import flexmedia_delivery_instance

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "current_problem.json"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for scalability workload generation."""
    parser = argparse.ArgumentParser(description="Generate a scalable FlexMedia scenario")
    parser.add_argument("seed", type=int, nargs="?", default=None, help="Random seed (optional)")
    parser.add_argument(
        "--n-consumers",
        type=int,
        default=100,
        help="Number of consumers/viewers in the generated workload",
    )
    parser.add_argument(
        "--n-datacentres",
        type=int,
        default=5,
        help="Number of datacentres in the generated workload",
    )
    parser.add_argument(
        "--n-hotspots",
        type=int,
        default=5,
        help="Number of consumer hotspots in the generated workload",
    )
    parser.add_argument(
        "--datacentre-capacity",
        type=float,
        default=1100.0,
        help="Capacity assigned to each datacentre after generation",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the generated problem JSON",
    )
    return parser


def generate_scalability_problem(
    *,
    seed: int | None,
    n_consumers: int,
    n_datacentres: int,
    n_hotspots: int,
    datacentre_capacity: float,
    output_path: Path,
) -> dict[str, object]:
    """Generate and persist a scalable FlexMedia workload instance."""
    if n_consumers <= 0:
        raise ValueError("--n-consumers must be positive")
    if n_datacentres <= 0:
        raise ValueError("--n-datacentres must be positive")
    if n_hotspots <= 0:
        raise ValueError("--n-hotspots must be positive")
    if datacentre_capacity <= 0:
        raise ValueError("--datacentre-capacity must be positive")

    seed_val = seed if seed is not None else int(np.random.randint(0, 1_000_000))
    np.random.seed(seed_val)

    inst = flexmedia_delivery_instance(n_datacentres, n_consumers, n_hotspots)
    fixed_capacities = np.full(n_datacentres, math.ceil(datacentre_capacity), dtype=float)

    data: dict[str, object] = {
        "seed": seed_val,
        "n_datacentres": n_datacentres,
        "n_consumers": n_consumers,
        "n_hotspots": n_hotspots,
        "datacentre_capacity": math.ceil(datacentre_capacity),
        "total_datacentre_capacity": float(fixed_capacities.sum()),
        "datacentre_locations": inst.datacentre_locations.tolist(),
        "consumer_locations": inst.consumer_locations.tolist(),
        "datacentre_costs": inst.datacentre_costs.tolist(),
        "datacentre_capacities": fixed_capacities.tolist(),
        "consumer_costs": inst.consumer_costs.tolist(),
        "consumer_latencies": inst.consumer_latencies.tolist(),
        "energy_costs": inst.energy_costs.tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


def main() -> None:
    """Generate a scenario from CLI arguments."""
    args = build_parser().parse_args()
    data = generate_scalability_problem(
        seed=args.seed,
        n_consumers=args.n_consumers,
        n_datacentres=args.n_datacentres,
        n_hotspots=args.n_hotspots,
        datacentre_capacity=args.datacentre_capacity,
        output_path=args.output_path,
    )
    print(
        f"Scalability scenario saved to {args.output_path} "
        f"(seed={data['seed']}, consumers={data['n_consumers']}, "
        f"datacentres={data['n_datacentres']}, total_capacity={data['total_datacentre_capacity']})"
    )


if __name__ == "__main__":
    main()
