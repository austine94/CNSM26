import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "optimisation_engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from flexmedia_delivery_instance import flexmedia_delivery_instance
from config import COMPUTE_NODES_CSV, RAW_DIR, VIEWERS_CSV
from database import ingest_csvs


def generate_snapshot(n_viewers=24, n_nodes=6, n_hotspots=3, missing_rate=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    inst = flexmedia_delivery_instance(n_datacentres=n_nodes, n_consumers=n_viewers, n_hotspots=n_hotspots)

    with open(VIEWERS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["viewer_id", "longitude", "latitude"])
        writer.writeheader()
        for viewer_id, (lon, lat) in enumerate(inst.consumer_locations):
            writer.writerow({"viewer_id": viewer_id, "longitude": lon, "latitude": lat})

    fields = ["node_id", "longitude", "latitude", "capacity", "provision_financial_cost", "provision_energy_cost"]
    for viewer_id in range(n_viewers):
        fields.extend([f"viewer_{viewer_id}_financial_cost", f"viewer_{viewer_id}_energy_cost", f"viewer_{viewer_id}_latency"])

    def maybe_missing(value):
        return "" if random.random() < missing_rate else value

    with open(COMPUTE_NODES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for node_id, (lon, lat) in enumerate(inst.datacentre_locations):
            row = {
                "node_id": node_id,
                "longitude": lon,
                "latitude": lat,
                "capacity": maybe_missing(int(max(1, round(inst.datacentre_capacities[node_id])))),
                "provision_financial_cost": maybe_missing(float(inst.datacentre_costs[node_id])),
                "provision_energy_cost": maybe_missing(float(np.mean(inst.energy_costs[node_id, :]))),
            }
            for viewer_id in range(n_viewers):
                row[f"viewer_{viewer_id}_financial_cost"] = maybe_missing(float(inst.consumer_costs[node_id, viewer_id]))
                row[f"viewer_{viewer_id}_energy_cost"] = maybe_missing(float(inst.energy_costs[node_id, viewer_id]))
                row[f"viewer_{viewer_id}_latency"] = maybe_missing(float(inst.consumer_latencies[node_id, viewer_id]))
            writer.writerow(row)
    return ingest_csvs(VIEWERS_CSV, COMPUTE_NODES_CSV)


def main():
    parser = argparse.ArgumentParser(description="Generate changing demo workloads and ingest them into SQLite.")
    parser.add_argument("--loop", action="store_true", help="Regenerate and ingest every interval seconds.")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--viewers", type=int, default=24)
    parser.add_argument("--nodes", type=int, default=6)
    args = parser.parse_args()
    while True:
        summary = generate_snapshot(args.viewers, args.nodes)
        print(f"Generated and ingested workload: {summary}")
        if not args.loop:
            break
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
