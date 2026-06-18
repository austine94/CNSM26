import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "optimisation_engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from flexmedia_delivery_instance import flexmedia_delivery_instance
from database import connect, now_iso


def policy_to_parameters(priority: str):
    text = priority.lower()
    if "energy" in text:
        return 0.0, 0.0
    if "latency" in text or "qoe" in text or "quality" in text:
        return 1.0, 0.5
    if "balanced" in text or "trade" in text:
        return 0.1, 0.5
    return 0.0, 1.0


def load_instance_from_db(db_path):
    with connect(db_path) as conn:
        viewers = conn.execute("SELECT * FROM viewers ORDER BY viewer_id").fetchall()
        nodes = conn.execute("SELECT * FROM compute_nodes ORDER BY node_id").fetchall()
        costs = conn.execute("SELECT * FROM viewer_node_costs ORDER BY node_id, viewer_id").fetchall()
    inst = flexmedia_delivery_instance(len(nodes), len(viewers), 1)
    inst.datacentre_locations = np.array([[r["longitude"], r["latitude"]] for r in nodes])
    inst.consumer_locations = np.array([[r["longitude"], r["latitude"]] for r in viewers])
    inst.datacentre_capacities = np.array([r["capacity"] for r in nodes])
    inst.datacentre_costs = np.array([r["provision_financial_cost"] for r in nodes])
    n_nodes, n_viewers = len(nodes), len(viewers)
    inst.consumer_costs = np.zeros((n_nodes, n_viewers))
    inst.energy_costs = np.zeros((n_nodes, n_viewers))
    inst.consumer_latencies = np.zeros((n_nodes, n_viewers))
    for row in costs:
        i, j = row["node_id"], row["viewer_id"]
        inst.consumer_costs[i, j] = row["financial_cost"]
        inst.energy_costs[i, j] = row["energy_cost"]
        inst.consumer_latencies[i, j] = row["latency"]
    return inst


def solve_and_store(db_path, priority):
    alpha, beta = policy_to_parameters(priority)
    inst = load_instance_from_db(db_path)
    inst.exact_solve(alpha=alpha, beta=beta)
    status = inst.solve_status_message or "optimal"
    metrics = None if inst.exact_obj is None else inst.get_exact_solution_breakdown()
    open_nodes = [] if inst.open_centres is None else inst.open_centres
    assignments = {} if inst.assignments is None else inst.assignments
    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO optimal_solutions
            (solved_at, priority, alpha, beta, objective_value, status, financial_cost, energy_cost, avg_latency, open_nodes, assignments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now_iso(), priority, alpha, beta, inst.exact_obj, status,
             None if metrics is None else metrics["total_financial_cost"],
             None if metrics is None else metrics["total_energy_score"],
             None if metrics is None else metrics["avg_latency"],
             json.dumps(open_nodes), json.dumps(assignments)),
        )
        conn.commit()
        return cur.lastrowid
