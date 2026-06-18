import json
from pathlib import Path

from config import OUTPUT_DIR
from database import connect


def render_latest_solution(db_path, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        sol = conn.execute("SELECT * FROM optimal_solutions ORDER BY solution_id DESC LIMIT 1").fetchone()
        viewers = conn.execute("SELECT * FROM viewers ORDER BY viewer_id").fetchall()
        nodes = conn.execute("SELECT * FROM compute_nodes ORDER BY node_id").fetchall()
    if sol is None:
        return "No solution available."
    assignments = {int(k): v for k, v in json.loads(sol["assignments"]).items()}
    open_nodes = set(json.loads(sol["open_nodes"]))
    lines = [
        "# Latest Agentic Compute Provisioning Solution",
        "",
        f"Solved at: {sol['solved_at']}",
        f"Priority: {sol['priority']}",
        f"Weights: alpha={sol['alpha']}, beta={sol['beta']}",
        f"Objective: {sol['objective_value']}",
        f"Financial cost: {sol['financial_cost']}",
        f"Energy score: {sol['energy_cost']}",
        f"Average latency: {sol['avg_latency']}",
        f"Open nodes: {sorted(open_nodes)}",
        "",
        "## Allocation summary",
    ]
    for node in nodes:
        assigned = assignments.get(node["node_id"], [])
        status = "OPEN" if node["node_id"] in open_nodes else "closed"
        lines.append(f"- Node {node['node_id']} ({status}, capacity {node['capacity']}): {len(assigned)} viewers")
    path = output_dir / "latest_solution.md"
    path.write_text("\n".join(lines) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        for node in nodes:
            marker = "s" if node["node_id"] in open_nodes else "x"
            size = 160 if node["node_id"] in open_nodes else 80
            plt.scatter(node["longitude"], node["latitude"], marker=marker, s=size, label=f"Node {node['node_id']}")
            plt.text(node["longitude"], node["latitude"], f"N{node['node_id']}")
        for viewer in viewers:
            plt.scatter(viewer["longitude"], viewer["latitude"], marker="o", s=25, c="tab:red", alpha=0.65)
        plt.title(f"Optimised allocation: {sol['priority']}")
        plt.xlim(0, 1); plt.ylim(0, 1); plt.xlabel("longitude"); plt.ylabel("latitude")
        plt.tight_layout()
        png = output_dir / "latest_solution.png"
        plt.savefig(png, dpi=150)
        plt.close()
        return f"Wrote {path} and {png}"
    except Exception as exc:
        return f"Wrote {path}; plot skipped: {exc}"
