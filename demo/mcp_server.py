import json
from pathlib import Path

from fastmcp import FastMCP

from config import DB_PATH
from database import connect, initialise_database, set_policy
from optimisation import solve_and_store, policy_to_parameters
from visualise import render_latest_solution

mcp = FastMCP("Agentic Compute Provisioning Demo")

@mcp.tool()
def get_system_status() -> str:
    """Check the ingestion summary table and report whether data is available."""
    initialise_database(DB_PATH)
    with connect(DB_PATH) as conn:
        row = conn.execute("SELECT * FROM ingestion_summary WHERE id=1").fetchone()
    return json.dumps(dict(row), indent=2)

@mcp.tool()
def get_problem_status() -> str:
    """Summarise the number of viewers needing service and compute nodes available."""
    with connect(DB_PATH) as conn:
        viewers = conn.execute("SELECT COUNT(*) AS c FROM viewers").fetchone()["c"]
        nodes = conn.execute("SELECT COUNT(*) AS c FROM compute_nodes").fetchone()["c"]
        capacity = conn.execute("SELECT COALESCE(SUM(capacity), 0) AS c FROM compute_nodes").fetchone()["c"]
    return f"Current problem: {viewers} viewers, {nodes} compute nodes, total capacity {capacity}."

@mcp.tool()
def read_policy_database() -> str:
    """Read the single active operational priority set by the media provider."""
    initialise_database(DB_PATH)
    with connect(DB_PATH) as conn:
        row = conn.execute("SELECT priority, updated_at FROM policies WHERE id=1").fetchone()
    alpha, beta = policy_to_parameters(row["priority"])
    return f"Active priority: {row['priority']} (updated {row['updated_at']}); selected alpha={alpha}, beta={beta}."

@mcp.tool()
def update_policy_database(priority: str) -> str:
    """Update the active operational priority while the demo is running."""
    set_policy(priority, DB_PATH)
    return f"Policy updated to: {priority}"

@mcp.tool()
def solve_optimisation_problem() -> str:
    """Read viewer/node data, select objective weights from policy, solve, and store solution."""
    with connect(DB_PATH) as conn:
        priority = conn.execute("SELECT priority FROM policies WHERE id=1").fetchone()["priority"]
    solution_id = solve_and_store(DB_PATH, priority)
    return f"Optimisation complete. Stored solution_id={solution_id}."

@mcp.tool()
def read_optimal_solution() -> str:
    """Read and summarise the latest optimal provisioning and allocation."""
    with connect(DB_PATH) as conn:
        row = conn.execute("SELECT * FROM optimal_solutions ORDER BY solution_id DESC LIMIT 1").fetchone()
    if row is None:
        return "No optimal solution stored yet."
    return json.dumps(dict(row), indent=2)

@mcp.tool()
def render_optimal_solution() -> str:
    """Render a Markdown summary and optional PNG plot of the latest solution."""
    return render_latest_solution(DB_PATH)

if __name__ == "__main__":
    mcp.run(transport="stdio")
