import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from config import DB_PATH, OUTPUT_DIR
from database import connect, initialise_database, set_policy
from visualise import render_latest_solution

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agentic Compute Provisioning Demo</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f8fb; color: #172033; }
    header { background: linear-gradient(135deg, #0f3d5e, #216b91); color: white; padding: 1.5rem 2rem; }
    main {
      padding: 1.5rem 2rem;
      display: grid;
      grid-template-columns: minmax(320px, 0.95fr) minmax(480px, 1.35fr);
      grid-template-areas:
        "priority plot"
        "terminal plot"
        "solution nodes"
        "status status";
      gap: 1rem;
      align-items: start;
    }
    section { background: white; border: 1px solid #dce4ef; border-radius: 14px; padding: 1rem; box-shadow: 0 8px 24px rgb(23 32 51 / 8%); }
    h1, h2 { margin: 0 0 .75rem; }
    .metric-grid { display: grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap: .75rem; }
    .metric { background: #f1f5fa; border-radius: 10px; padding: .8rem; }
    .metric strong { display: block; font-size: 1.4rem; color: #0f3d5e; }
    table { border-collapse: collapse; width: 100%; }
    th, td { text-align: left; border-bottom: 1px solid #e5ebf2; padding: .45rem; font-size: .92rem; }
    input, button { font: inherit; padding: .65rem; border-radius: 8px; border: 1px solid #b9c7d6; }
    button { background: #216b91; color: white; border: 0; cursor: pointer; }
    img { width: 100%; max-height: 640px; object-fit: contain; border: 1px solid #dce4ef; border-radius: 12px; background: white; }
    pre { white-space: pre-wrap; background: #0d1726; color: #dbeafe; border-radius: 10px; padding: .8rem; overflow: auto; max-height: 360px; }
    .priority-panel { grid-area: priority; }
    .terminal-panel { grid-area: terminal; }
    .plot-panel { grid-area: plot; }
    .solution-panel { grid-area: solution; }
    .nodes-panel { grid-area: nodes; }
    .status-panel { grid-area: status; }
    .muted { color: #607089; font-size: .9rem; }
    @media (max-width: 900px) {
      main {
        grid-template-columns: 1fr;
        grid-template-areas:
          "priority"
          "plot"
          "terminal"
          "solution"
          "nodes"
          "status";
      }
    }
  </style>
</head>
<body>
<header>
  <h1>Agentic Compute Provisioning Demo</h1>
  <div class="muted" style="color:#d7ecff">Live SQLite-backed dashboard for ingestion, policy, optimisation, and visual solution output.</div>
</header>
<main>
  <section class="priority-panel">
    <h2>Operational priority</h2>
    <form id="policy-form">
      <input id="priority" name="priority" style="width:70%" placeholder="minimise financial cost">
      <button type="submit">Update policy</button>
    </form>
    <p class="muted">The running agent reads this policy through the MCP policy tool on each cycle.</p>
  </section>
  <section class="plot-panel">
    <h2>Allocation plot</h2>
    <img id="plot" alt="Latest allocation plot will appear here once a solution is rendered.">
  </section>
  <section class="terminal-panel">
    <h2>Terminal readings</h2>
    <pre id="raw"></pre>
  </section>
  <section class="solution-panel">
    <h2>Latest solution</h2>
    <div id="solution" class="metric-grid"></div>
  </section>
  <section class="nodes-panel">
    <h2>Open compute nodes</h2>
    <table><thead><tr><th>Node</th><th>Capacity</th><th>Assigned viewers</th><th>Status</th></tr></thead><tbody id="nodes"></tbody></table>
  </section>
  <section class="status-panel">
    <h2>System status</h2>
    <div id="status" class="metric-grid"></div>
  </section>
</main>
<script>
function metric(label, value) { return `<div class="metric"><span>${label}</span><strong>${value ?? '—'}</strong></div>`; }
async function refresh() {
  const response = await fetch('/api/state');
  const state = await response.json();
  document.getElementById('status').innerHTML = [
    metric('Viewers', state.ingestion?.viewers_processed),
    metric('Compute nodes', state.ingestion?.nodes_processed),
    metric('Imputed values', state.ingestion?.imputed_values),
    metric('Last ingest', state.ingestion?.last_viewer_ingest || '—')
  ].join('');
  document.getElementById('priority').value = state.policy?.priority || '';
  const sol = state.solution || {};
  document.getElementById('solution').innerHTML = [
    metric('Objective', sol.objective_value),
    metric('Financial cost', sol.financial_cost),
    metric('Energy score', sol.energy_cost),
    metric('Avg latency', sol.avg_latency)
  ].join('');
  const assignments = sol.assignments || {};
  const open = new Set(sol.open_nodes || []);
  document.getElementById('nodes').innerHTML = (state.nodes || []).map(node => {
    const assigned = assignments[node.node_id] || [];
    return `<tr><td>${node.node_id}</td><td>${node.capacity}</td><td>${assigned.length}</td><td>${open.has(node.node_id) ? 'OPEN' : 'closed'}</td></tr>`;
  }).join('');
  document.getElementById('raw').textContent = JSON.stringify(state, null, 2);
  document.getElementById('plot').src = '/latest_solution.png?cache=' + Date.now();
}
document.getElementById('policy-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  await fetch('/api/policy', { method: 'POST', body: new URLSearchParams(new FormData(event.target)) });
  await refresh();
});
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


def _json_response(handler, payload, status=200):
    body = json.dumps(payload, indent=2).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def latest_state():
    initialise_database(DB_PATH)
    render_latest_solution(DB_PATH)
    with connect(DB_PATH) as conn:
        ingestion = conn.execute("SELECT * FROM ingestion_summary WHERE id=1").fetchone()
        policy = conn.execute("SELECT * FROM policies WHERE id=1").fetchone()
        solution = conn.execute("SELECT * FROM optimal_solutions ORDER BY solution_id DESC LIMIT 1").fetchone()
        nodes = conn.execute("SELECT * FROM compute_nodes ORDER BY node_id").fetchall()
    solution_dict = dict(solution) if solution else None
    if solution_dict:
        solution_dict["open_nodes"] = json.loads(solution_dict["open_nodes"] or "[]")
        solution_dict["assignments"] = {int(k): v for k, v in json.loads(solution_dict["assignments"] or "{}").items()}
    return {
        "ingestion": dict(ingestion) if ingestion else None,
        "policy": dict(policy) if policy else None,
        "solution": solution_dict,
        "nodes": [dict(node) for node in nodes],
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = DASHBOARD_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/state"):
            _json_response(self, latest_state())
            return
        if self.path.startswith("/latest_solution.png"):
            plot_path = OUTPUT_DIR / "latest_solution.png"
            if not plot_path.exists():
                self.send_response(404)
                self.end_headers()
                return
            body = plot_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/policy":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode()
        priority = parse_qs(body).get("priority", [""])[0].strip()
        if not priority:
            _json_response(self, {"error": "priority is required"}, status=400)
            return
        set_policy(priority, DB_PATH)
        _json_response(self, {"priority": priority})

    def log_message(self, format, *args):
        return


def run_dashboard(host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard available at http://{host}:{port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Serve the standalone demo dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_dashboard(args.host, args.port)

if __name__ == "__main__":
    main()
