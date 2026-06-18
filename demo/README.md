# Standalone Agentic-AI Compute Provisioning Demo

This folder is a standalone demonstration of the proposed three-layer Agentic-AI system for compute provisioning. It is intentionally separate from the paper experiments, but it adapts their design patterns and optimisation objective.

## Architecture

The demo has three layers:

1. **Data ingestion layer**
   - `data_generator.py` generates a new workload using the repository's scenario-generation routines.
   - Raw viewer and compute-node data are written to `raw_data/viewers.csv` and `raw_data/compute_nodes.csv`.
   - The screening pipeline imputes missing costs and capacities with the average observed value, normalises costs onto `[0, 1]`, and loads the cleaned data into SQLite.

2. **MCP server layer**
   - `mcp_server.py` exposes the same experiment-style optimisation tools, plus the explicit tools described in the Agentic-AI system text:
     - `get_system_status`
     - `get_problem_status`
     - `solve_optimisation_problem`
     - `read_optimal_solution`
     - `read_policy_database`
     - `update_policy_database`
     - `render_optimal_solution`
   - The optimisation call uses the current repository's `flexmedia_delivery_instance.exact_solve` objective.

3. **SLM agent layer**
   - `agent.py` uses Ollama for structured JSON action selection.
   - The workflow is deterministic and validated to reduce unstable tool calls and infinite ReAct-style loops.
   - If the selected action is malformed or unexpected, the agent replaces it with the expected next workflow step and records that validation event.
   - If no optimisation tool is called within the retry budget, the agent emits a failure flag.

## Quick start

Run the demo from the repository root. The dashboard and the workload/agent loop are two separate processes, so use separate terminal tabs/windows.

### Terminal 1: install dependencies and start Ollama

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
pip install -r demo/requirements.txt
python -c "import gurobipy as gp; print(gp.gurobi.version())"
ollama serve
```

Keep this terminal running. In a second terminal, make sure the same virtual environment is active and pull the model once if you have not already done so:

```bash
source .venv-demo/bin/activate
ollama pull llama3.2
```

### Terminal 2: start the live dashboard

```bash
source .venv-demo/bin/activate
python demo/frontend.py
```

This starts a small local web server on your own machine. "Open <http://127.0.0.1:8765>" means open a browser, such as Chrome/Firefox/Safari, paste `http://127.0.0.1:8765` into the address bar, and press Enter. You should see the Agentic Compute Provisioning dashboard.

### Terminal 3: run the workload and MCP/Ollama agent loop

For a single workload/optimisation cycle, run:

```bash
source .venv-demo/bin/activate
python demo/run_demo.py --model llama3.2 --policy "minimise financial cost"
```

For the intended live demo, run a changing workload every 30 seconds:

```bash
source .venv-demo/bin/activate
python demo/run_demo.py --loop --interval 30 --model llama3.2
```

With the loop running, leave the dashboard browser tab open. Every 30 seconds the demo generates a fresh workload, writes the raw CSVs, ingests them into SQLite, asks the Ollama-backed MCP agent to solve the allocation, stores the solution, and refreshes the dashboard view.

## Changing the operational priority during runtime

The active priority is stored as a single policy row in SQLite and mirrored in `policy.txt`. You can update it by using the MCP tool `update_policy_database`, or directly before launch:

```bash
python demo/run_demo.py --policy "minimise energy usage"
python demo/run_demo.py --policy "minimise QoE latency"
python demo/run_demo.py --policy "balanced cost energy and latency trade-off"
```

The demo maps the natural-language policy to the experiment objective parameters:

| Priority wording | alpha | beta | Meaning |
| --- | ---: | ---: | --- |
| financial / cost | 0.0 | 1.0 | minimise financial cost |
| energy | 0.0 | 0.0 | minimise energy-weighted delivery cost |
| latency / QoE / quality | 1.0 | 0.5 | prioritise latency/QoE |
| balanced / trade-off | 0.1 | 0.5 | mixed cost-energy-latency objective |

## Browser dashboard

`frontend.py` provides a lightweight browser dashboard without adding another web framework dependency. It reads directly from the SQLite database and refreshes every five seconds. The dashboard shows ingestion status, current policy, latest optimisation metrics, open compute nodes, the rendered allocation plot, and the raw latest solution JSON. You can also update the active operational priority from the dashboard while the demo is running; the next agent cycle will read it through the MCP policy tool.

```bash
python demo/frontend.py --host 127.0.0.1 --port 8765
```

## Outputs

Each cycle produces:

- fresh raw CSVs in `demo/raw_data/`;
- cleaned SQLite tables in `demo/agentic_demo.sqlite`;
- a new row in `optimal_solutions`;
- a human-readable report in `demo/outputs/latest_solution.md`;
- a visual allocation plot in `demo/outputs/latest_solution.png` if Matplotlib is available.

## Gurobi requirement

This demo uses the repository's Gurobi-backed optimisation engine rather than a replacement solver. The demo requirements pin `gurobipy==12.0.2` to match the Gurobi Optimizer 12.0.2 installation used by the paper experiments. If you previously created `.venv-demo` before this pin was added, reinstall the demo dependencies or run `pip install --force-reinstall gurobipy==12.0.2` inside `.venv-demo`. The version check in the quick start should print `(12, 0, 2)`.

## Notes for reviewers

- SQLite is used instead of Postgres to keep the demo portable.
- The MCP server is real and runs over stdio through FastMCP.
- Ollama is used for the SLM layer, but the workflow remains deterministic and validated so that reviewers can reproduce behaviour.
- The optimisation is intentionally delegated to the existing repository optimisation engine, so the demo follows the same mathematical objective as the experiments.
