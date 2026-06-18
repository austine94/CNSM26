import argparse
import asyncio
import json
import subprocess
import urllib.request
from pathlib import Path

from config import DEFAULT_INTERVAL_SECONDS
from data_generator import generate_snapshot
from database import initialise_database, set_policy

WORKFLOW = [
    {"tool": "get_system_status", "arguments": {}},
    {"tool": "get_problem_status", "arguments": {}},
    {"tool": "read_policy_database", "arguments": {}},
    {"tool": "solve_optimisation_problem", "arguments": {}},
    {"tool": "read_optimal_solution", "arguments": {}},
    {"tool": "render_optimal_solution", "arguments": {}},
]


def ask_ollama_for_next_action(model, context, step_index):
    prompt = (
        "You are an SLM controller for compute provisioning. Return only JSON with keys tool and arguments. "
        "Follow the deterministic workflow unless the context proves the step is invalid. "
        f"Workflow: {json.dumps(WORKFLOW)}\nContext: {context}\nNext step index: {step_index}"
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = json.loads(response.read().decode())
    return json.loads(payload["message"]["content"])


def validated_action(candidate, step_index):
    expected = WORKFLOW[step_index]
    if not isinstance(candidate, dict):
        return expected, "candidate was not a JSON object"
    if candidate.get("tool") != expected["tool"]:
        return expected, f"candidate tool {candidate.get('tool')} replaced with deterministic {expected['tool']}"
    args = candidate.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    return {"tool": expected["tool"], "arguments": args}, "validated"


async def call_mcp_tool(client, name, arguments):
    result = await client.call_tool(name, arguments)
    if hasattr(result, "content"):
        return "\n".join(getattr(item, "text", str(item)) for item in result.content)
    return str(result)


async def run_agent_cycle(model, retries):
    try:
        from fastmcp import Client
    except ImportError as exc:
        raise RuntimeError("fastmcp is required. Install demo/requirements.txt before running the MCP agent.") from exc

    context = []
    server_path = Path(__file__).resolve().parent / "mcp_server.py"
    async with Client(str(server_path)) as client:
        optimisation_called = False
        for step_index in range(len(WORKFLOW)):
            action = None
            validation = ""
            for attempt in range(retries):
                try:
                    candidate = ask_ollama_for_next_action(model, context, step_index)
                except Exception as exc:
                    candidate = WORKFLOW[step_index]
                    validation = f"Ollama unavailable or invalid on attempt {attempt + 1}: {exc}; used deterministic fallback"
                action, validation = validated_action(candidate, step_index)
                if action["tool"] == WORKFLOW[step_index]["tool"]:
                    break
            output = await call_mcp_tool(client, action["tool"], action.get("arguments", {}))
            if action["tool"] == "solve_optimisation_problem":
                optimisation_called = True
            context.append({"action": action, "validation": validation, "output": output})
            print(f"\nTOOL {action['tool']}\nVALIDATION {validation}\n{output}")
        if not optimisation_called:
            print("FAILURE_FLAG: no optimisation tool was called within the retry budget.")
    return context


async def main_async(args):
    initialise_database()
    if args.policy:
        set_policy(args.policy)
    cycle = 0
    while True:
        cycle += 1
        print(f"\n=== DEMO CYCLE {cycle}: generating workload and ingesting database ===")
        print(generate_snapshot(args.viewers, args.nodes, missing_rate=args.missing_rate))
        await run_agent_cycle(args.model, args.retries)
        if not args.loop:
            break
        await asyncio.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description="Run the standalone Agentic-AI provisioning demo with Ollama + MCP.")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name.")
    parser.add_argument("--policy", default=None, help="Optional priority to write before starting, e.g. 'minimise energy usage'.")
    parser.add_argument("--loop", action="store_true", help="Repeat with a changing workload.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--viewers", type=int, default=24)
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
