import asyncio
import json
import os
import re
import subprocess
import time
from typing import List, Dict, Any

# WE USE OLLAMA DIRECTLY - NO KEYS NEEDED
import ollama 
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from parameter_selection_accuracy import SYSTEM_PROMPT, TOOL_DEFINITION

# --- CONFIGURATION ---
N_REPS = 250   #number of sim reps
MODEL = "interstellarninja/llama3.1-8b-tools:latest"  # Ensure you have pulled this model: `ollama pull interstellarninja/llama3.1-8b-tools:latest`
EXPERIMENT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_data")
RESULTS_FILE = os.path.join(EXPERIMENT_DATA_DIR, "illustrative_experiment_results.json")

TEST_PROMPTS = {
    "focus_cost": "Optimise this using all datacentres. My absolute priority is minimising cost, regardless of latency or energy usage.",
    "focus_energy": "Optimise this using all datacentres. Minimise energy usage above all else.",
    "focus_latency": "Optimise this using all datacentres. We need the lowest possible latency and best user experience.",
    "tradeoff_cost_energy": "Optimise this using all datacentres. Balance financial cost and energy usage; latency is not the main priority.",
    "tradeoff_cost_latency": "Optimise this using all datacentres. Keep financial cost controlled, but maintain acceptable user experience.",
    "tradeoff_energy_latency": "Optimise this using all datacentres. Prefer a greener allocation minimising energy, but do not let user experience become poor.",
}

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
server_script = os.path.join(project_root, 'server', 'mcp_server.py')
generator_script = os.path.join(project_root, 'scripts', 'generate_scenario.py')

def generate_fresh_problem():
    """Runs the generator script to overwrite data/current_problem.json"""
    subprocess.run(["python", generator_script], check=True, capture_output=True)

def parse_metrics_from_text(text_output: str) -> Dict[str, float]:
    """
    Regex magic to pull numbers back out of the Agent's text response.
    Expected format: 
     Financial Cost: $1200.50
     Energy Usage:    45.20 units
     Avg Latency:     0.3500 ms
    """
    metrics = {}
    
    # Regex patterns to find numbers after specific keywords
    cost_match = re.search(r"Financial Cost:\s*\$?([\d\.]+)", text_output)
    energy_match = re.search(r"Energy Usage:\s*([\d\.]+)", text_output)
    lat_match = re.search(r"Latency.*:\s*([\d\.]+)", text_output)
    
    if cost_match: metrics['cost'] = float(cost_match.group(1))
    if energy_match: metrics['energy'] = float(energy_match.group(1))
    if lat_match: metrics['latency'] = float(lat_match.group(1))
    
    return metrics

async def run_benchmark():
    print(f"🚀 Starting Benchmark: {N_REPS} reps x {len(TEST_PROMPTS)} prompts")
    print(f"🤖 Model: {MODEL} (Local)")

    # Start MCP Server
    server_params = StdioServerParameters(
        command="python", 
        args=[server_script],
        env=os.environ.copy()
    )

    results_db = {k: [] for k in TEST_PROMPTS.keys()}

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # --- MAIN LOOP (INVERTED) ---
            # Loop 1: Iterate through SEEDS (0 to N_REPS)
            for i in range(N_REPS):
                current_seed = i
                print(f"\n📢 --- REP {i+1}/{N_REPS} (World Seed {current_seed}) ---")
                
                # A. Generate the World ONCE for this seed
                # We pass the seed as a command line argument
                subprocess.run(
                    ["python", generator_script, str(current_seed)], 
                    check=True, capture_output=True
                )
                
                # Loop 2: Run ALL strategies on this identical world
                for category, prompt in TEST_PROMPTS.items():
                    print(f"  > Strat: {category:<15}...", end="", flush=True)
                    
                    # B. Ask LLM
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                    
                    try:
                        response = ollama.chat(
                            model=MODEL,
                            messages=messages,
                            tools=TOOL_DEFINITION,
                            options={
                                'seed': i,
                                'temperature': 0.7 
                            }
                        )
                        
                        msg = response['message']
                        tool_calls = msg.get('tool_calls', [])

                        if not tool_calls:
                            print(" ❌ Failed (No Tool)")
                            continue
                            
                        # C. Execute Tool
                        tool_call = tool_calls[0]
                        fn_name = tool_call['function']['name']
                        fn_args = tool_call['function']['arguments']
                        
                        if fn_name == "solve_allocation":
                            result = await session.call_tool(fn_name, fn_args)
                            output_text = result.content[0].text
                            parsed = parse_metrics_from_text(output_text)
                            
                            record = {
                                "rep": i,
                                "seed": current_seed, # Track the seed used
                                "prompt_category": category,
                                "tool_args": fn_args,
                                "metrics": parsed
                            }
                            results_db[category].append(record)
                            print(" ✅")
                        else:
                            print(f" ⚠️ Skipped ({fn_name})")

                    except Exception as e:
                        print(f" ❌ Error: {e}")

    # 4. Save to JSON
    os.makedirs(EXPERIMENT_DATA_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_db, f, indent=2)
    
    print(f"\n✅ Benchmark Complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())