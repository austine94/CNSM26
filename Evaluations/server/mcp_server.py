# server/mcp_server.py
from fastmcp import FastMCP
import json
import numpy as np
import os
import sys
import contextlib
from typing import Any

#SETUP PATHS 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
engine_path = os.path.join(project_root, 'optimisation_engine')
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

from flexmedia_delivery_instance import flexmedia_delivery_instance

#START SERVER
mcp = FastMCP("FlexMedia Optimizer")

#DEFINE DATA PATH
DATA_PATH = os.path.join(project_root, "data", "current_problem.json")

#STORE LAST SOLVED INSTANCE
LAST_SOLVED_INSTANCE = None

DEFAULT_MAX_DATACENTRES = 9999


def normalise_max_datacentres(value: Any) -> int:
    """Return a safe datacentre limit, defaulting invalid tool inputs to 9999."""
    if value is None or isinstance(value, bool):
        return DEFAULT_MAX_DATACENTRES

    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return DEFAULT_MAX_DATACENTRES
            parsed = float(value)
        else:
            parsed = float(value)

        if not np.isfinite(parsed) or parsed < 0:
            return DEFAULT_MAX_DATACENTRES

        return int(parsed)
    except (TypeError, ValueError):
        return DEFAULT_MAX_DATACENTRES

def load_problem_into_instance():
    """
    Function to load in JSON of problem instance and then write into 
    correct class structure.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No problem found at {DATA_PATH}. Run scripts/generate_scenario.py first.")

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    #Instantiate the class (ignore random data)
    inst = flexmedia_delivery_instance(
        n_datacentres=data['n_datacentres'], 
        n_consumers=data['n_consumers'], 
        n_hotspots=1 
    )

    #overwrite with data from JSON
    raw_dc_costs = np.array(data['datacentre_costs'])
    raw_consumer_costs = np.array(data['consumer_costs'])    
    inst.datacentre_capacities = np.array(data['datacentre_capacities'])
    raw_consumer_latencies = np.array(data['consumer_latencies'])
    inst.datacentre_locations = np.array(data['datacentre_locations'])
    inst.consumer_locations = np.array(data['consumer_locations'])
    raw_energy_costs = np.array(data['energy_costs'])
    
    #normalise costs to [0,1]
    max_financial = max(np.max(raw_dc_costs), np.max(raw_consumer_costs))
    if max_financial > 0:
        inst.datacentre_costs = raw_dc_costs / max_financial
        inst.consumer_costs = raw_consumer_costs / max_financial
    else:
        inst.datacentre_costs = raw_dc_costs
        inst.consumer_costs = raw_consumer_costs

    #normalise energy
    max_energy = np.max(raw_energy_costs)
    if max_energy > 0:
        inst.energy_costs = raw_energy_costs / max_energy
    else:
        inst.energy_costs = raw_energy_costs

    #normalise latency
    max_latency = np.max(raw_consumer_latencies)
    if max_latency > 0:
        inst.consumer_latencies = raw_consumer_latencies / max_latency
    else:
        inst.consumer_latencies = raw_consumer_latencies
        
    #save scalers so we can re-interpret later
    inst.financial_scaler = max_financial
    inst.energy_scaler = max_energy
    inst.latency_scaler = max_latency
    
    return inst


def capacity_diagnostic(inst) -> str:
    """Return a concise capacity diagnostic for infeasible assignment models."""
    n_consumers = len(inst.consumer_locations)
    usable_capacity = int(np.floor(inst.datacentre_capacities).sum())
    raw_capacity = float(np.sum(inst.datacentre_capacities))
    if usable_capacity < n_consumers:
        return (
            f" Total usable datacentre capacity is {usable_capacity} "
            f"(raw capacity {raw_capacity:.2f}) for {n_consumers} viewers; "
            "regenerate the scenario with more/larger datacentres."
        )
    return ""

@contextlib.contextmanager
def suppress_stdout():
    """Silences Gurobi prints"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@mcp.tool()
def get_problem_status() -> str:
    """
    Returns the number of viewers and available datacentres.
    """
    if not os.path.exists(DATA_PATH):
        return "No problem data found."
    
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        
    return f"Current Scenario: {data['n_consumers']} Viewers, {data['n_datacentres']} Data Centres available."

@mcp.tool()
def solve_allocation(
    alpha: float,
    beta: float,
    max_datacentres: Any = DEFAULT_MAX_DATACENTRES,
) -> str:
    """
    Solves the facility location problem exactly.
    
    Args:
        alpha: Float to penalise latency in the objective.
            Use 0.0 to ignore latency, 0.1 for moderate latency trade-offs,
            and 1.0 for latency-priority optimisation.
        beta: Float between zero and one to balance financial cost and energy.
            Use 1.0 for financial-cost priority, 0.0 for energy priority,
            and 0.5 for cost-energy trade-offs.
        max_datacentres: Maximum number of facilities to open. Invalid or missing
            values default to 9999.

    Canonical presets:
        cost only: alpha=0.0, beta=1.0
        energy only: alpha=0.0, beta=0.0
        latency only: alpha=1.0, beta=0.5
        cost-energy trade-off: alpha=0.0, beta=0.5
        cost-latency trade-off: alpha=0.1, beta=1.0
        energy-latency trade-off: alpha=0.1, beta=0.0
    """
    global LAST_SOLVED_INSTANCE
    
    try:
        max_datacentres = normalise_max_datacentres(max_datacentres)
        inst = load_problem_into_instance()
        
        #solve and save. Suppress stdout because MCP stdio must only emit JSON-RPC.
        with suppress_stdout():
            inst.exact_solve(alpha = alpha, beta = beta)
        LAST_SOLVED_INSTANCE = inst
        
        if inst.exact_obj is None:
            status_message = inst.solve_status_message or "Solver failed or found no feasible solution."
            return f"{status_message}{capacity_diagnostic(inst)}"

        #get metrics for solution
        metrics = inst.get_exact_solution_breakdown()
        
        num_open = len(inst.open_centres)
        warning = ""
        if num_open > max_datacentres:
            warning = f"\nWARNING: Solution used {num_open} DCs, exceeding limit of {max_datacentres}."

        return (
            f"Optimization Successful.\n"
            f"-----------------------\n"
            f"Strategy Inputs: Alpha={alpha}, Beta={beta}\n"
            f"Objective Func:  {inst.exact_obj:.2f}\n"
            f"-----------------------\n"
            f"Metrics:\n"
            f"Financial Cost: ${metrics['total_financial_cost']}\n"
            f"Energy Usage:    {metrics['total_energy_score']} units\n"
            f"Avg Latency:     {metrics['avg_latency']} ms\n"
            f"DCs Opened:      {num_open}\n"
            f"{warning}"
        )

    except Exception as e:
        return f"Error running optimization: {str(e)}"
    
@mcp.tool()
def analyse_alpha_tradeoffs(max_datacentres: Any, alphas: list | str, beta: float):
    """
    Runs a simulation across multiple alpha values 
    to compare cost and latency trade-offs.
    
    Args:
        max_datacentres: The limit on DCs. Invalid or missing values default to 9999.
        alpha_vec: A list of alpha values to tru
    """
    try:
        max_datacentres = normalise_max_datacentres(max_datacentres)
        #when the model sends a string that shoulkd be a list, e.g. "[0.1, 0.2]", 
        #parse it into a real list
        if isinstance(alphas, str):
            try:
                alphas = json.loads(alphas)
            except:
                # Fallback cleanup for messy strings (like "0.1, 0.2")
                cleaned = alphas.replace('[', '').replace(']', '')
                alphas = [float(x) for x in cleaned.split(',') if x.strip()]

        #ensure it is now a list
        if not isinstance(alphas, list):
             return f"Error: Input 'alphas' must be a list, got {type(alphas)}."
         
        #read in problem
        inst = load_problem_into_instance()
        
        
        #create report
        report = f"TRADEOFF ANALYSIS (Constraint: Max {max_datacentres} DCs)\n"
        report += f"{'Alpha':<6} | {'Objective':<10} | {'DCs Used':<8} | {'Status'}\n"
        report += "-" * 45 + "\n"
        
        #loop over potential alphas
        with suppress_stdout():   #silence Guribi
            for alpha in alphas:
                val = float(alpha)
                inst.exact_solve(alpha=val, beta=beta)
                
                if inst.exact_obj is None:
                    report += f"{val:<6.2f} | {'Failed':<9} | {'-':<8} | {'-':<8} | {'-':<6} | -\n"
                    continue
                
                #metrics for this value of alpha
                m = inst.get_exact_solution_breakdown()
                n_open = len(inst.open_centres)
                
                #flag violations with an asterisk
                warn = "*" if n_open > max_datacentres else " "
                
                report += (f"{val:<6.2f} | "
                           f"{inst.exact_obj:<9.2f} | "
                           f"{m['total_financial_cost']:<8.0f} | "
                           f"{m['total_energy_score']:<8.0f} | "
                           f"{m['avg_latency']:<6.3f} | "
                           f"{n_open}{warn}\n")
            
        return report
    except Exception as e:
        return f"Error running analysis: {str(e)}"    
    
@mcp.tool()
def analyse_beta_tradeoffs(max_datacentres: Any, betas: list | str, alpha: float = 1):
    """
    Runs a simulation across multiple beta values to compare
    financial cost vs energy usage.
    
    Args:
        max_datacentres: Hard limit on DCs. Invalid or missing values default to 9999.
        betas: List of beta values to test (e.g. [0.0, 0.2, 0.5, 0.8, 1.0]).
               0.0 = energy priority
               0.5 = cost-energy trade-off
               1.0 = financial-cost priority
        alpha: fixed latency penalty
    """
    try:
        max_datacentres = normalise_max_datacentres(max_datacentres)
        #as for alpha robust parsing for strings of json
        if isinstance(betas, str):
            try:
                betas = json.loads(betas)
            except:
                cleaned = betas.replace('[', '').replace(']', '')
                betas = [float(x) for x in cleaned.split(',') if x.strip()]

        if not isinstance(betas, list):
             return "Error: Input 'betas' must be a list."

        inst = load_problem_into_instance()
        
        #report header
        report = f"BETA TRADEOFF ANALYSIS (Alpha={alpha}, Max {max_datacentres} DCs)\n"
        report += f"{'Beta':<6} | {'Obj':<9} | {'Cost ($)':<9} | {'Energy':<8} | {'Lat':<6} | {'DCs'}\n"
        report += "-" * 65 + "\n"
        
        with suppress_stdout():  #silence Gurobi
            for b in betas:
                val = float(b)
                
                #solve with iterating Beta, fixed Alpha
                inst.exact_solve(alpha=alpha, beta=val)
                
                if inst.exact_obj is None:
                    report += f"{val:<6.2f} | {'Failed':<9} | {'-':<9} | {'-':<8} | {'-':<6} | -\n"
                    continue
                
                #metrics
                m = inst.get_exact_solution_breakdown()
                n_open = len(inst.open_centres)
                
                #violations
                warn = "*" if n_open > max_datacentres else " "
                
                report += (f"{val:<6.2f} | "
                           f"{inst.exact_obj:<9.2f} | "
                           f"{m['total_financial_cost']:<9.0f} | "
                           f"{m['total_energy_score']:<8.0f} | "
                           f"{m['avg_latency']:<6.3f} | "
                           f"{n_open}{warn}\n")
            
        return report

    except Exception as e:
        return f"Error running analysis: {str(e)}"

@mcp.tool()
def get_solution_metrics():
    """
    Retrieves the detailed metrics (Cost, Energy, Latency) for the 
    LAST solved solution stored in memory.
    """
    global LAST_SOLVED_INSTANCE  #reads in the global instance of last solved
    
    if LAST_SOLVED_INSTANCE is None:
        return "No solution in memory. You must run 'solve_allocation' first."
    
    if LAST_SOLVED_INSTANCE.exact_obj is None:
        return "The last run failed to find a solution."

    try:
        # Calculate metrics using the method you just wrote
        m = LAST_SOLVED_INSTANCE.get_exact_solution_breakdown()
        n_open = len(LAST_SOLVED_INSTANCE.open_centres)
            
        return (
            f"--- SOLUTION BREAKDOWN ---\n"
            f"  Financial Cost: ${m['total_financial_cost']}\n"
            f"  Energy Usage:    {m['total_energy_score']} units\n"
            f"  Avg Latency:     {m['avg_latency']} ms\n"
            f"  DCs Opened:      {n_open}\n"
            f"  Consumers:       {m['consumers_served']}\n"
            f"--------------------------"
        )
    except Exception as e:
        return f"Error retrieving metrics: {str(e)}"


    
if __name__ == "__main__":
    print("Starting FlexMedia Optimizer MCP server on stdio...", file=sys.stderr)
    mcp.run(transport="stdio")

    
    
