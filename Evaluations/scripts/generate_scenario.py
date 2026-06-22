#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import numpy as np
import sys

#import class
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
engine_path = os.path.join(project_root, 'optimisation_engine')
if engine_path not in sys.path:
    sys.path.append(engine_path)                          # Add to python path

from flexmedia_delivery_instance import flexmedia_delivery_instance

#config
n_datacentres = 5
n_consumers = 100
n_hotspots = 5
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'current_problem.json')


def ensure_feasible_capacity(inst, n_consumers):
    """Increase generated capacities if needed so every consumer can be assigned.

    Gurobi assigns whole consumers to data centres.  Because generated
    capacities are floating-point values, the usable capacity of a centre is
    its floor; a scenario is feasible only when the sum of those floors covers
    all consumers.
    """
    usable_capacity = int(np.floor(inst.datacentre_capacities).sum())
    if usable_capacity >= n_consumers:
        return

    shortfall = n_consumers - usable_capacity
    n_datacentres = len(inst.datacentre_capacities)
    base_increase = shortfall // n_datacentres
    remainder = shortfall % n_datacentres

    inst.datacentre_capacities = np.floor(inst.datacentre_capacities).astype(float)
    inst.datacentre_capacities += base_increase
    inst.datacentre_capacities[:remainder] += 1

if __name__ == "__main__":
    
    #parser for optional seed argument
    parser = argparse.ArgumentParser(description="Generate FlexMedia Scenario")
    parser.add_argument("seed", type=int, nargs='?', default=None, help="Random seed (optional)")
    args = parser.parse_args()

    #seed can be none, in which case we set it randomly
    if args.seed is not None:
        #for benchmarking
        seed_val = args.seed
    else:
        seed_val = np.random.randint(0, 1000000)

    np.random.seed(seed_val)

    #create instance
    inst = flexmedia_delivery_instance(n_datacentres, n_consumers, n_hotspots)
    ensure_feasible_capacity(inst, n_consumers)
    
    # 4. Save to JSON
    data = {
        'n_datacentres': n_datacentres,
        'n_consumers': n_consumers,
        'datacentre_locations': inst.datacentre_locations.tolist(),
        'consumer_locations': inst.consumer_locations.tolist(),
        'datacentre_costs': inst.datacentre_costs.tolist(),
        'datacentre_capacities': inst.datacentre_capacities.tolist(),
        'consumer_costs': inst.consumer_costs.tolist(),
        'consumer_latencies': inst.consumer_latencies.tolist(),
        'energy_costs': inst.energy_costs.tolist() 
    }

    #check if path exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f)
        
    print(f"Scenario saved to {output_path}")