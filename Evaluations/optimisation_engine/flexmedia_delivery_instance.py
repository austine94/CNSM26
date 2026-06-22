#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 10:23:09 2025

@author: localadmin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import gurobipy as gp
from gurobipy import GRB

from random_datacentres import random_datacentres
from random_consumers import random_consumers
from random_costs import random_datacentre_costs, random_consumer_costs, random_datacentre_capacities, random_consumer_latencies, random_energy_costs

class flexmedia_delivery_instance:
    
    '''
    This is a class to store information about a single instance of the 
    flexmedia delivery optimisation problem
    
    We have n_datacentres and n_consumers and the consumers are randomly
    drawn from a mixture of Bivariate Gaussians - all this is over [0,1]^2.
    
    Energy costs are randomly sampled from between energy_min and energy_max
    
    The min_radius is the min gap between data centres - this is ensured using
    rejection sampling.
    
    The min and max var are the ranges for the min and max variances of the
    bivariate Gaussian hotspot centres - note the variance is circular,
    there is no correlation parameter.
    '''
    
    def __init__(self, n_datacentres, n_consumers, n_hotspots, 
                 min_radius = 0.01, min_var = 0.01, max_var = 0.1,
                 edge_min = 10, edge_max = 100, core_min = 1, core_max = 10,
                 consumer_min = 0.01, consumer_max = 1, 
                 energy_min = 0.001, energy_max = 0.1,
                 edge_capacity = 10, core_capacity = 100):
        self.datacentre_locations = random_datacentres(n_datacentres,
                                                       min_radius)
        self.consumer_locations = random_consumers(n_consumers, n_hotspots,
                                                   min_var, max_var)
        self.datacentre_costs = random_datacentre_costs(self.datacentre_locations,
                                                        edge_min, edge_max, 
                                                        core_min, core_max)
        self.consumer_costs = random_consumer_costs(n_consumers, n_datacentres,
                                                    consumer_min, consumer_max)
        self.energy_costs = random_energy_costs(n_consumers, n_datacentres,
                                                energy_min, energy_max)
        self.datacentre_capacities = random_datacentre_capacities(self.datacentre_locations, 
                                                                  edge_capacity,
                                                                  core_capacity)
        self.consumer_latencies = random_consumer_latencies(self.datacentre_locations,
                                                            self.consumer_locations,
                                                            1.0)
        
        self.optimisation_model = None  #these attributes are for the opt results
        self.optimisation_model_type = None
        self.assignments = None
        self.open_centres = None
        self.exact_obj = None
        self.lagrangian_feasible_obj= None
        self.solve_status = None
        self.solve_status_message = None
    
    
    def plot_instance(self):
        '''
        This method plots the optimisation instance - so we see a unit square
        with blue squares for the DCs and red dots for the consumers.
        '''
        
        fig, ax = plt.subplots()
        
        # plot data centres as large blue squares, consumers as red dots

        ax.scatter(self.consumer_locations[:, 0], self.consumer_locations[:, 1],
                    marker='o',      # circle
                    s=20,            
                    c='red',         
                    label='Consumers')
        
        ax.scatter(self.datacentre_locations[:, 0],
                    self.datacentre_locations[:, 1],
                    marker='s',      # square
                    s=120,           
                    c='blue',        
                    label='Data Centres')
        
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
                
        ax.legend(loc='upper left',
                  bbox_to_anchor=(1.02, 1))      
        
        plt.tight_layout()  # make room for the legend
        plt.show()            
        
    def plot_density(self):
        '''
        This method plots the density of the consumers, which can be viewed as
        a 2D density plot for a mixture of bivariate Gaussians.
        
        '''
        kde = gaussian_kde(self.consumer_locations.T)

        # Create evaluation grid
        xmin, ymin = 0, 0
        xmax, ymax = 1, 1
        x_lin = np.linspace(xmin, xmax, 200)
        y_lin = np.linspace(ymin, ymax, 200)
        
        X, Y = np.meshgrid(x_lin, y_lin, indexing='xy')
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions), X.shape)
        
        # Plot density map with sample overlay
        plt.figure()
        plt.imshow(Z, origin='lower', aspect='auto',
                   extent=[xmin, xmax, ymin, ymax])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('KDE-based Consumer Density Estimate')
        plt.show()
        
        
    def exact_solve(self, alpha, beta):
        '''
        This method interfaces with Gurobi and solves the instance using MIP
        methods (as implemented by Gurobi, so various cutting planes basically).
        
        Alpha is the parameter for weighting the QoE / latency penalty.
        beta is the parameter weighting between cost and energy usage
        
        The function will print information from the Gurobi solver, however
        the class is also updated with a new attribute - optimisation_model -
        which contains all of the information returned from the solver.
        
        '''
        #######
        #Read in data from self
        #######

        m = len(self.datacentre_costs)    # number of data‐centres
        n = len(self.consumer_costs[0])   # number of consumers

        D = self.datacentre_costs           #costs to open
        B = self.datacentre_capacities #centre capacities
        c = beta*self.consumer_costs + (1-beta)*self.energy_costs  #overall weighted delivery costs
        d = self.consumer_latencies #latencies/distances

        #######
        #Set Variables
        #######

        #flexmedia_model = gp.Model("flexmedia_datacentre_placement")
        #flexmedia_model.setParam('OutputFlag', 0) #prevents console printing
        
        
        #this code creates a new env and silences output so we don't get json
        #errors from the agent when first running Gurobi
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        flexmedia_model = gp.Model("flexmedia_datacentre_placement", env=env)

        #Decision variables
        #y[i] = 1 if we open centre i
        y = flexmedia_model.addVars(m, vtype=GRB.BINARY, name="y")

        #x[i,j] = 1 if consumer j is served by centre i
        x = flexmedia_model.addVars(m, n, vtype=GRB.BINARY, name="x")

        #######
        #Objective Function
        ########

        #quicksum just gives the sum of the arrays as sum() does, but is faster in gurobi
        flexmedia_model.setObjective(
            gp.quicksum(D[i] * y[i] 
                        for i in range(m))
          + gp.quicksum((c[i,j] + alpha * d[i,j]) * x[i,j]
                        for i in range(m) for j in range(n)),
            GRB.MINIMIZE
        )

        ########
        #Add constraints
        #######

        #consumer j must be assigned exactly once:
        flexmedia_model.addConstrs(
            (gp.quicksum(x[i,j] for i in range(m)) == 1
             for j in range(n)),
            name="assign_once"
        )

        #only assign consumers to an open centre:
        flexmedia_model.addConstrs(
            (x[i,j] <= y[i]
             for i in range(m) for j in range(n)),
            name="open_if_assigned"
        )

        #centre i serves at most B[i] consumers if opened:
        flexmedia_model.addConstrs(
            (gp.quicksum(x[i,j] for j in range(n)) <= B[i] * y[i]
             for i in range(m)),
            name="capacity"
        )

        #########
        #Solve
        #########

        flexmedia_model.optimize()
        
        status = flexmedia_model.status
        self.solve_status = status
        self.solve_status_message = None
        
        if status == GRB.OPTIMAL:
            #if the optimal solution has been found
            obj = flexmedia_model.objVal   
        
            self.optimisation_model = flexmedia_model
            self.optimisation_model_type = 'Exact'
            self.exact_obj = obj
        
            #Extract assignments and open DCs
            self.assignments = {
                i: [j for j in range(n) if x[i, j].X > 0.5]
                for i in range(m)
                if y[i].X > 0.5
            }
            self.open_centres = [i for i in range(m) if y[i].X > 0.5]
        
        elif status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
            #in this instance the model cannot solve the instance.
            self.solve_status_message = f"Model is infeasible or unbounded (status = {status})."
            self.optimisation_model = flexmedia_model
            self.optimisation_model_type = 'Exact'
            self.exact_obj = None
            self.assignments = None
            self.open_centres = None
        
        else:
            #if the model has not returned the optimal sol, but also not 
            #proven infeasibility.
            self.solve_status_message = f"Optimization ended with status {status}."
            if flexmedia_model.SolCount > 0:
                # There *is* a feasible solution, just not proven optimal.
                obj = flexmedia_model.objVal
                self.solve_status_message += f" Best found objective = {obj}"
                self.optimisation_model = flexmedia_model
                self.optimisation_model_type = 'Exact'
                self.exact_obj = obj
            else:
                self.exact_obj = None
                self.assignments = None
                self.open_centres = None
        

    def print_exact_solution(self):
        '''
        This method prints out the exact solution should it exist, showing which
        consumers go from each DC. Will be of limited to no use for large
        instances!
        '''
        if self.optimisation_model is None:
            print("Optimisation istance has not been solved exactly yet")
            return
        
        for i in self.open_centres:
            print(f"Data centre {i} opened")
            for j in self.assignments[i]:
                print(f"  - Viewer {j} assigned")
                
    def plot_exact_solution(self):
        '''
        This method plots the exact solution, with each DC a different coloured
        square, and each consumer a dot coloured according to the DC it is 
        assigned to.
        
        To help with plot clarity, this function only shows the opened DCs.
        
        Again, likely to be of limited to no use for large instances !
        '''
        
        if self.optimisation_model is None:
            print("Optimisation istance has not been solved exactly yet")
            return
        
        m = len(self.datacentre_costs)    # number of data‐centres
        n = len(self.consumer_costs[0])   # number of consumers
        
        consumer_to_centre = np.full(n, -1, dtype=int)
        for i, consumers in self.assignments.items():
            for j in consumers:
                consumer_to_centre[j] = i
        
        cmap = plt.get_cmap('tab10', m)
        dc_colors = cmap(np.arange(m))
        
        plt.figure(figsize=(6, 6))
        for i in self.open_centres:
            x, y = self.datacentre_locations[i]
            plt.scatter(x, y, marker='s', s=200, color=dc_colors[i], edgecolor='k', label=f'Data Centre {i}')
        
        for j in range(n):
            i = consumer_to_centre[j]
            x, y = self.consumer_locations[j]
            plt.scatter(x, y, marker='o', s=50, color=dc_colors[i])
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.show()
        
    def get_exact_solution_breakdown(self):
        """
        Calculates the metrics breakdown for the current solution
        Returns a dictionary:
        {
            'total_financial_cost': float,
            'total_energy_cost': float,
            'avg_latency': float,
            'consumers_served' : int
        }
        """
        if self.assignments is None:
            return None

        total_financial = 0.0
        total_energy = 0.0
        total_latency = 0.0
        
        #Fixed Costs for Open Data Centres
        # Sum D[i] for all open i
        fixed_cost_total = sum(self.datacentre_costs[i] for i in self.open_centres)
        total_financial += fixed_cost_total
        
        #Variable financial cost is the cost of delivering to consumers
        #assignments is dict {dc_index: [consumer_indices]}
        count_consumers = 0
        variable_financial = 0.0
        
        for dc_idx, consumers in self.assignments.items():
            for c_idx in consumers:
                #financial: Fixed + delivery cost
                variable_financial += self.consumer_costs[dc_idx, c_idx]
                
                #energy
                total_energy += self.energy_costs[dc_idx, c_idx]
                
                #QoE
                total_latency += self.consumer_latencies[dc_idx, c_idx]
                
                count_consumers += 1
                
        total_financial += variable_financial
        
        # Avoid division by zero
        avg_lat = (total_latency / count_consumers) if count_consumers > 0 else 0.0
        
        #un-scale the final data
        f_scale = getattr(self, 'financial_scaler', 1.0)
        e_scale = getattr(self, 'energy_scaler', 1.0)
        l_scale = getattr(self, 'latency_scaler', 1.0)
        
        return {
            "total_financial_cost": round(total_financial * f_scale, 2),
            "total_energy_score": round(total_energy * e_scale, 2),
            "avg_latency": round(avg_lat * l_scale, 4),
            "consumers_served": count_consumers
        }

        
        
        
        