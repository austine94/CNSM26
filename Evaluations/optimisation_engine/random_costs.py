#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 08:38:58 2025

@author: localadmin
"""

import numpy as np
import math
from scipy.spatial.distance import cdist

def l2_dist(a, b):
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def random_datacentre_costs(datacentre_locations, edge_min, edge_max, core_min,
                     core_max):
    '''
    This function generates random costs for datacentres depending on if they
    are core or edge. A core DC is defined as one within 0.2 distance of the
    centre, noting that all DCs occur in [0,1]^2.
    
    The function returns an array of length m, where m is the number of DCs,
    with each entry the cost of opening that centre.
    '''
    
    n_datacentres = len(datacentre_locations)
    centre = [0.5, 0.5] #for working out distance of dc to centre
    
    datacentre_costs = np.zeros(n_datacentres)
    
    for i in range(n_datacentres):
        dc_dist = l2_dist(datacentre_locations[i], centre)
        if dc_dist <= 0.2:  #assign costs depending on if core or edge
            datacentre_costs[i] = np.random.uniform(core_min, core_max)
        else:
            datacentre_costs[i] = np.random.uniform(edge_min, edge_max)   
    return datacentre_costs

def random_consumer_costs(n_consumers, n_datacentres, 
                          consumer_min, consumer_max):
    '''
    This function randomly sampled the cost of serving each of the n consumers
    from each of the m datacentres.
    
    The data is returned such that we have a length m array, one entry per 
    DC, and each of the n entries is a length n array wth each 
    entry being the cost of serving each consumer from that datacentre.
    '''
    
    return np.random.uniform(consumer_min, consumer_max, 
                             size=(n_datacentres, n_consumers))

def random_energy_costs(n_consumers, n_datacentres, 
                          energy_min, energy_max):
    '''
    This function randomly sampled the enegy cost of serving each of the n
    consumers from each of the m datacentres.
    
    The data is returned such that we have a length m array, one entry per 
    DC, and each of the n entries is a length n array wth each 
    entry being the cost of serving each consumer from that datacentre.
    '''
    
    return np.random.uniform(energy_min, energy_max, 
                             size=(n_datacentres, n_consumers))

def random_datacentre_capacities(datacentre_locations, edge_capacity,
                                 core_capacity):
    '''
    This function generates random capacities for each data centre.
    It uses different capacities depending on if a data centre is edge or
    core, with the core defined as one within 0.2 of the centre, noting that
    all DCs occur in [0,1]^2.
    
    The capacities are randomly selected in the interval
    0.9*capacity, 1.1*capacity.
    
    The function returns an array of m integer capacities. 
    '''
    
    edge_lower = 0.9*edge_capacity
    edge_upper = 1.1*edge_capacity
    core_lower = 0.9*core_capacity
    core_upper = 1.1*core_capacity
    
    
    n_datacentres = len(datacentre_locations)
    centre = [0.5, 0.5] #for working out distance of dc to centre
    
    datacentre_capacities = np.zeros(n_datacentres)
    
    for i in range(n_datacentres):
        dc_dist = l2_dist(datacentre_locations[i], centre)
        if dc_dist <= 0.2:
            datacentre_capacities[i] = np.random.uniform(core_lower, core_upper)
        else:
            datacentre_capacities[i] = np.random.uniform(edge_lower, edge_upper)
            
    return datacentre_capacities

def random_consumer_latencies(datacentre_locations, consumer_locations,
                              weight):
    '''
    This function works out a proxy latency based on a function of the 
    l2 distance between the consumer and datacentre locations.
    
    Returns a length m array, with each entry a length n array, where
    m is the number of DCs and n is the number of consumers, and entry
    [m,n] is latency of consumer n to DC m.
    '''
    latencies = cdist(datacentre_locations, consumer_locations,
                  metric='euclidean')
    #We use cdist to vectorise the operation
    return latencies
    