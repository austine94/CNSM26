#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:21:22 2025

@author: localadmin
"""
import math
import numpy as np

def l2_dist(a, b):
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def random_datacentres(n, min_radius = 0.01):
    '''
    This function generates n random data centres located in [0,1]^2.
    
    We assume that no data-centre is contained within n_radius of another.
    
    The function returns an array of tuples (x,y)
    
    '''
    
    data_centre_locations = np.zeros((n,2))
    
    data_centre_locations[0,:] = np.random.uniform(0, 1, 2)
    filled = 1
    
    while filled < n:
        #generate new location and check if it is outside min radii
        new_location = np.random.uniform(0,1,2)
        
        proposed_location_distances = np.zeros(filled)
        for i in range(filled):
            proposed_location_distances[i] = l2_dist(new_location, data_centre_locations[i,:])
            
        if all(proposed_location_distances > min_radius):
            data_centre_locations[filled, :] = new_location
            filled += 1
    
    return data_centre_locations
    