#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 09:56:32 2025

@author: localadmin
"""
import numpy as np
def random_consumers(n_consumers, n_hotspots, min_var = 0.01, max_var = 0.1):
    '''
    This function samples from a density over [0,1]^2 .
    
    This is just a mixture of 2D-Gaussians - with the mean of each Gaussian
    corresponding to a hotspot, and the variance denoting how far out the 
    hotspot will reach.
    
    For simplicity I have not added in any covariance structure so the 
    hotspots will all be circular. The variance is from min to max var
    '''
    
    #generate means, vars, and weights for the Gaussian Mixture
    weights = np.random.uniform(0, 1, n_hotspots)
    weights /= weights.sum() #normalise
    
    x_means = np.random.uniform( 0.1, 0.9, n_hotspots)
    y_means = np.random.uniform(0.1, 0.9, n_hotspots)
    means = np.column_stack((x_means, y_means))
    
    var_vals = np.random.uniform(min_var, max_var, n_hotspots)
    
    consumer_locations = np.zeros((n_consumers,2))
    
    #which hotspot each consumer is sampled from
    which_hotspot = np.random.choice(np.arange(0, n_hotspots), n_consumers, 
                                     True, weights)
    
    for i in range(n_consumers):
        cov = np.diag(np.full(2, var_vals[which_hotspot[i]]))
        consumer_locations[i,:] = np.random.multivariate_normal(means[which_hotspot[i]],
                                                                cov)
    
    return consumer_locations
    
    

    
    
    