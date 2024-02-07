#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:59:55 2023

@author: ebernardes
"""

from abc import ABC, abstractmethod
import time
import random
import numpy as np

# random.seed(951)

from .RANSAC_Base import RANSAC_Base

class RANSAC_Classic(RANSAC_Base):
    """
    Implementation of RANSAC Classic method.
    
    Attributes
    ----------
    _type
    num_samples
    max_weight
    primitives
    limits
    num_primitives
    options
    
    Methods
    -------
    __init__
    __repr__
    add
    remove
    remove_all  
    copy_options
    weight_distances
    weight_angles
    get_total_weight
    compare_metrics
    termination_criterion
    get_metrics
    get_model
    get_samples
    get_biggest_connected_component
    get_inliers
    fit
    """
    
    _type = "RANSAC Classic"
    
    def weight_distances(self, distances, distance_threshold):
        """ Gives weights for each point based on the distances.

        For `RANSAC_Weighted`, the weight function is a simple
        step that changes at threshold.
        
        Parameters
        ----------
        distances : array
            Distances of each point to shape
        threshold_distances : float
            Max distance accepted between points and shape
        
        Returns
        -------
        array
            Weights of each point
        """
        weight = np.zeros(len(distances))
        weight[distances < distance_threshold] = 1
        return weight
    
    def compare_metrics(self, metrics, metrics_best):
        """ Compare metrics to decide if new fit is better than current
        best fit.

        For RANSAC Classic, choose the shape with best fitness, meaning
        the higher number of inliers.
        
        Parameters
        ----------
        metrics : dict
            Metrics analyzing current fit
        metrics_best : dict
            Metrics analyzing current best fit
        
        Returns
        -------
        bool
            True if `metrics` is considered better than `metrics_best`
        """
        return (metrics['fitness'] > metrics_best['fitness'] or 
                (metrics['fitness'] == metrics_best['fitness'] and 
                 metrics['rmse_distances'] < metrics_best['rmse_distances']))
    
