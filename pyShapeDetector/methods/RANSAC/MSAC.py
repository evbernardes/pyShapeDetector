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

random.seed(951)

from .RANSAC_WeightedBase import RANSAC_WeightedBase

class MSAC(RANSAC_WeightedBase):
    """
    MSAC weighted RANSAC.
    
    Methods
    -------
    compare_metrics(metrics, metrics_best):
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.

    termination_criterion(metrics):
        Gives number of max needed iterations, depending on current metrics.
        
    get_metrics(num_points=None, num_inliers=None, distances=None, 
    angles=None):
        Gives a dictionary with metrics that can be used to compared fits.
        
    get_model(points, normals, samples):
        Fits shape, then test if its model parameters respect input
        max and min values. If it does, return shape, otherwise, return None.
        
    get_samples(points, num_samples, tries_max=5000):
        Sample points and return indices of sampled points.

    get_inliers_from_residuals(distances, angles):
        Return indices of inliers: points whose distance to shape and
        angle with normal vector are below the given thresholds.
        
    fit(points, normals=None, debug=False):#, filter_model=True):
        Main loop implementing RANSAC algorithm.
               
    get_residuals(points, normals):
        Convenience function returning both distances and angles.

    weight_distances(distances, threshold_distances):
        Gives weights for each point based on the distances.

    weight_angles(angles, threshold_angles):
        Gives weights for each point based on the distances.

    get_total_weight(distances, angles):
        Total weight of shape.
    """
    
    _type = "MSAC"
        
    def weight_distances(self, distances, distance_threshold):
        """ Gives weights for each point based on the distances.
        
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
        threshold = self.reduction_rate * distance_threshold
        
        weight = np.zeros(len(distances))
        idx = distances > threshold
        weight[idx] = 1 - (distances[idx] / threshold) ** 2
        
        return weight
        
    
        
    
    
