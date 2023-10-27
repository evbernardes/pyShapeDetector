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

class LDSAC(RANSAC_WeightedBase):
    """
    LDSAC weighted RANSAC.
    
    Attributes
    ----------
    _type : str
        Name of method.
    
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
        
    get_samples(points):
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
    
    _type = "LDSAC"
    
    def __init__(self,
                 primitives,
                 reduction_rate=1.0,
                 threshold_distance=0.1,
                 threshold_angle=3.141592653589793,  # ~ 180 degrees
                 threshold_refit_ratio=3,
                 ransac_n=None,
                 num_iterations=100,
                 probability=0.99999,
                 max_point_distance=None,
                 limits=None,
                 max_normal_angle_degrees=10,
                 inliers_min=None,
                 fitness_min=None,
                 connected_components_density=None,
                 threshold_ratios=[0.2, 0.7]):
        
        if len(threshold_ratios) != 2 or \
            not (0 <= threshold_ratios[0] <= 1) or \
                not (0 <= threshold_ratios[1] <= 1):
            
            raise ValueError('threshold_ratios must be a tuple of two '
                             'values between 0 and 1, got {threshold_ratios}')
            
        if threshold_ratios[1] < threshold_ratios[0]:
            threshold_ratios = threshold_ratios[::-1]
        
        RANSAC_WeightedBase.__init__(
            self,
            primitives=primitives,
            reduction_rate=reduction_rate,
            threshold_distance=threshold_distance,
            threshold_angle=threshold_angle, 
            threshold_refit_ratio=threshold_refit_ratio,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            probability=probability,
            max_point_distance=max_point_distance,
            limits=limits,
            max_normal_angle_degrees=max_normal_angle_degrees,
            inliers_min=inliers_min,
            fitness_min=fitness_min,
            connected_components_density=connected_components_density)
        
        self.threshold_ratios = threshold_ratios
        
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
        d1 = distance_threshold * self.threshold_ratios[0]
        d2 = distance_threshold * self.threshold_ratios[1]
        
        weight = np.zeros(len(distances))
        weight[np.abs(distances) < d1] = 1
        idx = (np.abs(distances) > d1) * (np.abs(distances) < d2)
        weight[idx] = (d2 - distances[idx]) / (d2 - d1)
        
        return weight
    
    
