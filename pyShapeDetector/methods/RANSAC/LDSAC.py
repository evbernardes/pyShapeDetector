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
# import open3d as o3d
# import multiprocessing
# from multiprocessing import Process, Manager

random.seed(951)

from .RANSAC_WeightedBase import RANSAC_WeightedBase

class LDSAC(RANSAC_WeightedBase):
    
    _type = "LDSAC"
    
    def __init__(self,
                 primitive,
                 reduction_rate=1.0,
                 threshold_distance=0.1,
                 threshold_angle=10,
                 ransac_n=None,
                 num_iterations=100,
                 probability=0.99999,
                 max_point_distance=None,
                 model_max=None,
                 model_min=None,
                 max_normal_angle_degrees=10,
                 inliers_min=None,
                 threshold_ratios=[0.2, 0.7]):
        
        if len(threshold_ratios) != 2 or \
            not (0 <= threshold_ratios[0] <= 1) or \
                not (0 <= threshold_ratios[1] <= 1):
            
            raise ValueError('threshold_ratios must be a tuple of two '
                             'values between 0 and 1, got {threshold_ratios}')
            
        if threshold_ratios[1] < threshold_ratios[0]:
            threshold_ratios = threshold_ratios[::-1]
        
        RANSAC_WeightedBase.__init__(self, primitive, reduction_rate,
            threshold_distance, threshold_angle, ransac_n, num_iterations,
            probability, max_point_distance, model_max, model_min, 
            max_normal_angle_degrees, inliers_min)
        
        self.threshold_ratios = threshold_ratios
        
    def weight_distances(self, distances, distance_threshold):
        d1 = distance_threshold * self.threshold_ratios[0]
        d2 = distance_threshold * self.threshold_ratios[1]
        
        weight = np.zeros(len(distances))
        weight[np.abs(distances) < d1] = 1
        idx = (np.abs(distances) > d1) * (np.abs(distances) < d2)
        weight[idx] = (d2 - distances[idx]) / (d2 - d1)
        
        return weight
    
    
