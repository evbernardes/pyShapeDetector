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

from .RANSAC_Base import RANSAC_Base

class RANSAC_WeightedBase(RANSAC_Base):
    
    @abstractmethod
    def weight_distances(self, distances, threshold_distances):
        pass

    def weight_angles(self, angles, threshold_angles):
        return self.weight_distances(angles, threshold_angles)
    
    def get_total_weight(self, distances, angles):
        if distances is None:
            weight_distance = 1
        else:
            weight_distance = sum(
                self.weight_distances(distances, self.threshold_distance))
            
        if angles is None:
            weight_angle = 1
        else:
            weight_angle = sum(
                self.weight_angles(angles, self.threshold_angle))
            
        return weight_distance * weight_angle
    
    def compare_metrics(self, metrics, metrics_best):
        return metrics['weight'] > metrics_best['weight']
    
    def get_metrics(self, 
                 num_points=None, num_inliers=None, 
                 distances=None, angles=None):
        metrics = super().get_metrics(num_points, num_inliers, distances, angles)
        if num_points is None or num_inliers is None or distances is None:
            metrics['weight'] = 0
        else:
            metrics['weight'] = self.get_total_weight(distances, angles)
        return metrics
