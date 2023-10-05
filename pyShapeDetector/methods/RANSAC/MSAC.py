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

class MSAC(RANSAC_WeightedBase):
    
    _type = "MSAC"
        
    def weight_distance(self, distances):
        threshold = self.reduction_rate * self.threshold_distance
        
        weight = np.zeros(len(distances))
        idx = distances > threshold
        weight[idx] = 1 - (distances[idx] / threshold) ** 2
        
        return weight
        
    def weight_angle(self, angles):
        threshold = self.reduction_rate * self.threshold_angle
        
        weight = np.zeros(len(angles))
        idx = angles > threshold
        weight[idx] = 1 - (angles[idx] / threshold) ** 2
        
        return weight
    
    
