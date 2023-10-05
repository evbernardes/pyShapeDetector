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
    def weight_distance(self, distances):
        pass

    @abstractmethod
    def weight_angle(self, angles):
        pass
    
    def get_total_weight(self, distances, angles):
        weight_distance = sum(self.weight_distance(distances))
        weight_angle = sum(self.weight_angle(angles))
        return weight_distance * weight_angle
    
    def compare_info(self, info, info_best):
        return info['weight'] > info_best['weight']
    
    def get_info(self, 
                 num_points=None, num_inliers=None, 
                 distances=None, angles=None):
        info = super().get_info(num_points, num_inliers, distances, angles)
        if num_points is None or num_inliers is None or distances is None:
            info['weight'] = 0
        else:
            info['weight'] = self.get_total_weight(distances, angles)
        return info
