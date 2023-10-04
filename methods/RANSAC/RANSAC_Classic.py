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

class RANSAC_Classic(RANSAC_Base):
    
    _type = "RANSAC Classic"
        
    def weight_distance(self, distances):
        weight = np.zeros(len(distances))
        weight[distances > self.threshold_distance] = 1
        return weight
        
    def weight_angle(self, angles):
        weight = np.zeros(len(angles))
        weight[angles > self.threshold_angular] = 1
        return weight
    
