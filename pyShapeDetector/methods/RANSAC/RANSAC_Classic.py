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
    
    def compare_metrics(self, metrics, metrics_best):
        return (metrics['fitness'] > metrics_best['fitness'] or 
                (metrics['fitness'] == metrics_best['fitness'] and 
                 metrics['rmse_distances'] < metrics_best['rmse_distances']))
    
