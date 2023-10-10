#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:57:08 2023

@author: ebernardes
"""
from abc import ABC, abstractmethod
import random
import copy
import numpy as np
import open3d as o3d
from open3d.geometry import TriangleMesh, PointCloud
from open3d.utility import Vector3iVector, Vector3dVector

from .PrimitiveBase import PrimitiveBase
    
class Template(PrimitiveBase):
    
    _fit_n_min = 0
    _model_args_n = 0
    name = 'template' 
    
    def get_distances(self, points):
        points = np.asarray(points)
        return np.sum(points, axis=1)
    
    def get_normals(self, points):
        return points
    
    def get_mesh(self, points):
        return TriangleMesh()
    
    @staticmethod
    def fit(points, normals=None):
        # points_ = np.asarray(points)[samples]
        
        num_points = len(points)
        
        if num_points < 0:
            raise ValueError('A minimun of 0 points are needed to fit a '
                             'template')
        
        return Template([]) 
