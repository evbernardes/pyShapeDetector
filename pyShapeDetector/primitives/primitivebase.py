#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
from abc import ABC, abstractmethod
import random
import copy
import numpy as np
import open3d as o3d
    
class PrimitiveBase(ABC):
    
    @property
    @abstractmethod
    def _fit_n_min(self):
        pass

    @property
    @abstractmethod
    def _model_args_n(self):
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass

    def __init__(self, model):
        if len(model) < self._model_args_n:
            raise ValueError(f'{self.name.capitalize()} primitives take '
                             f'{self._model_args_n} elements, got {model}')
        self.model = model
    
    def get_angles_cos(self, points, normals):
        if normals is None:
            return None
        normals = np.asarray(normals)
        normals_from_points = self.get_normals(points)
        angles_cos = np.clip(
            np.sum(normals * normals_from_points, axis=1), -1, 1)
        return np.abs(angles_cos)
    
    def get_angles(self, points, normals):
        if normals is None:
            return None
        
        return np.arccos(
            self.get_angles_cos(points, normals))
    
    def get_residuals(self, points, normals):
        return self.get_distances(points), \
            self.get_angles(points, normals)
            
    @staticmethod
    def create_limits(args_n, idx, value):
        values = [None] * args_n
        values[idx] = value
        return values
    
    @staticmethod
    def get_normal_eig(normals):
        C = np.zeros([3, 3])
        for n in np.asarray(normals):
            C += n[..., np.newaxis] * n
        return np.linalg.eig(C)
    
    @staticmethod
    def get_distances(self, points):
        pass
    
    @staticmethod
    def get_normals(self, points):
        pass
    
    @staticmethod
    def get_mesh(self, pcd):
        pass
    
    @staticmethod
    @abstractmethod
    def fit(points, normals=None):
        pass

        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    