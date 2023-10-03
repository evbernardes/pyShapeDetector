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
    
    def __init__(self, model):
        if len(model) < self._model_args_n:
            raise ValueError(f'{self._name.capitalize()} primitives need take '
                             f'{self._model_args_n} elements, got {model}')
        self.model = model

    @abstractmethod
    def get_normal_angles_cos(self, points, normals):
        pass
    
    @staticmethod
    def get_distances(self, points):
        pass
    
    @staticmethod
    def get_mesh(self, pcd):
        pass
    
    @staticmethod
    @abstractmethod
    def create_from_points(points):
        pass

        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    