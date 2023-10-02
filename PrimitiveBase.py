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
    
    @staticmethod
    @abstractmethod
    def get_mesh(model, pcd):
        pass
    
    @staticmethod
    @abstractmethod
    def get_distances(points, model):
        pass
    
    @staticmethod
    @abstractmethod
    def get_normal_angles(points, normals, model):
        pass
    
    @staticmethod
    @abstractmethod
    def get_model(points, samples):
        pass
        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    