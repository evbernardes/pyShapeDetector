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

from .ShapeDetector import ShapeDetector
    
class CylinderDetector(ShapeDetector):
    
    _ransac_n_min = 3
    _model_args_n = 4
    
    def __init__(self, 
                distance_threshold=0.01, 
                ransac_n=3, 
                num_iterations=100, 
                probability=0.99999999):
        
        super().__init__(distance_threshold, 
                         ransac_n, 
                         num_iterations, 
                         probability)
    
    @staticmethod
    def get_distances(points, model):
        return np.abs(points.dot(model[:3]) + model[3])
    
    @staticmethod
    def get_model(points, samples):
        points_ = np.asarray(points)[samples]
        
        # if simplest case, the result is direct
        if len(samples) == 3:
            p0, p1, p2 = points_
            
            e0 = p1 - p0
            e1 = p2 - p0
            abc = np.cross(e0, e1)
            centroid = p0
        
        # for more points, find the plane such that the summed squared distance 
        # from the plane to all points is minimized.
        # Reference: https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        else:
            centroid = sum(points_) / len(samples)
            x, y, z = np.asarray(points_ - centroid).T
            xx = x.dot(x)
            yy = y.dot(y)
            zz = z.dot(z)
            xy = x.dot(y)
            yz = y.dot(z)
            xz = z.dot(x)
                
            det_x = yy * zz - yz * yz
            det_y = xx * zz - xz * xz
            det_z = xx * yy - xy * xy
            
            if (det_x > det_y and det_x > det_z):
                abc = np.array([det_x, 
                                xz * yz - xy * zz, 
                                xy * yz - xz * yy])
            elif (det_y > det_z):
                abc = np.array([xz * yz - xy * zz, 
                                det_y, 
                                xy * xz - yz * xx])
            else:
                abc = np.array([xy * yz - xz * yy, 
                                xy * xz - yz * xx, 
                                det_z])
        
        norm = np.linalg.norm(abc)
        if norm == 0.0:
            return np.array([0, 0, 0, 0])
        
        abc /= norm
        return np.array([abc[0], abc[1], abc[2], -abc.dot(centroid)]) 
        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    