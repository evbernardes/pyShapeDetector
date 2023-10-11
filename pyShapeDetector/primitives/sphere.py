#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
import numpy as np
from open3d.geometry import TriangleMesh

from .primitivebase import PrimitiveBase
    
class Sphere(PrimitiveBase):
    
    _fit_n_min = 4
    _model_args_n = 4
    name = 'sphere'
    
    @property
    def radius(self):
        return self.model[-1]
    
    @property
    def center(self):
        return np.array(self.model[:3])
    
    @staticmethod
    def limit_radius(value):
        return PrimitiveBase.create_limits(Sphere._model_args_n, 3, value)
    
    def get_distances(self, points):
        points = np.asarray(points)
        model = self.model
        distances = np.linalg.norm(points - model[:3], axis=1) - model[3]
        return np.abs(distances)
    
    def get_normals(self, points):
        points = np.asarray(points)
        dist_vec = points - self.model[:3]
        normals = dist_vec / np.linalg.norm(dist_vec, axis=1)[..., np.newaxis]
        return normals

    def get_mesh(self, points):
        mesh = TriangleMesh.create_sphere(radius=self.model[3])
        mesh.translate(self.model[:3])
        return mesh
   
    @staticmethod
    def fit(points, normals=None):
        # points_ = np.asarray(points)[samples]
        
        num_points = len(points)
        
        if num_points < 4:
            raise ValueError('A minimun of 4 points are needed to fit a '
                             'sphere')
        
        # if simplest case, the result is direct
        elif num_points == 4:
            p0, p1, p2, p3 = points
            
            r1 = p0 - p1
            r2 = p0 - p2
            r3 = p0 - p3
            n0 = p0.dot(p0)
            c12 = np.cross(r1, r2)
            c23 = np.cross(r2, r3)
            c31 = np.cross(r3, r1)
            
            det = r1.dot(c23)
            if det == 0:
                return None
            
            center = 0.5 * (
                (n0 - p1.dot(p1)) * c23 + \
                (n0 - p2.dot(p2)) * c31 + \
                (n0 - p3.dot(p3)) * c12) / det
                
            radiuses = np.linalg.norm(points - center, axis=1)
            radius = sum(radiuses) / num_points
            
        # for more points, find the plane such that the summed squared distance 
        # from the plane to all points is minimized. 
        else:
            
            b = sum(points.T * points.T)
            a = np.c_[2 * points, np.ones(num_points)]
            X = np.linalg.lstsq(a, b)[0]
            
            center = X[:3]
            radius = np.sqrt(X[3] + center.dot(center))
        
        if radius < 0:
            return None

        return Sphere([center[0], center[1], center[2], radius]) 
        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    