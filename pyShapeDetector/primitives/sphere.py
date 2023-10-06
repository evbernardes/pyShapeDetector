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
from open3d.geometry import TriangleMesh

from .primitivebase import PrimitiveBase
    
class Sphere(PrimitiveBase):
    
    _fit_n_min = 4
    _model_args_n = 4
    _name = 'sphere'
    
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
    def fit(points):
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
                
        # for more points, find the plane such that the summed squared distance 
        # from the plane to all points is minimized. 
        else:
            b = sum(points.T * points.T)
            b = b[0] - b[1:]
            
            A = points[0] - points[1:]
            
            # try:
            #     A_ = np.linalg.pinv(A)
            #     center = 0.5 * A_ @ b
                
            # except np.linalg.LinAlgError:
            #     return np.array([0, 0, 0, 0])
            
            AT = A.T
            A = AT @ A
            det = np.linalg.det(A)
            if det == 0:
                return None
            
            A_ = np.linalg.inv(A) @ AT
            center = 0.5 * A_ @ b
            
        radiuses = np.linalg.norm(points - center, axis=1)
        radius = sum(radiuses) / num_points
        
        return Sphere([center[0], center[1], center[2], radius]) 
        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    