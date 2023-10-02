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

from .PrimitiveBase import PrimitiveBase
    
class Sphere(PrimitiveBase):
    
    _fit_n_min = 4
    _model_args_n = 4
    _name = 'sphere'
    
    @staticmethod
    def get_mesh(model, points):
        mesh = TriangleMesh.create_sphere(radius=model[3])
        mesh.translate(model[:3])
        return mesh
    
    # if simplest case, the result is direct
    @staticmethod
    def get_distances(points, model):
        points = np.asarray(points)
        distances = np.linalg.norm(points - model[:3], axis=1) - model[3]
        return np.abs(distances)
    
    @staticmethod
    def get_normal_angles(points, normals, model):
        dist_vec = points - model[:3]
        dist_vec /= np.linalg.norm(dist_vec)
        angles = np.arccos(np.dot(dist_vec, model[:3]))
        return np.abs(angles)

    # for more points, find the plane such that the summed squared distance 
    # from the plane to all points is minimized.    
    @staticmethod
    def get_model(points, samples):
        points_ = np.asarray(points)[samples]
        
        # if simplest case, the result is direct
        if len(samples) == 4:
            p0, p1, p2, p3 = points_
            
            r1 = p0 - p1
            r2 = p0 - p2
            r3 = p0 - p3
            n0 = p0.dot(p0)
            c12 = np.cross(r1, r2)
            c23 = np.cross(r2, r3)
            c31 = np.cross(r3, r1)
            
            det = r1.dot(c23)
            if det == 0:
                return np.array([0, 0, 0, 0])
            
            center = 0.5 * (
                (n0 - p1.dot(p1)) * c23 + \
                (n0 - p2.dot(p2)) * c31 + \
                (n0 - p3.dot(p3)) * c12) / det
        
        else:
            b = sum(points_.T * points_.T)
            b = b[0] - b[1:]
            
            A = points_[0] - points_[1:]
            
            # try:
            #     A_ = np.linalg.pinv(A)
            #     center = 0.5 * A_ @ b
                
            # except np.linalg.LinAlgError:
            #     return np.array([0, 0, 0, 0])
            
            AT = A.T
            A = AT @ A
            det = np.linalg.det(A)
            if det == 0:
                return np.array([0, 0, 0, 0])
            
            A_ = np.linalg.inv(A) @ AT
            center = 0.5 * A_ @ b
            
        radiuses = np.linalg.norm(points_ - center, axis=1)
        radius = sum(radiuses) / len(samples)
        # radius = np.linalg.norm(radiuses)
        
        # print(f'center = {center}')
        # print(f'radius = {radius}')
        # print(f'points_ = {points_}')
        
        return np.hstack([center, radius]) 
        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    