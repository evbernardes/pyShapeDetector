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

from skspatial.objects.cylinder import Cylinder as skcylinder

from .primitivebase import PrimitiveBase
    
class Cylinder(PrimitiveBase):
    
    _fit_n_min = 6
    _model_args_n = 7
    name = 'cylinder' 
    
    @property
    def point(self):
        return np.array(self.model[:3])
    
    @property
    def vector(self):
        return np.array(self.model[3:6])
    
    @property
    def axis(self):
        v = self.vector
        return v / np.linalg.norm(v)
    
    @property
    def radius(self):
        return self.model[-1]
    
    @property
    def center(self):
        return self.point + self.vector / 2
    
    @staticmethod
    def limit_radius(value):
        return PrimitiveBase.create_limits(
            Cylinder._model_args_n, 6, value)
    
    def _closest_to_line(self, points):
        points = np.asarray(points)
        projection = (points - self.point).dot(self.axis)
        return self.point + projection[..., np.newaxis] * self.axis

    def get_distances(self, points):
        points = np.asarray(points)      
        points_closest = self._closest_to_line(points)
        distances = np.linalg.norm(points_closest - points, axis=1)
        return np.abs(distances - self.radius)
    
    def get_normals(self, points):
        points = np.asarray(points)     
        normals = points - self._closest_to_line(points)
        normals /= np.linalg.norm(normals, axis=1)[..., np.newaxis]
        return normals
    
    @property
    def rotation_from_axis(self):
        axis = self.axis
        if axis.dot([0, 0, 1]) == 0:
            axis = -axis
        halfway_axis = (np.array([0, 0, 1]) + axis)[..., np.newaxis]
        halfway_axis /= np.linalg.norm(halfway_axis)
        return 2 * halfway_axis * halfway_axis.T - np.eye(3)
    
    def get_mesh(self, points):
        
        points = np.asarray(points)
        
        mesh = TriangleMesh.create_cylinder(radius=self.radius, 
                                            height=np.linalg.norm(self.vector))

        mesh.rotate(self.rotation_from_axis)
        mesh.translate(self.center)
        
        return mesh
    
    @staticmethod
    def fit(points, normals=None):
        points = np.asarray(points)
        
        num_points = len(points)
        
        if num_points < 6:
            raise ValueError('A minimun of 6 points are needed to fit a '
                             'cylinder')
            
        if normals is not None:
            # use http://dx.doi.org/10.1016/j.cag.2014.09.027
            normals = np.asarray(normals)
            if len(normals) != num_points:
                raise ValueError('Different number of points and normals')
        
            eigval, eigvec = PrimitiveBase.get_normal_eig(normals)
            idx = eigval == min(eigval)
            if sum(idx) != 1:  # no well defined minimum eigenvalue
                return None
            
            axis = eigvec.T[idx][0]
            ax, ay = eigvec.T[~idx]
            
            projections = points.dot(eigvec)
            projection_plane = projections.T[~idx].T
            projection_axis = projections.T[idx].T
            
            b = sum(projection_plane.T * projection_plane.T)
            b = b[0] - b[1:]
            A = projection_plane[0] - projection_plane[1:]
            
            AT = A.T
            A = AT @ A
            det = np.linalg.det(A)
            if det == 0:
                return None
            
            A_ = np.linalg.inv(A) @ AT
            X = 0.5 * A_ @ b
            
            # find point in base of cylinder
            idx = np.where(projection_axis == min(projection_axis))[0][0]
            point = X[0] * ax + X[1] * ay + points[idx].dot(axis) * axis
            
            radiuses = np.linalg.norm(projection_plane - X, axis=1)
            radius = [sum(radiuses) / num_points]
            
            point = list(point)
            vector = list(axis * (max(projection_axis) - min(projection_axis)))
            
            # print(f'v: {vector}, p: {point}, r:{radius}')
        
        else:
            # if no normals, use scikit spatial, slower
            solution = skcylinder.best_fit(points)
            
            point = list(solution.point)
            vector = list(solution.vector)
            radius = [solution.radius]
        
        return Cylinder(point+vector+radius) 
