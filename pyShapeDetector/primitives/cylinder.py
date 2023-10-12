#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:57:08 2023

@author: ebernardes
"""
import warnings
import numpy as np
from open3d.geometry import TriangleMesh
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
    def height(self):
        return np.linalg.norm(self.vector)
    
    @property
    def axis(self):
        return self.vector / self.height
    
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
        return self.get_rotation_from_axis(self.axis)
    
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
            # Reference for axis estimation with normals: 
            # http://dx.doi.org/10.1016/j.cag.2014.09.027
            normals = np.asarray(normals)
            if len(normals) != num_points:
                raise ValueError('Different number of points and normals')
        
            eigval, eigvec = np.linalg.eig(normals.T @ normals)
            idx = eigval == min(eigval)
            if sum(idx) != 1:  # no well defined minimum eigenvalue
                return None
            
            axis = eigvec.T[idx][0]
            
            # Reference for the rest:
            # Was revealed to me in a dream
            axis_neg_squared_skew = np.eye(3) - axis[np.newaxis].T * axis
            points_skew = (axis_neg_squared_skew @ points.T).T
            b = sum(points_skew.T * points.T)
            a = np.c_[2 * points_skew, np.ones(num_points)]
            X = np.linalg.lstsq(a, b, rcond=None)[0]
            
            center = X[:3]
            radius = np.sqrt(X[3] + center.dot(axis_neg_squared_skew @ center))
            
            # find point in base of cylinder
            proj = points.dot(axis)
            idx = np.where(proj == min(proj))[0][0]
            point = center + points[idx].dot(axis) * axis            
            
            point = list(point)
            vector = list(axis * (max(proj) - min(proj)))
        
        else:
            # if no normals, use scikit spatial, slower
            warnings.warn('Cylinder fitting works much quicker if normals '
                          'are given.')
            solution = skcylinder.best_fit(points)
            
            point = list(solution.point)
            vector = list(solution.vector)
            radius = solution.radius
        
        return Cylinder(point+vector+[radius]) 
