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
        return self.axis / self.height
    
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
            # Reference: http://dx.doi.org/10.1016/j.cag.2014.09.027
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
            a = np.c_[2 * projection_plane, np.ones(num_points)]
            X = np.linalg.lstsq(a, b, rcond=None)[0]
            
            radius = np.sqrt(X[2] + X[:2].dot(X[:2]))
            
            # find point in base of cylinder
            idx = np.where(projection_axis == min(projection_axis))[0][0]
            point = X[0] * ax + X[1] * ay + points[idx].dot(axis) * axis            
            
            point = list(point)
            height = max(projection_axis) - min(projection_axis)
            vector = list(axis * height)
        
        else:
            # if no normals, use scikit spatial, slower
            warnings.warn('Cylinder fitting works much quicker if normals '
                          'are given.')
            solution = skcylinder.best_fit(points)
            
            point = list(solution.point)
            vector = list(solution.vector)
            radius = solution.radius
        
        return Cylinder(point+vector+[radius]) 
