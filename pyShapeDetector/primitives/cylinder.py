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
    
    def get_mesh(self, points):
        
        points = np.asarray(points)
        point_in_line = np.array(self.model[:3])
        axis = np.array(self.model[3:6])
        center = point_in_line + axis / 2
        axis /= np.linalg.norm(axis)
        radius = self.model[-1]
        
        if axis.dot([0, 0, 1]) == 0:
            axis = -axis
        
        diff = points - point_in_line
        projection = np.dot(diff, axis)
        height = max(projection) - min(projection)
        
        mesh = TriangleMesh.create_cylinder(radius=radius, height=height)
        
        # # remove top and bottom planes
        # bbox = mesh.get_axis_aligned_bounding_box()
        # x, y, z = bbox.min_bound
        # bbox_min = [x, y, z+0.00001]
        # x, y, z = bbox.max_bound
        # bbox_max = [x, y, z-0.00001]
        # bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
        # mesh = mesh.crop(bbox)
        
        halfway_axis = (np.array([0, 0, 1]) + axis)[..., np.newaxis]
        halfway_axis /= np.linalg.norm(halfway_axis)
        rot = 2 * halfway_axis * halfway_axis.T - np.eye(3)
        
        mesh.rotate(rot)
        mesh.translate(center)
        
        return mesh
    
    @staticmethod
    def fit(points):
        # points_ = np.asarray(points)[samples]
        
        num_points = len(points)
        
        if num_points < 6:
            raise ValueError('A minimun of 6 points are needed to fit a '
                             'cylinder')
        
        solution = skcylinder.best_fit(points)
        
        point = list(solution.point)
        vector = list(solution.vector)
        radius = [solution.radius]
        
        return Cylinder(point+vector+radius) 
