#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:26:20 2024

@author: ebernardes
"""
from open3d.geometry import PointCloud as open3d_PointCloud
from open3d.utility import Vector3dVector
import numpy as np

from .open3d_geometry import (
    link_to_open3d_geometry,
    Open3D_Geometry)

@link_to_open3d_geometry(open3d_PointCloud)
class PointCloud(Open3D_Geometry):
    
    # @property
    # def points(self):
    #     return np.asarray(self.as_open3d.points)
    
    # @points.setter
    # def points(self, points):
    #     self.as_open3d.points = Vector3dVector(points)
        
    # @property
    # def normals(self):
    #     return np.asarray(self.as_open3d.normals)
    
    # @normals.setter
    # def normals(self, normals):
    #     self.as_open3d.normals = Vector3dVector(normals)
        
    # @property
    # def colors(self):
    #     return np.asarray(self.as_open3d.colors)
    
    # @colors.setter
    # def colors(self, colors):
    #     self.as_open3d.colors = Vector3dVector(colors)
    
    def from_points_normals_colors(element, normals=None, colors=None):
        """ Creates PointCloud instance from points, normals or colors.

        Parameters
        ----------
        element : instance Primitive, PointCloud or numpy.array
            Shape with inliers or Points of the PointCloud.
        normals : numpy.array, optional
            Normals for each point of the PointCloud, must have same length as 
            points. Default: None.
        colors : numpy.array, optional
            Colors for each point of the PointCloud, must have same length as 
            points. Default: None.
            
        Returns
        -------
        Open3d.geometry.PointCloud
            PointCloud defined by the inputs.
        """
        from pyShapeDetector.primitives import Primitive
        
        if isinstance((shape := element), Primitive):
            points = shape.inlier_points
            if normals is None:
                normals = shape.inlier_normals
            if colors is None:
                colors = shape.inlier_colors
        else:
            points = element
        
        pcd = PointCloud()
        pcd.points = points
        pcd.normals = normals
        pcd.colors = colors
        # if len(points) == 0:
        #     return pcd
        # pcd.points = points
        # if normals is not None and len(normals) > 0:
        #     pcd.normals = normals
        # if colors is not None and len(colors) > 0:
        #     pcd.colors = colors
        return pcd
    pass