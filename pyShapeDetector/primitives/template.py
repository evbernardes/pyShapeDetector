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

from .primitivebase import Primitive
    
class Template(Primitive):
    """
    Template primitive.
    
    Use this as a base to implement your own primitives.
    
    Attributes
    ----------
    fit_n_min 
    model_args_n
    name
    model
    equation
    surface_area 
    volume
    canonical
    color
    inlier_points
    inlier_points_flattened
    inlier_normals
    inlier_colors
    inlier_PointCloud
    inlier_PointCloud_flattened
    metrics
    axis_spherical
    axis_cylindrical
        
    Methods
    -------
    __init__
    __repr__
    __eq__
    random
    fit
    get_signed_distances
    get_distances
    get_normals
    get_angles_cos
    get_angles
    get_residuals
    flatten_points
    flatten_PointCloud
    add_inliers
    closest_inliers
    inliers_average_dist
    inliers_bounding_box
    sample_points_uniformly
    sample_points_density
    get_mesh
    get_cropped_mesh
    is_similar_to
    copy
    translate
    rotate
    align
    save
    load
    check_bbox_intersection
    check_inlier_distance
    fuse
    """
    
    _fit_n_min = 0
    _model_args_n = 0
    _name = 'template' 
    _color = np.array([0.1, 0.2, 0.3])
    
    @property
    def surface_area(self):
        """ Surface area of primitive """
        return 0
    
    @property
    def volume(self):
        """ Volume of primitive. """
        return 0
    
    @staticmethod
    def fit(points, normals=None):
        """ Gives shape that fits the input points. If the number of points is
        higher than the `_fit_n_min`, the fitted shape will return some kind of
        estimation.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        normals : N x 3 array
            N normal vectors
        
        Returns
        -------
        Template
            Dummy template.
        """
        # points_ = np.asarray(points)[samples]
        
        num_points = len(points)
        
        if num_points < 0:
            raise ValueError('A minimun of 0 points are needed to fit a '
                             'template')
        
        return Template([]) 
    
    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the model.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        
        Returns
        -------
        distances
            Nx1 array distances.
        """
        return np.zeros(len(points))
    
    def get_normals(self, points):
        """ Gives, for each input point, the normal vector of the point closest 
        to the primitive.
        
        Returns
        -------
        normals
            Nx3 array containing normal vectors.
        """
        return points
    
    def get_mesh(self, **options):
        """ Creates mesh of the shape.      
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the shape.
        """
        return TriangleMesh()
    