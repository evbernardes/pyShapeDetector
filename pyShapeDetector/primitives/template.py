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
    _fit_n_min : int
        Minimum number of points necessary to fit a model.
    _model_args_n : str
        Number of parameters in the model.
    name : str
        Name of primitive.
    equation : str
        Equation that defines the primitive.
    canonical : Template
        Return canonical form for testing.
    surface_area : float
        Surface area of primitive
    volume : float
        Volume of primitive.
    
    Methods
    -------
    get_distances(points)
        Gives the minimum distance between each point to the model. 
        
    get_normals(points)
        Gives, for each input point, the normal vector of the point closest 
        to the primitive. 
        
    random(scale):
        Generates a random shape.
        
    fit(points, normals=None):
        Gives shape that fits the input points. If the number of points is
        higher than the `_fit_n_min`, the fitted shape will return some kind of
        estimation. 
    
    get_angles_cos(points, normals):
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
    
    get_rotation_from_axis(axis, axis_origin=[0, 0, 1])
        Rotation matrix that transforms `axis_origin` in `axis`.
        
    flatten_points(points):
        Stick each point in input to the closest point in shape's surface.
        
    get_angles_cos(points, normals):
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
        
    get_angles(points, normals):
        Gives the angles between the input normal vectors and the 
        calculated normal vectors from the input points.
        
    get_residuals(points, normals):
        Convenience function returning both distances and angles.
        
    create_limits(args_n, idx, value):
        Create a list of length `args_n` that stores `value` at index `idx`
        and `None` elsewhere.
        
    get_mesh(points=None):
        Creates mesh of the shape.    
    """
    
    _fit_n_min = 0
    _model_args_n = 0
    name = 'template' 
    
    @property
    def surface_area(self):
        """ Surface area of primitive """
        return 0
    
    @property
    def volume(self):
        """ Volume of primitive. """
        return 0
    
    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the model.
        
        Parameters
        ----------
        points : 3 x N array
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
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        normals
            Nx3 array containing normal vectors.
        """
        return points
    
    def get_mesh(self, points=None):
        """ Creates mesh of the shape.      
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the shape.
        """
        return TriangleMesh()
    
    @staticmethod
    def fit(points, normals=None):
        # points_ = np.asarray(points)[samples]
        
        num_points = len(points)
        
        if num_points < 0:
            raise ValueError('A minimun of 0 points are needed to fit a '
                             'template')
        
        return Template([]) 
