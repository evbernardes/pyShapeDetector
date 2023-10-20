#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
from abc import ABC, abstractmethod
import numpy as np
    
class PrimitiveBase(ABC):
    """
    Base class used to represent a geometrical primitive.
    
    To define a primitive, inherit from this class and define at least the 
    following properties:
        `_fit_n_min`
        `_model_args_n`
        `name`
    And the following methods:
        `get_distances`
        `get_normals`
        
    The method `get_mesh` can also optionally be implemented to return a 
    TriangleMesh instance.
    
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
    
    Methods
    -------
    get_distances(points)
        Gives the minimum distance between each point to the model. 
        
    get_normals(points)
        Gives, for each input point, the normal vector of the point closest 
        to the primitive. 
        
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
        Creates mesh of the shape. Points are not always necessary.
    """
    @property
    def equation(self):
        """ Equation that defines the primitive."""
        raise NotImplementedError(f'Equation not implemented for {self.name} '
                                  'primitives.')
    
    @property
    @abstractmethod
    def _fit_n_min(self):
        """ Minimum number of points necessary to fit a model."""
        pass

    @property
    @abstractmethod
    def _model_args_n(self):
        """ Number of parameters in the model. """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """ Name of primitive. """
        pass
    
    @abstractmethod
    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the model. 
        
        Actual implementation depends on the type of primitive.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        distances
            Nx1 array distances.
        """
        pass
    
    @abstractmethod
    def get_normals(self, points):
        """ Gives, for each input point, the normal vector of the point closest 
        to the primitive. 
        
        Actual implementation depends on the type of primitive.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        normals
            Nx3 array containing normal vectors.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def fit(points, normals=None):
        """ Gives shape that fits the input points. If the number of points is
        higher than the `_fit_n_min`, the fitted shape will return some kind of
        estimation. 
        
        Moreover, some primitives do not need the normal vectors to fit, while
        others (like cylinders) might benefit from it.
        
        Actual implementation depends on the type of primitive, m
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        normals : 3 x N array
            N normal vectors
        
        Returns
        -------
        PrimitiveBase
            Fitted shape.
        """
        pass

    def __init__(self, model):
        """
        Parameters
        ----------
        model : list or tuple
            Parameters defining the shape model
                        
        Raises
        ------
        ValueError
            If number of parameters is incompatible with the model of the 
            primitive.
        """
        
        if len(model) != self._model_args_n:
            raise ValueError(f'{self.name.capitalize()} primitives take '
                             f'{self._model_args_n} elements, got {model}')
        self.model = model
        
    @staticmethod
    def get_rotation_from_axis(axis, axis_origin=[0, 0, 1]):
        """ Rotation matrix that transforms `axis_origin` in `axis`.
        
        Parameters
        ----------
        axis : 3 x 1 array
            Goal axis (default is None)
        axis_origin : 3 x 1 array, optional
            Initial axis (default is the z-axis)
        
        Returns
        -------
        rotation
            3x3 array representing a rotation matrix
        """
        axis_origin = np.array(axis_origin)
        if axis.dot(axis_origin) == 0:
            axis = -axis
        halfway_axis = (axis_origin + axis)[..., np.newaxis]
        halfway_axis /= np.linalg.norm(halfway_axis)
        return 2 * halfway_axis * halfway_axis.T - np.eye(3)
    
    def flatten_points(self, points):
        """ Stick each point in input to the closest point in shape's surface.
        
        Parameters
        ----------
        points : 3 x N array
            N input points
        
        Returns
        -------
        points_flattened : 3 x N array
            N points on the surface
            
        """
        points = np.asarray(points)
        distances = self.get_signed_distances(points)
        points_flattened = points - (distances * self.normal[..., np.newaxis]).T
        distances_flatened = self.get_distances(points_flattened)
        if sum(distances_flatened) > 1e-7:
            raise ValueError('Error of flattened distance adds up to'
                             f'{sum(distances_flatened)}')
        return points_flattened
    
    def get_angles_cos(self, points, normals):
        """ Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        normals : 3 x N array
            N normal vectors
            
        Raises
        ------
        ValueError
            If number of points and normals are not equal
        
        Returns
        -------
        angles_cos or None
            Nx1 array with the absolute value of the cosines of the angles, or
            `None` if `normals` is `None`.
            
        """
        if len(normals) != len(points):
            raise ValueError('Number of points and normals should be equal.')
        
        if normals is None:
            return None
        normals = np.asarray(normals)
        normals_from_points = self.get_normals(points)
        angles_cos = np.clip(
            np.sum(normals * normals_from_points, axis=1), -1, 1)
        return np.abs(angles_cos)
    
    def get_angles(self, points, normals):
        """ Gives the angles between the input normal vectors and the 
        calculated normal vectors from the input points.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        normals : 3 x N array
            N normal vectors
            
        Raises
        ------
        ValueError
            If number of points and normals are not equal
        
        Returns
        -------
        angles or None
            Nx1 array with the angles, or `None` if `normals` is `None`.
            
        """
        if normals is None:
            return None
        
        return np.arccos(
            self.get_angles_cos(points, normals))
    
    def get_distances(self, points):
        """ Gives the absolute value of the minimum distance between each point 
        to the model. 
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        distances
            Nx1 array distances.
        """
        return abs(self.get_signed_distances(points))
    
    def get_residuals(self, points, normals):
        """ Convenience function returning both distances and angles.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        normals : 3 x N array
            N normal vectors
            
        Raises
        ------
        ValueError
            If number of points and normals are not equal
        
        Returns
        -------
        tuple
            Tuple containing distances and angles.
            
        """
        return self.get_distances(points), \
            self.get_angles(points, normals)

    @staticmethod
    def create_limits(args_n, idx, value):
        """ Create a list of length `args_n` that stores `value` at index `idx`
        and `None` elsewhere.
        
        This is used to create max or min checks when testing the fitted 
        primitives.

        Parameters
        ----------
        args_n : int
            Length of list
        idx : int
            Index to store value
        value : float
            Value limit value.
            
        Raises
        ------
        IndexError
            If `idx` is greater or equal to `args_n`
        
        Returns
        -------
        list
            List containing limit.
        """
        values = [None] * args_n
        values[idx] = value
        return values
    
    @staticmethod
    def get_mesh(self, points=None):
        """ Creates mesh of the shape. Points are not always necessary.

        Parameters
        ----------
        points, optional : 3 x N array
            Points corresponding to the fitted shape.
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the shape.
        """
        raise NotImplementedError('The mesh generating function for '
                                  f'primitives of type {self.name} has not '
                                  'been implemented.')
        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    