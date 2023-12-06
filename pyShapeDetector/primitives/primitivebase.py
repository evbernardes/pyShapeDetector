#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
from abc import ABC, abstractmethod
import numpy as np
from open3d.geometry import PointCloud, AxisAlignedBoundingBox
from open3d.utility import Vector3dVector

from scipy.spatial.transform import Rotation
    
class Primitive(ABC):
    """
    Base class used to represent a geometrical primitive.
    
    To define a primitive, inherit from this class and define at least the 
    following internal attributes:
        `_fit_n_min`
        `_model_args_n`
        `_name`
    And the following methods:
        `get_distances`
        `get_normals`
        
    The method `get_mesh` can also optionally be implemented to return a 
    TriangleMesh instance.
    
    The properties `surface_area` and `volume` can also be implemented.
    
    When multiple set of parameters can define the same surface, it might be
    useful to implement the property `canonical` to return the canonical form 
    (useful for testing).
    
    Attributes
    ----------
    fit_n_min : int
        Minimum number of points necessary to fit a model.
    model_args_n : str
        Number of parameters in the model.
    name : str
        Name of primitive.
    equation : str
        Equation that defines the primitive.
    canonical : Primitive
        Return canonical form for testing.
    surface_area : float
        Surface area of primitive
    volume : float
        Volume of primitive.
    inlier_points : N x 3 array
        Convenience attribute that can be set to save inlier points
        
    Methods
    -------    
    get_signed_distances(points):
        Gives the minimum distance between each point to the model. 
    
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
        
    flatten_pcd(pcd):
        Return new pointcloud with flattened points.
        
    flatten_points(points):
        Stick each point in input to the closest point in shape's surface.
        
    get_angles_cos(points, normals):PrimitiveBase
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
        
    get_angles(points, normals):
        Gives the angles between the input normal vectors and the 
        calculated normal vectors from the input points.
        
    get_residuals(points, normals):
        Convenience function returning both distances and angles.
        
    get_mesh():
        Creates mesh of the shape.
    """
    _inlier_points = np.asarray([])
    _inlier_normals = np.asarray([])
    # _inlier_indices = np.asarray([])
    _metrics = {}
    
    def __repr__(self):
        round_ = lambda x:round(x, 5)
        params = list(map(round_, self.model))
        return type(self).__name__+'('+str(params)+')'
    
    
    def add_inliers(self, points, normals=None):
        points = np.asarray(points)
        if points.shape == (3, ):
            points = np.reshape(points, (1,3))
        elif points.shape[1] != 3:
            raise ValueError('Invalid shape for input points, must be a single'
                             ' point or an array of shape (N, 3), got '
                             f'{points.shape}')
        self._inlier_points = points
        
        if normals is None:
            pass
        
        normals = np.asarray(normals)
        if normals.shape == (3, ):
            normals = np.reshape(normals, (1,3))
        elif normals.shape[1] != 3:
            raise ValueError('Invalid shape for input normals, must be a single'
                             ' point or an array of shape (N, 3), got '
                             f'{normals.shape}')
        self._inlier_normals = normals
    
    @property
    def color(self):
        seed = int(str(abs(hash(self.name)))[:9])
        np.random.seed(seed)
        return np.random.random(3)
        
    @property
    def inlier_points(self):
        """ Convenience attribute that can be set to save inlier points """
        return self._inlier_points
    
    # @inlier_points.setter
    # def inlier_points(self, points):
    #     points = np.asarray(points)
    #     if points.shape == (3, ):
    #         points = np.reshape(points, (1,3))
    #     elif points.shape[1] != 3:
    #         raise ValueError('Invalid shape for input points, must be a single'
    #                          ' point or an array of shape (N, 3), got '
    #                          f'{points.shape}')
    #     self._inlier_points = points
        
    @property
    def inlier_normals(self):
        """ Convenience attribute that can be set to save inlier points """
        return self._inlier_normals
    
    # @inlier_normals.setter
    # def inlier_normals(self, normals):
    #     normals = np.asarray(normals)
    #     if normals.shape == (3, ):
    #         normals = np.reshape(normals, (1,3))
    #     elif normals.shape[1] != 3:
    #         raise ValueError('Invalid shape for input normals, must be a single'
    #                          ' point or an array of shape (N, 3), got '
    #                          f'{normals.shape}')
    #     self._inlier_normals = normals
        
    # @property
    # def inlier_indices(self):
    #     """ Convenience attribute that can be set to save inlier points """
    #     return self._inlier_indices
    
    # @inlier_indices.setter
    # def inlier_indices(self, indices):
    #     indices = np.asarray(indices)
    #     if len(indices.shape) != 1:
    #         raise ValueError('Invalid shape for input indices.')
    #     self._inlier_indices = indices
        
    @property
    def metrics(self):
        """ Convenience attribute that can be set to save shape metrics """
        return self._metrics
    
    @metrics.setter
    def metrics(self, metrics):
        if type(metrics) != dict:
            raise ValueError('metrics should be a dict')
        self._metrics = metrics
    
    @property
    def canonical(self):
        """ Return canonical form for testing."""
        return self
    
    @property
    def equation(self):
        """ Equation that defines the primitive."""
        raise NotImplementedError(f'Equation not implemented for {self.name} '
                                  'primitives.')
    
    @property
    def fit_n_min(self):
        """ Minimum number of points necessary to fit a model."""
        return self._fit_n_min

    @property
    def model_args_n(self):
        """ Number of parameters in the model. """
        return self._model_args_n
    
    @property
    def name(self):
        """ Name of primitive. """
        return self._name
    
    @property
    def surface_area(self):
        """ Surface area of primitive. """
        
        raise NotImplementedError('Surface area not implemented for '
                                  f'{self.name} primitives.')
    
    @property
    def volume(self):
        """ Volume of primitive. """
        raise NotImplementedError(f'Volume not implemented for {self.name} '
                                  'primitives.')
    
    @abstractmethod
    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the model. 
        
        Actual implementation depends on the type of primitive.
        
        Parameters
        ----------
        points : N x 3 array
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
        points : N x 3 array
            N input points 
        
        Returns
        -------
        normals
            Nx3 array containing normal vectors.
        """
        pass
    
    @classmethod
    def random(cls, scale=1):
        """ Generates a random shape.
        
        Parameters
        ----------
        scale : float, optional
            scaling factor for random model values.

        Returns
        -------
        Primitive
            Random shape.
        """
        return cls(np.random.random(cls._model_args_n) * scale)
    
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
        points : N x 3 array
            N input points 
        normals : N x 3 array
            N normal vectors
        
        Returns
        -------
        Primitive
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
        
        if len(model) != self.model_args_n:
            raise ValueError(f'{self.name.capitalize()} primitives take '
                             f'{self.model_args_n} elements, got {model}')
        self.model = model
        
    @staticmethod
    def get_rotation_from_axis(axis_origin, axis):
        """ Rotation matrix that transforms `axis_origin` in `axis`.
        
        Parameters
        ----------
        axis : 3 x 1 array
            Goal axis.
        axis_origin : 3 x 1 array
            Initial axis.
        
        Returns
        -------
        rotation
            3x3 rotation matrix
        """
        axis = np.array(axis) / np.linalg.norm(axis)
        axis_origin = np.array(axis_origin) / np.linalg.norm(axis_origin)
        if abs(axis.dot(axis_origin) + 1) > 1E-6:
            # axis_origin = -axis_origin
            halfway_axis = (axis_origin + axis)[..., np.newaxis]
            halfway_axis /= np.linalg.norm(halfway_axis)
            return 2 * halfway_axis * halfway_axis.T - np.eye(3)
        else:
            orthogonal_axis = np.cross(np.random.random(3), axis)
            orthogonal_axis /= np.linalg.norm(orthogonal_axis)
            return Rotation.from_quat(list(orthogonal_axis)+[0]).as_matrix()
            
    
    def flatten_pcd(self, pcd):
        """ Return new pointcloud with flattened points.
        
        Parameters
        ----------
        pcd : Open3D.geometry.PointCloud
            Input pointcloud
        
        Returns
        -------
        Open3D.geometry.PointCloud
            Pointcloud with points flattened
            
        """
        pcd_flattened = PointCloud()
        pcd_flattened.points = Vector3dVector(self.flatten_points(pcd.points))
        pcd_flattened.colors = pcd.colors
        return pcd_flattened
    
    def flatten_points(self, points):
        """ Stick each point in input to the closest point in shape's surface.
        
        Parameters
        ----------
        points : N x 3 array
            N input points
        
        Returns
        -------
        points_flattened : N x 3 array
            N points on the surface
            
        """
        if len(points) == 0:
            return points
    
        points = np.asarray(points)        
        difference = self.get_signed_distances(points)[..., np.newaxis] * self.get_normals(points)
        points_flattened = points - difference
        return points_flattened
    
    def get_angles_cos(self, points, normals):
        """ Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        normals : N x 3 array
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
        points : N x 3 array
            N input points 
        normals : N x 3 array
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
        points : N x 3 array
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
        points : N x 3 array
            N input points 
        normals : N x 3 array
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
    def get_mesh(self, resolution=30):
        """ Creates mesh of the shape.
        
        Parameters
        ----------
        resolution : int, optional
            Resolution parameter for mesh. Default: 30
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the shape.
        """
        raise NotImplementedError('The mesh generating function for '
                                  f'primitives of type {self.name} has not '
                                  'been implemented.')
        
    def get_cropped_mesh(self, points=None, eps=1E-3):
        mesh = self.get_mesh()
        points_flattened = self.flatten_points(points)
        pcd = PointCloud(Vector3dVector(points_flattened))
        bb = pcd.get_axis_aligned_bounding_box()
        bb = AxisAlignedBoundingBox(bb.min_bound - [eps]*3, 
                                    bb.max_bound + [eps]*3)
        return mesh.crop(bb)
    
    def is_similar_to(self, other_shape, rtol=1e-02, atol=1e-02):
        """ Check if shapes represent same model.
        
        Parameters
        ----------
        other_shape : Primitive
            Primitive to compare
        
        Returns
        -------
        Bool
            True if shapes are similar.
        """
        if not isinstance(self, type(other_shape)):
            return False
        
        compare = np.isclose(self.canonical.model, other_shape.canonical.model,
                             rtol=rtol, atol=atol)
        return compare.all()
            
    def __eq__(self, other_shape):
        return self.is_similar_to(other_shape, rtol=1e-05, atol=1e-08)
        
