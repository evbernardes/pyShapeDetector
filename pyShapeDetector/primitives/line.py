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
from .cylinder import Cylinder
    
class Line(Primitive):
    """
    Template primitive.
    
    Use this as a base to implement your own primitives.
    
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
    canonical : Line
        Return canonical form for testing.
    surface_area : float
        Surface area of primitive
    volume : float
        Volume of primitive.
    beginning : 3 x 1 float
        Start point of line
    ending : 3 x 1 float
        End point of line
    vector : 3 x 1 array
        Vector from beginning point to end point.
    axis : 3 x 1 array
        Unit vector defining axis of line.
    length : float
        Length of line.
         
    Methods
    -------
    inliers_bounding_box(slack=0):
        If the shape includes inlier points, returns the minimum and 
        maximum bounds of their bounding box.
        
    get_distances(points)
        Gives the minimum distance between each point to the model. 
        
    get_normals(points)
        Gives, for each input point, the normal vector of the point closest 
        to the primitive. 
        
    random(scale):
        Generates a random shape.
        
    from_point_vector(beginning, vector):
        Creates cylinder from center base point, vector and radius as 
        separated arguments.
        
    from_two_points(beginning, ending):
        Creates cylinder from center base point, vector and radius as 
        separated arguments.
        
    closest_to_line(points):
        Returns points in line that are the closest to the input
        points.
        
    get_orthogonal_component(points):
        Removes the axis-aligned components of points.
        
    point_from_projection(projection):
        Gives point in line whose projection is equal to the input value.
        
    from_plane_intersection(plane1, plane2):
        Calculate the line defining the intersection between two planes.
    
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
        
    get_mesh(resolution=5, radius_ratio=0.001):
        Creates mesh of the shape.    
    """
    
    _fit_n_min = 2
    _model_args_n = 6
    _name = 'line' 
    _translatable = [0, 1, 2]
    _rotatable = [3, 4, 5]
    
    @property
    def beginning(self):
        """ Start point of line. """
        return np.array(self.model[:3])
    
    @property
    def ending(self):
        """ End point of line. """
        return self.beginning + self.vector

    @property
    def vector(self):
        """ Vector from beginning point to end point. """
        return np.array(self.model[3:])
    
    @property
    def axis(self):
        """ Unit vector defining axis of line. """
        return self.vector / self.length
    
    @property
    def length(self):
        """ Length of line. """
        return np.linalg.norm(self.vector)
    
    @property
    def surface_area(self):
        """ Surface area of primitive """
        return 0
    
    @property
    def volume(self):
        """ Volume of primitive. """
        return 0
    
    @classmethod
    def from_point_vector(cls, beginning, vector):
        """ Creates cylinder from center base point, vector and radius as 
        separated arguments.
        
        Parameters
        ----------            
        beginning : 3 x 1 array
            First point in line.
        vector : 3 x 1 array
            Direction of line.

        Returns
        -------
        Line
            Generated shape.
        """
        # vector = np.array(ending) - np.array(beginning)
        return cls(list(beginning) + list(vector))
    
    @classmethod
    def from_two_points(cls, beginning, ending):
        """ Creates cylinder from center base point, vector and radius as 
        separated arguments.
        
        Parameters
        ----------            
        beginning : 3 x 1 array
            First point in line.
        ending : 3 x 1 array
            Ending point in line.

        Returns
        -------
        Line
            Generated shape.
        """
        vector = np.array(ending) - np.array(beginning)
        return cls.from_point_vector(beginning, vector)
    
    def closest_to_line(self, points):
        """ Returns points in line that are the closest to the input
        points.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        
        Returns
        -------
        points_closest: N x 3 array
            N points in line
        """
        points = np.asarray(points)
        projection = (points - self.beginning).dot(self.axis)
        return self.beginning + projection[..., np.newaxis] * self.axis
    
    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the line. 
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        
        Returns
        -------
        distances
            Nx1 array distances.
        """
        points = np.asarray(points)             
        distances = np.linalg.norm(
            np.cross(self.axis, points - self.beginning), axis=1)
        
        return distances
    
    def get_orthogonal_component(self, points):
        """ Removes the axis-aligned components of points.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        
        Returns
        -------
        points_orthogonal: N x 3 array
            N points
        """
        points = np.asarray(points)
        delta = points - self.beginning
        return -np.cross(self.axis, np.cross(self.axis, delta))
    
    def get_normals(self, points):
        """ Gives, for each input point, the normal vector of the point closest 
        to the cylinder.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        
        Returns
        -------
        normals
            Nx3 array containing normal vectors.
        """
        normals = self.get_orthogonal_component(points)
        normals /= np.linalg.norm(normals, axis=1)[..., np.newaxis]
        return normals
    
    def get_mesh(self, resolution=5, radius_ratio=0.001):
        """ Creates mesh of the shape.      
        
        Parameters
        ----------
        resolution : int, optional
            Mesh resolution. Default: 5.
        radius_ratio : float, optional
            Ratio between line thickness and line length. Default: 0.001.
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the shape.
        """
        cylinder = Cylinder.from_base_vector_radius(
            self.beginning, self.vector, self.length * radius_ratio)
            # self.beginning, self.vector, radius_ratio)
        return cylinder.get_mesh(resolution=resolution, closed=True)
    
    def point_from_projection(self, projection):
        """ Gives point in line whose projection is equal to the input value.
        
        Parameters
        ----------
        projection : float
            Target projection. 
        
        Returns
        -------
        point
            Nx1 array containing normal vectors.
        """
        return self.beginning + self.axis * projection
    
    def point_from_intersection(self, other, within_segment=True, eps=1e-3):
        """ Calculates intersection point between two lines.
        
        Parameters
        ----------
        other : instance of Line
            Other line to instersect.
        within_segment : boolean, optional
            If set to False, will suppose lines are infinite. Default: True.
        
        Returns
        -------
        point or None
            1x3 array containing point.
        """
        if not isinstance(other, Line):
            raise ValueError("'other' must be an instance of Line.")
            
        # dot_vectors = np.dot(self.vector, other.vector)
        
        if abs(abs(np.dot(self.axis, other.axis)) - 1) < 1e-7:
            return None
        
        dot_vectors = np.dot(self.vector, other.vector)
        diff = self.beginning - other.beginning
        
        A = np.array([
            [-self.length ** 2, dot_vectors],
            [dot_vectors, -other.length ** 2]])
        
        B = np.array([-self.vector.dot(diff), other.vector.dot(diff)])
        
        ta, tb = np.linalg.lstsq(A, B, rcond=None)[0]
        
        # not sure about including this "-1" there:
        if within_segment:
            if not (-1 - eps < ta < 1 + eps) or not (-1 -eps < tb < 1 + eps):
                return None
            
        pa = self.beginning + self.vector * ta
        pb = other.beginning + other.vector * tb
        return (pa + pb) / 2
        
    
    @staticmethod
    def from_plane_intersection(plane1, plane2):
        """ Calculate the line defining the intersection between two planes.
        
        If the planes are not bounded, give a line of length 1.
        
        Parameters
        ----------
        plane1 : instance of Plane of PlaneBounded
            First plane
        plane2 : instance of Plane of PlaneBounded
            Second plane
        
        Returns
        -------
        Line or None
        """
        axis = np.cross(plane1.normal, plane2.normal)
        norm = np.linalg.norm(axis)
        if norm < 1e-5:
            return None
        axis /= norm
        
        A = np.vstack([plane1.normal, plane2.normal])
        B = -np.vstack([plane1.dist, plane2.dist])
        point = np.linalg.lstsq(A, B, rcond=None)[0].T[0]
        
        line = Line.from_point_vector(point, axis)
        
        projections = []
        if hasattr(plane1, 'bounds'):
            projections += list(np.dot(plane1.bounds, axis))
        if hasattr(plane2, 'bounds'):
            projections += list(np.dot(plane2.bounds, axis))
        if len(projections) > 0:
            line = Line.from_two_points(
                line.point_from_projection(min(projections)), 
                line.point_from_projection(max(projections)))
        
        return line
    
    @staticmethod
    def fit(points, normals=None):
        """ Not defined for lines.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        normals : N x 3 array
            N normal vectors
        
        Returns
        -------
        None
        """
        raise RuntimeError('Fitting is not defined for lines.')
        
