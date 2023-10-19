#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
import numpy as np
from open3d.geometry import TriangleMesh

from .primitivebase import PrimitiveBase
    
class Sphere(PrimitiveBase):
    """
    Sphere primitive.
    
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
    radius : float
        Radius of the sphere.
    center : 3 x 1 array
        Center point of the sphere.
    
    Methods
    -------
    
    get_distances(points)
        Gives the minimum distance between each point to the sphere. 
        
    get_normals(points)
        Gives, for each input point, the normal vector of the point closest 
        to the sphere. 
        
    fit(points, normals=None):
        Gives sphere that fits the input points.
    
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
        
    limit_radius(value):
        Create a list of length `4` that stores `value` at last index and 
        `None` elsewhere.
    
    get_mesh(): TriangleMesh
        Returns mesh defined by the sphere model. 
    """
    
    _fit_n_min = 4
    _model_args_n = 4
    name = 'sphere'
    
    @property
    def equation(self):
        def sig(x):
            return "-" if x < 0 else '+'
        delta = [f'{p} {sig(-a)} {abs(a)}' for p, a in zip(('x','y','z'), 
                                                           self.center)]
        equation = " + ".join([f'({p})**2' for p in delta])
        return equation + f" = {self.radius ** 2}"
    
    @property
    def radius(self):
        """ Radius of the sphere."""
        return self.model[-1]
    
    @property
    def center(self):
        """ Center point of the sphere."""
        return np.array(self.model[:3])
    
    @staticmethod
    def limit_radius(value):
        """ Create a list of length `4` that stores `value` at last index and 
        `None` elsewhere.
        
        Parameters
        ----------
        value : float
            Radius limit value
        
        Returns
        -------
        list
            List containing limit.
        """
        return PrimitiveBase.create_limits(Sphere._model_args_n, 3, value)
    
    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the sphere.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        distances
            Nx1 array distances.
        """
        points = np.asarray(points)
        model = self.model
        return np.linalg.norm(points - model[:3], axis=1) - model[3]
    
    def get_normals(self, points):
        """ Gives, for each input point, the normal vector of the point closest 
        to the sphere.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        normals
            Nx3 array containing normal vectors.
        """
        points = np.asarray(points)
        dist_vec = points - self.model[:3]
        normals = dist_vec / np.linalg.norm(dist_vec, axis=1)[..., np.newaxis]
        return normals

    def get_mesh(self, points):
        """ Returns mesh defined by the sphere model.      
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        mesh = TriangleMesh.create_sphere(radius=self.model[3])
        mesh.translate(self.model[:3])
        return mesh
   
    @staticmethod
    def fit(points, normals=None):
        """ Gives sphere that fits the input points. If the number of points is
        higher than the `4`, the fitted shape will return a least squares 
        estimation.
        
        Parameters
        ----------
        points : 3 x N array
            N input points
        
        Returns
        -------
        Plane
            Fitted sphere.
        """
        # points_ = np.asarray(points)[samples]
        
        num_points = len(points)
        
        if num_points < 4:
            raise ValueError('A minimun of 4 points are needed to fit a '
                             'sphere')
        
        # if simplest case, the result is direct
        elif num_points == 4:
            p0, p1, p2, p3 = points
            
            r1 = p0 - p1
            r2 = p0 - p2
            r3 = p0 - p3
            n0 = p0.dot(p0)
            c12 = np.cross(r1, r2)
            c23 = np.cross(r2, r3)
            c31 = np.cross(r3, r1)
            
            det = r1.dot(c23)
            if det == 0:
                return None
            
            center = 0.5 * (
                (n0 - p1.dot(p1)) * c23 + \
                (n0 - p2.dot(p2)) * c31 + \
                (n0 - p3.dot(p3)) * c12) / det
                
            radiuses = np.linalg.norm(points - center, axis=1)
            radius = sum(radiuses) / num_points
            
        # for more points, find the plane such that the summed squared distance 
        # from the plane to all points is minimized. 
        else:
            
            b = sum(points.T * points.T)
            a = np.c_[2 * points, np.ones(num_points)]
            X = np.linalg.lstsq(a, b, rcond=None)[0]
            
            center = X[:3]
            radius = np.sqrt(X[3] + center.dot(center))
        
        if radius < 0:
            return None

        return Sphere([center[0], center[1], center[2], radius]) 
        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    