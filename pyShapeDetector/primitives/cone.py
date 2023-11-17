#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:57:08 2023

@author: ebernardes
"""
import warnings
import numpy as np
from open3d.geometry import TriangleMesh, AxisAlignedBoundingBox
from open3d.utility import Vector3iVector
from skspatial.objects.cylinder import Cylinder as skcylinder

from .primitivebase import Primitive
    
class Cone(Primitive):
    """
    Cone primitive.
    
    Attributes
    ----------
    fit_n_min : int
        Minimum number of points necessary to fit a model.
    model_args_n : str
        Number of parameters in the model.
    name : str
        Name of primitive.
    appex : 3 x 1 array
        Point at the appex of the cone.
    vector : 3 x 1 array
        Vector from appex point to top point.
    height: float
        Height of cone.
    axis : 3 x 1 float
        Unit vector defining axis of cone.
    radius : float
        Radius of the cone.
    half_angle : float
        Half angle of cone.
    center : 3 x 1 array
        Center point of the cone.
    rotation_from_axis : 3 x 3 array
        Rotation matrix that aligns z-axis with cone axis.
    canonical : Cylinder
        Return cone form for testing.
    surface_area : float
        Surface area of cone
    volume : float
        Volume of cone.
        
    Methods
    ------- 
    
    def get_signed_distances(points):
        Gives the minimum distance between each point to the model. 
    
    get_distances(points)
        Gives the minimum distance between each point to the cylinder. 
        
    get_normals(points)
        Gives, for each input point, the normal vector of the point closest 
        to the cylinder. 
        
    random(scale):
        Generates a random shape.
        
    fit(points, normals=None):
        Gives cylinder that fits the input points. 
    
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
        
    get_orthogonal_component(points):
        Removes the axis-aligned components of points.
        
    closest_to_line(points):
        Returns points in cylinder axis that are the closest to the input
        points.
    
    get_mesh(): TriangleMesh
        Returns mesh defined by the cylinder model. 
    
    """
    
    _fit_n_min = 7
    _model_args_n = 8
    _name = 'cone'
    
    @property
    def color(self):
        return np.array([0.707, 0, 0.707])

    @property
    def canonical(self):
        """ Return canonical form for testing."""
        if self.vector[-1] >= 0:
            return self
        
        return Cone(list(self.center) + list(-self.vector) + [self.radius] + [self.half_angle])

    @property
    def equation(self):
        def sig(x):
            return "-" if x < 0 else '+'
        delta = [f'{p} {sig(-a)} {abs(a)}' for p, a in zip(('x','y','z'), 
                                                           self.center)]
        A = " + ".join([f'({p})**2' for p in delta])
        B = " + ".join([ f'{e} * ({p})' for e, p in zip (self.axis, delta)])
        return A + " + [" + B + "]**2" + f" = {self.radius ** 2}"
    
    @property
    def appex(self):
        """ Point at the base of the cylinder. """
        # return np.array(self.model[:3])
        return self.center + self.vector / 2
    
    @property
    def vector(self):
        """ Vector from base point to top point. """
        return np.array(self.model[3:6])
    
    @property
    def height(self):
        """ Height of cone. """
        return np.linalg.norm(self.vector)
    
    @property
    def axis(self):
        """ Unit vector defining axis of cone. """
        return self.vector / self.height
    
    @property
    def radius(self):
        """ Radius of the cone. """
        return self.model[6]
    
    @property
    def half_angle(self):
        """ Half angle of cone. """
        return self.model[7]
    
    @property
    def center(self):
        """ Center point of the cone."""
        # return self.appex + self.vector / 2
        return np.array(self.model[:3])
    
    @property
    def surface_area(self):
        """ Surface area of primitive """
        return 2 * np.pi * self.radius * self.height
    
    @property
    def volume(self):
        """ Volume of primitive. """
        return np.pi * (self.radius ** 2) * self.height
    
    @property
    def rotation_from_axis(self):
        """ Rotation matrix that aligns z-axis with cylinder axis."""
        return self.get_rotation_from_axis(self.axis)
    
    def closest_to_line(self, points):
        """ Returns points in cylinder axis that are the closest to the input
        points.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        
        Returns
        -------
        points_closest: N x 3 array
            N points in cylinder line
        """
        points = np.asarray(points)
        projection = (points - self.appex).dot(self.axis)
        return self.appex + projection[..., np.newaxis] * self.axis
    
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
        delta = points - self.appex
        return -np.cross(self.axis, np.cross(self.axis, delta))

    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the cylinder. 
        
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
            np.cross(self.axis, points - self.appex), axis=1)
        
        return distances - self.radius
    
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
    
    def get_mesh(self, closed=False):
        """ Returns mesh defined by the cylinder model.
        
        Parameters
        ----------
        closed : bool, optional
            If True, does not remove top and bottom of cylinder
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        
        mesh = TriangleMesh.create_cylinder(
            radius=self.radius, height=self.height, resolution=100, split=100)
        
        # first and second points are the central points defining top / base
        triangles = np.asarray(mesh.triangles)
        
        if not closed:
            triangles = np.array(
                [t for t in triangles if 0 not in t and 1 not in t])
            triangles = np.vstack([triangles, triangles[:, ::-1]])
            mesh.triangles = Vector3iVector(triangles)
        
        mesh.rotate(self.rotation_from_axis)
        mesh.translate(self.center)
        
        return mesh
    
    @staticmethod
    def fit(points, normals=None):
        """ Gives cylinder that fits the input points. 
        
        If normals are given: first calculate cylinder axis using normals as
        explained in [1] and then use least squares to calculate center point
        and radius.
        
        If normals are not given, uses Scikit Spatial, which is slower and not
        recommended.
        
        References
        ----------
        [1]: http://dx.doi.org/10.1016/j.cag.2014.09.027
        
        Parameters
        ----------
        points : N x 3 array
            N input points
        
        Returns
        -------
        Plane
            Fitted sphere.
        """
        points = np.asarray(points)
        
        num_points = len(points)
        
        if num_points < 7:
            raise ValueError('A minimun of 6 points are needed to fit a '
                             'cone')
            
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
            # Was revealed to me in a dream, again
            axis_neg_squared_skew = np.eye(3) - axis[np.newaxis].T * axis
            points_skew = (axis_neg_squared_skew @ points.T).T
            b = sum(points_skew.T * points_skew.T)
            a = np.c_[2 * points_skew, np.ones(num_points)]
            X = np.linalg.lstsq(a, b, rcond=None)[0]
            
            point = X[:3]
            radius = np.sqrt(X[3] + point.dot(axis_neg_squared_skew @ point))
            
            # find point in base of cylinder
            proj = points.dot(axis)
            idx = np.where(proj == max(proj))[0][0]
            
            
            # point = list(point)
            height = max(proj) - min(proj)
            vector = axis * height
            center = -np.cross(axis, np.cross(axis, point)) + np.median(proj) * axis     
            # base = center - vector / 2
            
            # base = list(base)
            center = list(center)
            vector = list(vector)
        
        else:
            # if no normals, use scikit spatial, slower
            warnings.warn('Cylinder fitting works much quicker if normals '
                          'are given.')
            solution = skcylinder.best_fit(points)
            
            # base = list(solution.point)
            center = list(solution.point + solution.vector/2)
            vector = list(solution.vector)
            radius = solution.radius
        
        return Cone(center+vector+[radius]) 
