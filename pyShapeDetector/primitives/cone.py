#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:57:08 2023

@author: ebernardes
"""
import warnings
import numpy as np
from open3d.geometry import TriangleMesh, AxisAlignedBoundingBox
from open3d.utility import Vector3iVector, Vector3dVector
from skspatial.objects.cylinder import Cylinder as skcylinder

from scipy.spatial.transform import Rotation

from .primitivebase import Primitive
from .plane import Plane
    
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
        
    get_closest_axes(self, points):
        Get the axes that define the closest lines in the cone's surface
        for each point.
    
    closest_to_line(points):
        Returns points in cylinder axis that are the closest to the input
        points.
    
    get_mesh(): TriangleMesh
        Returns mesh defined by the cylinder model. 
    
    """
    
    _fit_n_min = 6
    _model_args_n = 7
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
        """ Cone appex. """
        return np.array(self.model[:3])
        # return self.center + self.vector / 2

    @property
    def top(self):
        """ Central point at top base. """
        return self.appex + self.vector
    
    @property
    def center(self):
        """ Center point of the cylinder."""
        return self.appex + self.vector / 2
        # return np.array(self.model[:3])
    
    @property
    def vector(self):
        """ Vector from appex point to top point. """
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
        return self.height * np.tan(self.half_angle)
    
    @property
    def half_angle(self):
        """ Half angle of cone. """
        return self.model[6]
    
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
        return self.get_rotation_from_axis([0, 0, 1], self.axis)
    
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
    
    def get_closest_axes(self, points):
        """ Get the axes that define the closest lines in the cone's surface
        for each point.
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        
        Returns
        -------
        axes: N x 3 array
            N closest axes
        """
        points = np.asarray(points)
        delta = points - self.appex
        
        normalized = lambda x: x / np.linalg.norm(x)
        
        rot = Rotation.from_rotvec(
            normalized(np.cross(self.axis, delta)) * self.half_angle)
        return rot.apply(self.axis)
    
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
        closest_axes = self.get_closest_axes(points)
        
        return -np.cross(closest_axes, np.cross(closest_axes, delta))
    
    def get_point_angle(self, points):
        delta = points - self.appex
        projection = delta.dot(self.axis)
        return np.arccos(projection / np.linalg.norm(delta, axis=1))

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
        # cone_normals = self.get_normals(points)
        # # return cone_normals.dot(points - self.appex)
        # # return np.sum(cone_normals * (points - self.appex), axis=1)
        # delta = points - self.appex
        # # return np.linalg.norm(np.cross(cone_normals, delta))
        # return sum(cone_normals.T * delta.T)
        delta = points - self.appex
        # axis = self.axis
        # return np.linalg.norm(delta - self.axis * delta.dot(self.axis)[None].T, axis=1)
        # axis = self.get_closest_axes(points)
        # return np.linalg.norm(delta - axis * sum(delta.T * axis.T)[None].T, axis=1)
        alpha = self.get_point_angle(points)
        return np.sin(alpha - self.half_angle) * np.linalg.norm(delta, axis=1)
    
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
            Mesh corresponding to the cone.
        """
        
        mesh = TriangleMesh.create_cone(
            radius=self.radius, height=self.height, resolution=100, split=100)
        mesh.vertices = Vector3dVector(-np.asarray(mesh.vertices))
        # mesh.translate(-mesh.get_center())
        
        mesh.rotate(self.rotation_from_axis)
        mesh.translate(self.center - mesh.get_center())
        
        if not closed:
            
            triangles = np.asarray(mesh.triangles)
            
            plane = Plane(list(self.axis)+[-np.dot(self.top, self.axis)])
            dist = plane.get_distances(mesh.vertices)
            base_vertices = np.where(abs(dist - min(dist)) < 1e-4)[0]
            triangles = np.asarray(mesh.triangles)
            triangles = np.array(
                [t for t in triangles if not np.any(np.isin(base_vertices, t))])
            triangles = np.vstack([triangles, triangles[:, ::-1]])
            mesh.triangles = Vector3iVector(triangles)
        # center = mesh.get_center()
        # mesh.translate()
        
        # mesh.translate(self.center)
        
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
        
        if num_points < 6:
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
            a = np.c_[2 * points_skew, 
                      np.ones(num_points),
                      sum(points.T * points.T),
                      points]
            
            X = np.linalg.lstsq(a, b, rcond=None)[0]
            
            point = X[:3]
            half_angle = np.arcsin( np.sqrt(X[5]) )
            # radius = np.sqrt(X[3] + point.dot(axis_neg_squared_skew @ point))
            
            # find point in base of cylinder
            proj = points.dot(axis)
            idx = np.where(proj == max(proj))[0][0]
            
            
            # point = list(point)
            height = max(proj) - min(proj)
            vector = axis * height
            center = -np.cross(axis, np.cross(axis, point)) + np.median(proj) * axis     
            appex = center - vector / 2
            
            appex = list(appex)
            # center = list(center)
            vector = list(vector)
        
        else:
            raise NotImplementedError('Fitting of cone without normals has not'
                                      'been implemented.')

        
        # return Cone(center+vector+[radius]) 
        return Cone(appex+vector+[half_angle]) 
