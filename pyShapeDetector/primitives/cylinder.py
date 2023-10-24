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

from .primitivebase import PrimitiveBase
    
class Cylinder(PrimitiveBase):
    """
    Cylinder primitive.
    
    Attributes
    ----------
    _fit_n_min : int
        Minimum number of points necessary to fit a model.
    _model_args_n : str
        Number of parameters in the model.
    name : str
        Name of primitive.
    base : 3 x 1 array
        Point at the base of the cylinder.
    vector : 3 x 1 array
        Vector from base point to top point.
    height: float
        Height of cylinder.
    axis : 3 x 1 float
        Unit vector defining axis of cylinder.
    radius : float
        Radius of the cylinder.
    center : 3 x 1 array
        Center point of the cylinder.
    rotation_from_axis : 3 x 3 array
        Rotation matrix that aligns z-axis with cylinder axis.
        
    Methods
    ------- 
    
    get_distances(points)
        Gives the minimum distance between each point to the cylinder. 
        
    get_normals(points)
        Gives, for each input point, the normal vector of the point closest 
        to the cylinder. 
        
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
    
    _fit_n_min = 6
    _model_args_n = 7
    name = 'cylinder'

    # def __init__(self, model):
    #     """
    #     Parameters
    #     ----------
    #     model : list or tuple
    #         Parameters defining the shape model
                        
    #     Raises
    #     ------
    #     ValueError
    #         If number of parameters is incompatible with the model of the 
    #         primitive.
    #     """
    #     model = np.array(model)
    #     if model[5] < 0:
    #         model[3:6] = -model[3:6] # define only one acceptable normal axis
    #     PrimitiveBase.__init__(self, model)

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
    def base(self):
        """ Point at the base of the cylinder. """
        return np.array(self.model[:3])
        # return self.center + self.vector / 2
    
    @property
    def vector(self):
        """ Vector from base point to top point. """
        return np.array(self.model[3:6])
    
    @property
    def height(self):
        """ Height of cylinder. """
        return np.linalg.norm(self.vector)
    
    @property
    def axis(self):
        """ Unit vector defining axis of cylinder. """
        return self.vector / self.height
    
    @property
    def radius(self):
        """ Radius of the cylinder. """
        return self.model[6]
    
    @property
    def center(self):
        """ Center point of the cylinder."""
        return self.base + self.vector / 2
        # return np.array(self.model[:3])
    
    @property
    def rotation_from_axis(self):
        """ Rotation matrix that aligns z-axis with cylinder axis."""
        return self.get_rotation_from_axis(self.axis)
    
    def closest_to_line(self, points):
        """ Returns points in cylinder axis that are the closest to the input
        points.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        points_closest: 3 x N array
            N points in cylinder line
        """
        points = np.asarray(points)
        projection = (points - self.base).dot(self.axis)
        return self.base + projection[..., np.newaxis] * self.axis
    
    def get_orthogonal_component(self, points):
        """ Removes the axis-aligned components of points.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        points_orthogonal: 3 x N array
            N points
        """
        points = np.asarray(points)
        delta = points - self.base
        return -np.cross(self.axis, np.cross(self.axis, delta))

    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the cylinder. 
        
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
        distances = np.linalg.norm(
            np.cross(self.axis, points - self.base), axis=1)
        
        return distances - self.radius
    
    def get_normals(self, points):
        """ Gives, for each input point, the normal vector of the point closest 
        to the cylinder.
        
        Parameters
        ----------
        points : 3 x N array
            N input points 
        
        Returns
        -------
        normals
            Nx3 array containing normal vectors.
        """
        normals = self.get_orthogonal_component(points)
        normals /= np.linalg.norm(normals, axis=1)[..., np.newaxis]
        return normals
    
    def get_mesh(self, points=None):
        """ Returns mesh defined by the cylinder model.      
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        
        points = np.asarray(points)
        eps = 1e-7
        # eps = 0
        
        mesh = TriangleMesh.create_cylinder(
            radius=self.radius, height=self.height+2*eps)
        
        bb = mesh.get_axis_aligned_bounding_box()
        bb = AxisAlignedBoundingBox(bb.min_bound + [0, 0, eps],
                                    bb.max_bound - [0, 0, eps])
        mesh = mesh.crop(bb)
        triangles = np.asarray(mesh.triangles)
        mesh.triangles = Vector3iVector(
            np.vstack([triangles, triangles[:, ::-1]]))

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
        points : 3 x N array
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
                             'cylinder')
            
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
            axis = axis / np.linalg.norm(axis)
            
            # Reference for the rest:
            # Was revealed to me in a dream
            axis_neg_squared_skew = np.eye(3) - axis[np.newaxis].T * axis
            points_skew = (axis_neg_squared_skew @ points.T).T
            b = sum(points_skew.T * points_skew.T)
            a = np.c_[2 * points_skew, np.ones(num_points)]
            X = np.linalg.lstsq(a, b, rcond=None)[0]
            
            center = X[:3]
            radius = np.sqrt(X[3] + X[:3].dot(axis_neg_squared_skew @ center))
            
            # find point in base of cylinder
            proj = points.dot(axis)

            idx = np.where(proj == min(proj))[0][0]
            
            height = 2 * abs(max(proj) - min(proj))
            
            # No idea why:
            center = X[:3] + (points[idx].dot(axis) + height / 4) * axis
            center = list(center)
            base = list(center - axis * height / 2)
            
            # axis = axis / np.linalg.norm(axis)
            vector = list(axis * height)
            # axis = list(axis)
        
        else:
            # if no normals, use scikit spatial, slower
            warnings.warn('Cylinder fitting works much quicker if normals '
                          'are given.')
            solution = skcylinder.best_fit(points)
            
            base = list(solution.point)
            # height = np.linalg.norm(solution.vector)
            # axis = list(solution.vector / height)
            vector = list(solution.vector)
            radius = solution.radius
        
        # return Cylinder(center+vector+[radius]) 
        return Cylinder(base+vector+[radius]) 
