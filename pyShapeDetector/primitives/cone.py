#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:57:08 2023

@author: ebernardes
"""
import warnings
import numpy as np
from open3d.geometry import PointCloud, TriangleMesh, AxisAlignedBoundingBox
from open3d.utility import Vector3iVector, Vector3dVector
from skspatial.objects.cylinder import Cylinder as skcylinder

from scipy.spatial.transform import Rotation


from pyShapeDetector.methods import RANSAC_Classic
from .primitivebase import Primitive
from .plane import Plane
from .cylinder import Cylinder
    
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
    from_appex_vector_radius(appex, vector, radius)
        Creates cone from appex, vector and radius as separated arguments.
    
    from_appex_vector_half_angle(appex, vector, half_angle)
        Creates cone from appex, vector and half_angle as 
        separated arguments.
        
    get_signed_distances(points):
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
        
    get_cropped_mesh(points=None, eps=1E-3):
        Creates mesh of the shape and crops it according to points.
        
    is_similar_to(other_shape, rtol=1e-02, atol=1e-02):
        Check if shapes represent same model.
        
    copy():
        Returns copy of shape
    
    align(axis):
        Returns aligned 
    
    """
    
    _fit_n_min = 15
    _model_args_n = 7
    _name = 'cone'
    
    def align(self, axis):
        """ Returns aligned 
        
        Parameters
        ----------
        axis : 3 x 1 array
            Axis to which the shape should be aligned.
        
        Returns
        -------
        Cone
            Aligned cone with axis
        """
        vector = self.height * np.array(axis) / np.linalg.norm(axis)
        return Cone.from_appex_vector_half_angle(
            self.appex, vector, self.half_angle)
    
    @property
    def color(self):
        return np.array([0, 0.707, 0.707])
        
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
        Primitive.__init__(self, model)

        if not (0 <= self.half_angle < np.pi / 2):
            raise ValueError('half_angle must be between 0 and 90 degrees, '
                             f'got {self.half_angle * 180 / np.pi}.')
            
    @classmethod
    def from_appex_top_radius(cls, appex, top, radius):
        """ Creates cone from appex, vector and radius as separated arguments.
        
        Parameters
        ----------            
        appex : 3 x 1 array
            Point at the appex of the cone.
        top : 3 x 1 array
            Point at top of cone.
        radius : float
            Radius of the cone.

        Returns
        -------
        Cone
            Generated shape.
        """
        if radius <= 0:
            raise ValueError('radius must be positive real value.')
        vector = np.array(top) - appex
        return cls.from_appex_vector_radius(appex, vector, radius)
    
    @classmethod
    def from_appex_vector_radius(cls, appex, vector, radius):
        """ Creates cone from appex, vector and radius as separated arguments.
        
        Parameters
        ----------            
        appex : 3 x 1 array
            Point at the appex of the cone.
        vector : 3 x 1 array
            Vector from appex point to top point.
        radius : float
            Radius of the cone.

        Returns
        -------
        Cone
            Generated shape.
        """
        if radius <= 0:
            raise ValueError('radius must be positive real value.')
        half_angle = np.arctan(radius / np.linalg.norm(vector))
        return cls.from_appex_vector_half_angle(appex, vector, half_angle)
    
    @classmethod
    def from_appex_vector_half_angle(cls, appex, vector, half_angle):
        """ Creates cone from appex, vector and half_angle as 
        separated arguments.
        
        Parameters
        ----------            
        appex : 3 x 1 array
            Point at the appex of the cone.
        vector : 3 x 1 array
            Vector from appex point to top point.
        half_angle : float
            Half angle of cone.

        Returns
        -------
        Cone
            Generated shape.
        """
        return cls(list(appex)+list(vector)+[half_angle])
            
            
    @classmethod
    def random(cls, scale=1):
        """ Generates a random cone.
        
        Parameters
        ----------
        scale : float, optional
            scaling factor for random model values.

        Returns
        -------
        Cone
            Random shape.
        """
        model = np.random.random(cls._model_args_n - 1)
        height = np.linalg.norm(model[3:6])
        radius = np.random.random()
        half_angle = np.arctan(radius / height)
        return cls(np.append(model * scale, half_angle))

    @property
    def canonical(self):
        """ Return canonical form for testing."""
        if self.vector[-1] >= 0:
            return self
        
        return Cone(list(self.center) + list(-self.vector) + [self.half_angle])

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
        
        axis_other = -np.cross(self.axis, np.cross(self.axis, delta))
        axis_other /= np.linalg.norm(axis_other, axis=1)[None].T
        C, S = np.cos(self.half_angle), np.sin(self.half_angle)
        return C * self.axis + S * axis_other
    
    def get_angle_diff(self, points):
        alpha = self.get_point_angle(points)
        # the clipping prevents distances to be calculated from the closets
        # point in the symmetric inverted cone
        return np.clip(alpha - self.half_angle, 0, np.pi/2)
    
    def get_point_angle(self, points):
        delta = points - self.appex
        delta_norm = np.linalg.norm(delta, axis=1)
        safe = delta_norm > 1E-8
        projection = delta.dot(self.axis)
        alpha = np.empty(len(points))
        alpha[safe] = np.arccos(projection[safe] / delta_norm[safe])
        alpha[~safe] = 0
        return alpha

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
        delta = points - self.appex
        delta_norm = np.linalg.norm(delta, axis=1)
        angle_diff = self.get_angle_diff(points)
        distances = np.empty(len(points))
        safe = abs(angle_diff) > 1E-7
        distances[safe] = np.sin(angle_diff[safe]) * delta_norm[safe]
        distances[~safe] = delta_norm[~safe] / (2 * np.cos(self.half_angle))
        return distances
    
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
        points = np.asarray(points)
        delta_norm = np.linalg.norm(points - self.appex, axis=1)
        alpha = self.get_point_angle(points)
        
        angle_diff = self.get_angle_diff(points)
        dist_axis = delta_norm * np.cos(angle_diff) / np.cos(self.half_angle)
        p = self.appex + self.axis * dist_axis[None].T
        
        # preventing numerical problems when point is in the axis
        eps = 1E-8
        unsafe1 = abs(alpha) < eps
        unsafe2 = abs(alpha - np.pi) < eps
        safe = np.logical_and(~unsafe1, ~unsafe2)
        normals = np.empty(points.shape)
        
        # points aligned to axis but inside cone, infinite solutions, finc
        # a random one
        axis_perpendicular = np.cross(np.random.random(3), self.axis)
        axis_perpendicular /= np.linalg.norm(axis_perpendicular)
        s, c = np.sin(self.half_angle), np.cos(self.half_angle) 
        normals[unsafe1] = -s * self.axis + c * axis_perpendicular
        
        # normals[unsafe1] = self.axis
        
        # points aligned to axis but behind the cone, normal is not defined
        # at the appex but give a value anyways
        normals[unsafe2] = -self.axis
        
        normals[safe] = points[safe] - p[safe]
        # coef = np.cos(alpha - self.half_angle) * np.linalg.norm(delta, axis=1)
        # normals = delta - closest_axes * coef[None].T
        normals[safe] /= np.linalg.norm(normals[safe], axis=1)[..., np.newaxis]
        return normals
    
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
        points_flattened = Primitive.flatten_points(self, points)
        # delta = points_flattened - self.appex
        # projection = (points_flattened - self.appex).dot(self.axis)
        # points_flattened[projection < 0] = self.appex
        return points_flattened
    
    def get_mesh(self, resolution=30, closed=False):
        """ Returns mesh defined by the cylinder model.
        
        Parameters
        ----------
        resolution : int, optional
            Resolution parameter for mesh. Default: 30
        closed : bool, optional
            If True, does not remove top and bottom of cylinder
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the cone.
        """
        
        mesh = TriangleMesh.create_cone(
            # radius=self.radius, height=self.height, resolution=100, split=100)
            radius=self.radius, height=self.height, resolution=resolution, split=100)
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
            
        if normals is None:
            raise NotImplementedError('Fitting of cone without normals has not'
                                      'been implemented.')
            
        normals = np.asarray(normals)
        if len(normals) != num_points:
            raise ValueError('Different number of points and normals')
        
        # appex is found minimizing distant to tangent planes
        A = normals.T @ normals
        B = normals.T @ (points * normals).sum(1)
        appex =  np.linalg.lstsq(A, B, rcond=None)[0]
        delta = points - appex
        
        # axis can be found by doing a plane fitting using the normals
        # instead of points, see: https://doi.org/10.48550/arXiv.1811.08988
        # pseudo_plane = Plane.fit(normals).normal
        
        # simple fitting does not give good results, though
        detector = RANSAC_Classic()
        detector.add(Plane)
        pseudo_plane = detector.fit(normals)[0]
        
        if pseudo_plane is None:
            # print('a')
            return None
        
        axis = pseudo_plane.normal
        
        # correct axis direction
        s = np.sign(delta.dot(axis).sum())
        axis *= s
        
        # detector.add(Cylinder)
        # pseudo_pcd = PointCloud(Vector3dVector(normals))
        # pseudo_pcd.estimate_normals()
        # # pseudo_cylinder = Cylinder.fit(pseudo_pcd.points, pseudo_pcd.normals)
        # pseudo_cylinder = detector.fit(pseudo_pcd.points, pseudo_pcd.normals)[0]
        # axis = pseudo_cylinder.axis
        
        # axis = -np.linalg.svd(delta)[-1][0]
        # axis = np.linalg.svd(delta)[-1][0]
        
        projection = points.dot(axis)
        # height = max(projection) - min(projection)
        height = max(projection) - axis.dot(appex)
        vector = axis * height
        
        norm = np.linalg.norm(delta, axis=1)
        cossines_half_angle = delta.dot(axis) / norm
        half_angle = np.arccos(cossines_half_angle.mean())
        # S = (norm ** 2) * np.sin(2 * cossines_half_angle)
        # C = (norm ** 2) * np.cos(2 * cossines_half_angle)
        # half_angle = np.arctan2(sum(S), sum(C)) / 2
        
        if abs(half_angle - np.pi / 2) < 1E-10:
            return None
        
        if not (0 <= half_angle < np.pi / 2):
            half_angle = np.pi - half_angle
            vector = -vector
            # print(half_angle * 180 / np.pi)
            # return None
        
        # return Cone(center+vector+[radius]) 
        # return Cone(appex+vector+[half_angle]) 
        return Cone.from_appex_vector_half_angle(appex, vector, half_angle)
