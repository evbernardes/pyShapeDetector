#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
import numpy as np
from open3d.geometry import TriangleMesh, AxisAlignedBoundingBox

from .primitivebase import Primitive
    
class Sphere(Primitive):
    """
    Sphere primitive.
    
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
    mesh
    inlier_points
    inlier_points_flattened
    inlier_normals
    inlier_colors
    inlier_PointCloud
    inlier_PointCloud_flattened
    metrics
    axis_spherical
    axis_cylindrical
    bbox
    bbox_bounds
    inlier_bbox
    inlier_bbox_bounds
    
    radius
    center
        
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
    set_inliers
    add_inliers
    closest_inliers
    inliers_average_dist
    get_inliers_axis_aligned_bounding_box
    get_axis_aligned_bounding_box
    sample_points_uniformly
    sample_points_density
    sample_PointCloud_uniformly
    sample_PointCloud_density
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
    
    from_center_radius
    """
    
    _fit_n_min = 4
    _model_args_n = 4
    _name = 'sphere'
    _translatable = [0, 1, 2]
    _rotatable = []
    _color = np.array([0, 1, 0])
    
    @property
    def equation(self):
        def sig(x):
            return "-" if x < 0 else '+'
        delta = [f'{p} {sig(-a)} {abs(a)}' for p, a in zip(('x','y','z'), 
                                                           self.center)]
        equation = " + ".join([f'({p})**2' for p in delta])
        return equation + f" = {self.radius ** 2}"
    
    @property
    def surface_area(self):
        """ Surface area of primitive """
        return 4 * np.pi * (self.radius ** 2)
    
    @property
    def volume(self):
        """ Volume of primitive. """
        return (4/3) * np.pi * (self.radius ** 3)
    
    @property
    def radius(self):
        """ Radius of the sphere."""
        return self.model[-1]
    
    @property
    def center(self):
        """ Center point of the sphere."""
        return np.array(self.model[:3])
        
    @staticmethod
    def fit(points, normals=None):
        """ Gives sphere that fits the input points. If the number of points is
        higher than the `4`, the fitted shape will return a least squares 
        estimation.
        
        Parameters
        ----------
        points : N x 3 array
            N input points
        
        Returns
        -------
        Plane
            Fitted sphere.
        """
        # points_ = np.asarray(points)[samples]
        points = np.asarray(points)
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
    
    def get_signed_distances(self, points):
        """ Gives the minimum distance between each point to the sphere.
        
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
        model = self.model
        return np.linalg.norm(points - model[:3], axis=1) - model[3]
    
    def get_normals(self, points):
        """ Gives, for each input point, the normal vector of the point closest 
        to the sphere.
        
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
        dist_vec = points - self.model[:3]
        normals = dist_vec / np.linalg.norm(dist_vec, axis=1)[..., np.newaxis]
        return normals 

    def get_axis_aligned_bounding_box(self, slack=0):
        """ Returns an axis-aligned bounding box of the primitive.
        
        Parameters
        ----------
        slack : float, optional
            Expand bounding box in all directions, useful for testing purposes.
            Default: 0.
        
        See: open3d.geometry.get_axis_aligned_bounding_box
            
        Returns
        -------
        open3d.geometry.AxisAlignedBoundingBox
        """
        if slack < 0:
            raise ValueError("Slack must be non-negative.")
            
        return AxisAlignedBoundingBox(self.center - self.radius - slack, 
                                      self.center + self.radius + slack)

    def get_mesh(self, **options):
        """ Returns mesh defined by the sphere model.   
        
        Parameters
        ----------
        resolution : int, optional
            Resolution parameter for mesh. Default: 30   
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        resolution = options.get('resolution', 30)
        
        mesh = TriangleMesh.create_sphere(radius=self.model[3], resolution=resolution)
        mesh.translate(self.model[:3])
        
        return mesh
    
    @classmethod
    def from_center_radius(cls, center, radius):
        """ Creates sphere from center and radius as separated arguments.
        
        Parameters
        ----------            
        center : 3 x 1 array
            Center point.
        radius : float
            Radius of the sphere.

        Returns
        -------
        Cone
            Generated shape.
        """
        return cls(list(center)+[radius])
    