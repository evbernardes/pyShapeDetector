#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
from warnings import warn
from itertools import permutations, product
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation
from open3d.geometry import TriangleMesh
from open3d.utility import Vector3iVector, Vector3dVector

from pyShapeDetector.utility import get_rotation_from_axis
from .primitivebase import Primitive
from alphashape import alphashape
# from .line import Line
    
class Plane(Primitive):
    """
    Plane primitive.
    
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
    normal : 3 x 1 array
        Normal vector defining plane.
    dist : float
        Distance to origin.
    centroid : 3 x 1 array
        A point in the plane.
    canonical : Plane
        Return canonical form for testing.
    surface_area : float
        For unbounded plane, returns NaN and gives warning
    volume : float
        Volume of plane, which is zero.
    holes : list
        List of holes in plane.
    bound_lines : list of Line instances
        Lines defining bounds.
    
    Methods
    ------- 
    inliers_bounding_box(slack=0):
        If the shape includes inlier points, returns the minimum and 
        maximum bounds of their bounding box.
        
    from_normal_dist(normal, dist):
        Creates plane from normal vector and distance to origin.
    
    from_normal_point(normal, point):
        Creates plane from normal vector and point in plane.
        
    get_bounded_plane(points):
        Gives bounded version of plane, using input points to define 
        border.
        
    get_square_plane(length=1):
        Gives four points defining boundary of square plane.
        
    get_square_mesh(length=1):
        Gives square plane defined by four points.
        
    add_holes(holes):
        Add one or more holes to plane.
        
    remove_hole(idx):
        Remove hole according to index.
    
    get_signed_distances(points):
        Gives the minimum distance between each point to the model. 
    
    get_distances(points)
        Gives the minimum distance between each point to the model.
        
    get_normals(points)
        Gives, for each input point, the normal vector of the point closest 
        to the primitive. 
        
    get_projections(points):
        Get 2D projections of points in plane.
        
    random(scale):
        Generates a random shape.
        
    fit(points, normals=None):
        Gives plane that fits the input points.
    
    get_angles_cos(self, points, normals):
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
        
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
    
    get_mesh(): TriangleMesh
        Flatten points and creates a simplified mesh of the plane. If the
        shape has pre-defined inlier points, use them to find borders.
        Otherwise, return square mesh.
        
    get_cropped_mesh(points=None, eps=1E-3):
        Creates mesh of the shape and crops it according to points.
        
    is_similar_to(other_shape, rtol=1e-02, atol=1e-02):
        Check if shapes represent same model.
        
    copy(copy_holes):
        Returns copy of plane
    
    align(axis):
        Returns aligned 
    
    """
    
    _fit_n_min = 3
    _model_args_n = 4
    _name = 'plane'
    _holes = []
    _rotatable = [0, 1, 2]
    _fusion_intersections = []
    
    def translate(self, translation):
        """ Translate the shape.
        
        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        """
        centroid = self.centroid + translation
        self._model = Plane.from_normal_point(
            self.normal, centroid).model
        
        self._translate_points(translation)
        for hole in self.holes:
            hole._translate_points(translation)
    
    def rotate(self, rotation):
        """ Rotate the shape.
        
        Parameters
        ----------
        rotation : 3 x 3 rotation matrix or scipy.spatial.transform.Rotation
            Rotation matrix.
        """
        if not hasattr(self, '_rotatable'):
            raise NotImplementedError('Shapes of type {shape.name} do not '
                                      'have an implemented _rotatable '
                                      'attribute')
            
        rotation = Primitive._parse_rotation(rotation)
            
        centroid = rotation.apply(self.centroid)
        normal = rotation.apply(self.normal)
        self._model = Plane.from_normal_point(normal, centroid).model
        self._rotate_points_normals(rotation)
        
    def copy(self, copy_holes=True):
        """ Returns copy of plane
        
        Parameters
        ----------
        copy_holes: boolean, optional
            If True, also copy holes. Default: True.
        
        Returns
        -------
        Primitive
            Copied primitive
        """
        shape = Plane(self.model.copy())
        shape._inlier_points = self._inlier_points.copy()
        shape._inlier_normals = self._inlier_normals.copy()
        shape._inlier_colors = self._inlier_colors.copy()
        shape._fusion_intersections = self._fusion_intersections.copy()
        shape._metrics = self._metrics.copy()
        if copy_holes:
            holes = [h.copy(copy_holes=False) for h in self._holes]
            shape._holes = holes
        return shape
    
    def add_holes(self, holes, remove_points=True):
        """ Add one or more holes to plane.
        
        Parameters
        ----------            
        holes : PlaneBounded or list of PlaneBounded instances
            Planes defining holes.
        remove_points : boolean, optional
            If True, removes points of plane inside holes and points of holes
            outside of plane. Default: True.
        """
        if remove_points and not isinstance(self, PlaneBounded):
            warn('Option remove_points only works if plane is bounded.')
            remove_points = False
        
        if not isinstance(holes, list):
            holes = [holes]
        
        fixed_holes = []
        for hole in holes:
            if not isinstance(hole, PlaneBounded):
                raise ValueError("Holes must be instances of PlaneBounded, got"
                                 f" {hole}.")
                
            cos_theta = np.dot(hole.normal, self.normal)
            if abs(cos_theta) < 1 - 1e-5:
                raise ValueError("Plane and hole must be aligned.")
            
            model = hole.model
            if cos_theta < 0:
                model = -model
            
            if remove_points:
                inside1 = self.contains_projections(hole.bounds)
                if sum(inside1) < 3:
                    print('shape does not contain hole')
                    continue
                
                # if sum(inside1) < 1:
                    # print('shape does not contain hole')
                    # continue
                
                inside2 = hole.contains_projections(self.bounds)
                # print(inside2)
                hole = PlaneBounded(model, hole.bounds[inside1])
                self._bounds = self._bounds[~inside2]
                self._bounds_projections = self._bounds_projections[~inside2]
            else: 
                hole = PlaneBounded(model, hole.bounds)
            fixed_holes.append(hole)
        self._holes += fixed_holes
    
    def remove_hole(self, idx):
        """ Remove hole according to index.
        
        Parameters
        ----------            
        idx : int
            Index of hole to be removed.
        """
        self._holes.pop(idx)
    
    @property
    def holes(self):
        """ Existing holes in plane. """
        return self._holes
    
    @property
    def bound_lines(self):
        """ Lines defining bounds. """
        from .line import Line
        
        bounds = self.bounds
        num_lines = len(bounds)
        lines = [Line.from_two_points(bounds[i-1], bounds[i]) for i in range(num_lines)]
        return lines
    
    def bound_lines_meshes(self, radius_ratio=0.001, color=(0, 0, 0)):
        lines = self.bound_lines
        meshes =  [line.get_mesh(radius_ratio=radius_ratio) for line in lines]
        [mesh.paint_uniform_color(color) for mesh in meshes]
        return meshes
    
    @classmethod
    def from_normal_dist(cls, normal, dist):
        """ Creates plane from normal vector and distance to origin.
        
        Parameters
        ----------            
        normal : 3 x 1 array
            Normal vector defining plane.
        radius : float
            Distance to origin.

        Returns
        -------
        Cone
            Generated shape.
        """
        return cls(list(normal)+[dist])
    
    @classmethod
    def from_normal_point(cls, normal, point):
        """ Creates plane from normal vector and point in plane.
        
        Parameters
        ----------            
        normal : 3 x 1 array
            Normal vector defining plane.
        point : 3 x 1 array
            Point in plane.

        Returns Creates plane from normal vector and point in plane.
        -------
        Cone
            Generated shape.
        """
        return cls.from_normal_dist(normal, -np.dot(normal, point))
    
    @property
    def color(self):
        return np.array([0, 0, 1])
    
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
        model = np.array(model)
        norm = np.linalg.norm(model[:3])
        model = model / norm
        Primitive.__init__(self, model)
        self._holes = []
        
    def get_bounded_plane(self, points=None):
        """ Gives bounded version of plane, using input points to define 
        border.
        
        Parameters
        ----------
        points : array_like, shape (N, 3)
            Parameters defining the shape model
                        
        Returns
        -------
        PlaneBounded
            Bounded version of plane.
        """
        if points is None:
            points = self.inlier_points
        if len(points) == 0:
            raise ValueError('if no points are given, shape must have inlier points')
        bounds = self.flatten_points(points)
        bounded_plane = PlaneBounded(self, bounds)
        bounded_plane._holes = self._holes
        return bounded_plane
    
    @property
    def canonical(self):
        """ Return canonical form for testing. """
        model = self.model
        if np.sign(self.dist) < 0:
            model = -model
        plane = Plane(list(self.model))
        plane._holes = self._holes
        return plane
    
    @property
    def equation(self):
        n = self.normal
        d = self.dist
        equation = ''
        equation += f'{n[0]} * x '
        equation += '-' if n[1] < 0 else '+'
        equation += f' {abs(n[1])} * y '
        equation += '-' if n[2] < 0 else '+'
        equation += f' {abs(n[2])} * z '
        equation += '-' if d < 0 else '+'
        equation += f' {abs(d)} = 0'
        
        return equation
    
    @property
    def normal(self):
        """ Normal vector defining point. """
        return np.array(self.model[:3])
    
    @property
    def dist(self):
        """ Distance to origin. """
        return self.model[3]
    
    @property
    def centroid(self):
        """ A point in the plane. """
        return -self.normal * self.dist
    
    @property
    def surface_area(self):
        """ For unbounded plane, returns NaN and gives warning """
        warn('For unbounded planes, the surface area is undefined')
        return float('nan')
    
    @property
    def volume(self):
        """ Volume of plane, which is zero. """
        return 0    
    
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
        points = np.asarray(points)
        return points.dot(self.normal) + self.dist
    
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
        return np.tile(self.normal, (len(points), 1))
    
    def get_projections(self, points):
        """ Get 2D projections of points in plane.

        Parameters
        ----------
        points : array_like, shape (N, 3)
            Points corresponding to the fitted shape.
        
        Returns
        -------
        projections : array_like, shape (N, 2)
            2D projections of boundary points in plane
        """
        points = np.asarray(points)
        rot = get_rotation_from_axis([0, 0, 1], self.normal)
        return (rot @ points.T).T[:, :2]
    
    def get_mesh(self, resolution=1):
        """ Flatten points and creates a simplified mesh of the plane. If the
        shape has pre-defined inlier points, use them to find borders.
        Otherwise, return square mesh.
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        if len(self.inlier_points) == 0:
            warn('No inlier points, returning square plane...')
            return self.get_square_mesh()
        bounds = self.flatten_points(self.inlier_points)
        
        bounded_plane = PlaneBounded(self, bounds)
        bounded_plane._holes = self._holes
        mesh = bounded_plane.get_mesh()
        
        if len(self.inlier_colors) > 0:
            mesh.paint_uniform_color(np.median(self.inlier_colors, axis=0))
        return mesh
    
    def get_square_plane(self, length=1):
        """ Gives square plane defined by four points.

        Parameters
        ----------
        length : float, optional
            Length of the sides of the square
        
        Returns
        -------
        Plane
            Square plane
        """
        # center = self.centroid
        # center = np.mean(np.asarray(pcd.points), axis=0)
        # bb = pcd.get_axis_aligned_bounding_box()
        # half_length = max(bb.max_bound - bb.min_bound) / 2
        
        normalized = lambda x : x / np.linalg.norm(x)
        
        if np.isclose(self.normal[1], 1, atol=1e-7): 
            v1 = normalized(np.cross(self.normal, [0, 0, 1]))
            v2 = normalized(np.cross(v1, self.normal))
        else:
            v1 = normalized(np.cross([0, 1, 0], self.normal))
            v2 = normalized(np.cross(self.normal, v1))

        centroid = self.centroid
        vertices = np.vstack([
            centroid + (+ v1 + v2) * length / 2,
            centroid + (+ v1 - v2) * length / 2,
            centroid + (- v1 + v2) * length / 2,
            centroid + (- v1 - v2) * length / 2])
        
        plane_bounded = PlaneBounded(self, vertices)
        plane_bounded._holes = self._holes
        return plane_bounded

        # triangles = Vector3iVector(np.array([
        #     [0, 1, 2], 
        #     [2, 1, 0],
        #     [0, 2, 3],
        #     [3, 2, 0]]))
        # vertices = Vector3dVector(vertices)

        # return TriangleMesh(vertices, triangles)
    
    def get_square_mesh(self, length=1):
        """ Gives a square mesh that fits the plane model.   

        Parameters
        ----------
        length : float, optional
            Length of the sides of the square
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        return self.get_square_plane(length).get_mesh()
    
    @staticmethod
    def fit(points, normals=None):
        """ Gives plane that fits the input points. If the number of points is
        higher than the `3`, the fitted shape will return a least squares 
        estimation.
        
        Reference:
            https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        normals : N x 3 array
            N normal vectors
        
        Returns
        -------
        Plane
            Fitted plane.
        """
        points = np.asarray(points)        
        num_points = len(points)
        
        if num_points < 3:
            raise ValueError('A minimun of 3 points are needed to fit a '
                             'plane')
        
        # if simplest case, the result is direct
        elif num_points == 3:
            p0, p1, p2 = points
            
            e1 = p1 - p0
            e2 = p2 - p0
            abc = np.cross(e1, e2)
            centroid = p0
        
        # for more points, find the plane such that the summed squared distance 
        # from the plane to all points is minimized.
        else:
            centroid = sum(points) / num_points
            x, y, z = np.asarray(points - centroid).T
            xx = x.dot(x)
            yy = y.dot(y)
            zz = z.dot(z)
            xy = x.dot(y)
            yz = y.dot(z)
            xz = z.dot(x)
                
            det_x = yy * zz - yz * yz
            det_y = xx * zz - xz * xz
            det_z = xx * yy - xy * xy
            
            if (det_x > det_y and det_x > det_z):
                abc = np.array([det_x, 
                                xz * yz - xy * zz, 
                                xy * yz - xz * yy])
            elif (det_y > det_z):
                abc = np.array([xz * yz - xy * zz, 
                                det_y, 
                                xy * xz - yz * xx])
            else:
                abc = np.array([xy * yz - xz * yy, 
                                xy * xz - yz * xx, 
                                det_z])
        
        norm = np.linalg.norm(abc)
        if norm == 0.0:
            return None

        return Plane.from_normal_point(abc / norm, centroid)
        
    
class PlaneBounded(Plane):
    """
    Plane primitive.
    
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
    normal : 3 x 1 array
        Normal vector defining plane.
    canonical : Plane
        Return canonical form for testing.
    unbounded : Plane
        Unbounded version of plane.
    surface_area : float
        Surface area of bounded plane.
    volume : float
        Volume of plane, which is zero.
    holes : list
        List of holes in plane.
    rotation_from_axis : 3 x 3 array
        Rotation matrix that aligns z-axis with plane normal.
    
    Methods
    ------- 
    
    get_bounded_plane(points):
        Gives bounded version of plane, using input points to define 
        border.
    
    get_signed_distances(points):
        Gives the minimum distance between each point to the model. 
    
    get_distances(points)
        Gives the minimum distance between each point to the model. 
        
    get_normals(points)
        Gives, for each input point, the normal vector of the point closest 
        to the primitive. 
    
    get_angles_cos(self, points, normals):
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.
        
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
    
    get_mesh(points): TriangleMesh
        Flatten points and creates a simplified mesh of the plane defined
        by the points at the borders.
        
    get_square_mesh(length=1):
        Gives a square mesh that fits the plane model.
        
    add_holes(holes):
        Add one or more holes to plane.
        
    remove_hole(idx):
        Remove hole according to index.
        
    fit(points, normals=None):
        Gives plane that fits the input points.
        
    create_box(center=[0,0,0], dimensions=[1, 1, 1]):
        Gives list of planes that create, together, a closed box.
        
    create_circle(center, normal, radius, resolution=30):
        Creates circular plane.
        
    get_cropped_mesh(points=None, eps=1E-3):
        Creates mesh of the shape and crops it according to points.
        
    is_similar_to(other_shape, rtol=1e-02, atol=1e-02):
        Check if shapes represent same model.
        
    copy(copy_holes):
        Returns copy of plane
    
    align(axis):
        Returns aligned 
    
    """
    
    _name = 'bounded plane'
    _bounds_indices = np.array([])
    _bounds = np.array([])
    _bounds_projections = np.array([])
    
    @property
    def bounds_indices(self):
        return self._bounds_indices
    
    @property
    def bounds(self):
        return self._bounds
    
    @property
    def bounds_projections(self):
        return self._bounds_projections
    
    def translate(self, translation):
        """ Translate the shape.
        
        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        """
        Plane.translate(self.unbounded, translation)
        self._bounds = self._bounds + translation
        for hole in self.holes:
            hole._translate_points(translation)
    
    def rotate(self, rotation, is_hole=False):
        """ Rotate the shape.
        
        Parameters
        ----------
        rotation : 3 x 3 rotation matrix or scipy.spatial.transform.Rotation
            Rotation matrix.
        """
        if not hasattr(self, '_rotatable'):
            raise NotImplementedError('Shapes of type {shape.name} do not '
                                      'have an implemented _rotatable '
                                      'attribute')
        rotation = self._parse_rotation(rotation)
        Plane.rotate(self.unbounded, rotation)
        
        self._rotate_points_normals(rotation)
        
        bounds = rotation.apply(self._bounds)
        self._set_bounds(bounds, flatten=True)
        
        if not is_hole and len(self.holes) > 0:
            for hole in self.holes:
                hole.rotate(rotation, is_hole=True)
    
    def contains_projections(self, points):
        """ For each point in points, check if its projection on the plane lies
        inside of the plane's bounds. 
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        
        Returns
        -------
        array of booleans
            True for points whose projection lies in plane's bounds
        """
        inside = np.array([True] * len(points))
        projections = self.get_projections(points)
        for i in range(len(points)):
            point = projections[i]
            for j in range(1, len(self.bounds_projections)):
                p1 = self.bounds_projections[j-1]
                p2 = self.bounds_projections[j]
                if (point[0] - p1[0]) * (p2[1] - p1[1]) - (point[1] - p1[1]) * (p2[0] - p1[0]) > 0:
                    inside[i] = False
                    continue
        return inside
    
    def copy(self, copy_holes=True):
        """ Returns copy of plane
        
        Parameters
        ----------
        copy_holes: boolean, optional
            If True, also copy holes. Default: True.
        
        Returns
        -------
        Primitive
            Copied primitive
        """
        shape = PlaneBounded(self.model.copy(), bounds=None)
        shape._bounds_indices = self._bounds_indices.copy()
        shape._bounds = self._bounds.copy()
        shape._bounds_projections = self._bounds_projections.copy()
        shape._inlier_points = self._inlier_points.copy()
        shape._inlier_normals = self._inlier_normals.copy()
        shape._inlier_colors = self._inlier_colors.copy()
        shape._fusion_intersections = self._fusion_intersections.copy()
        shape._metrics = self._metrics.copy()
        if copy_holes:
            holes = [h.copy(copy_holes=False) for h in self._holes]
            shape._holes = holes
        return shape
    
    @property
    def surface_area(self):
        """ Surface area of bounded plane. """
        return self.get_mesh().get_surface_area()
    
    @property
    def rotation_from_axis(self):
        """ Rotation matrix that aligns z-axis with plane normal."""
        return get_rotation_from_axis([0, 0, 1], self.normal)
    
    def __init__(self, planemodel, bounds=None, rmse_max=1e-3, 
                 method='convex', alpha=None):
        """
        Parameters
        ----------
        planemodel : Plane or list of 4 values
            Shape defining plane
        bounds : array_like, shape (N, 3)
            Points defining bounds
        method : string, optional
            "convex" for convex hull, or "alpha" for alpha shapes.
            Default: "convex"
        alpha : float, optional
            Alpha parameter for alpha shapes algorithm. If equal to None,
            calculates the optimal alpha. Default: None
                        
        Raises
        ------
        ValueError
            If number of parameters is incompatible with the model of the 
        -------
            primitive.
        """
        
        if isinstance(planemodel, PlaneBounded):
            self.unbounded = planemodel.unbounded
        elif isinstance(planemodel, Plane):
            self.unbounded = planemodel
        else:
            self.unbounded = Plane(planemodel)
        # self._model = self.unbounded.model
        
        if bounds is None:
            warn('No input bounds, returning square plane')
            self = self.unbounded.get_square_plane(1)
        
        else:
            bounds = np.asarray(bounds)
            if bounds.shape[1] != 3:
                raise ValueError('Expected shape of bounds is (N, 3), got '
                                 f'{bounds.shape}')
                
            distances = self.unbounded.get_distances(bounds)
            rmse = np.sqrt(sum(distances * distances)) / len(distances)
            if rmse_max is not None and rmse_max > 1e-3:
                raise ValueError('Boundary points are not close enough to '
                                 f'plane: rmse={rmse}, expected less than '
                                 f'{rmse_max}.')
            
            self._set_bounds(bounds, flatten=True)
            
            if self._bounds is None:
                self = None
                
        self._holes = []
        
    @property
    def model(self):
        """ Return model. """
        return self.unbounded.model
    
    @property
    def canonical(self):
        """ Return canonical form for testing. """
        model = self.model
        if self.dist >= 0:
            model = -model
        canonical_plane = PlaneBounded(list(-self.model), self.bounds)
        canonical_plane._holes = self._holes
        return canonical_plane        
    
    def _set_bounds(self, points, flatten=True, method='convex', alpha=None):
        """ Flatten points according to plane model, get projections of 
        flattened points in the model and compute its boundary using either 
        the convex hull or alpha shapes.

        Parameters
        ----------
        plane : Plane
            Plane model
        points : array_like, shape (N, 3)
            Points corresponding to the fitted shape.
        flatten : bool, optional
            If False, does not flatten points
        method : string, optional
            "convex" for convex hull, or "alpha" for alpha shapes.
            Default: "convex"
        alpha : float, optional
            Alpha parameter for alpha shapes algorithm. If equal to None,
            calculates the optimal alpha. Default: None
        
        Returns
        -------
        bounds : array_like, shape (M, 3)
            Boundary points in plane, where M is lower or equal to N.
        projections : array_like, shape (M, 2)
            projections of boundary points in plane, where M is lower or 
            equal to N.
        """
        method = method.lower()
        if method == 'convex':
            pass
        elif method == 'alpha':
            raise NotImplementedError("Alpha shapes not implemented yet.")
        else:
            raise ValueError(f"method can be 'convex' or 'alpha', got {method}.")
        
        if flatten:
            points = self.flatten_points(points)
        if np.any(np.isnan(points)):
            raise ValueError('NaN found in points')
            
        projections = self.get_projections(points)
        
        if method == 'convex':
            chull = ConvexHull(projections)
            self._bounds_indices = chull.vertices
            self._bounds = points[chull.vertices]
            self._bounds_projections = projections[chull.vertices]
    
    def get_mesh(self, resolution=None):
        """ Flatten points and creates a simplified mesh of the plane defined
        by the points at the borders.
    
        Parameters
        ----------
        points : N x 3 array
            Points corresponding to the fitted shape.
        
        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """

        
        # points = self.bounds
        # projections = self.bounds_projections
        # print(f'has {len(self._fusion_intersections)} intersections')
        if len(self._fusion_intersections) == 0:
            points = self.bounds
            projections = self.bounds_projections
            idx_intersections_sorted = []
        else:
            points = np.vstack([self.bounds, self._fusion_intersections])
            rot = self.rotation_from_axis
            projections = (rot @ points.T).T[:, :2]
        
            angles = projections - projections.mean(axis=0)
            angles = np.arctan2(*angles.T) + np.pi
            idx = np.argsort(angles)
            
            points = points[idx]
            projections = projections[idx]
            
            idx_intersections = list(range(len(points) - len(self._fusion_intersections), len(points)))
            idx_intersections_sorted = [np.where(i == idx)[0][0] for i in idx_intersections]
            # print(idx_intersections)
        
        # holes = self._holes
        # has_holes = len(holes) != 0
        # has_holes = False
        # if has_holes:
        #     labels = []
        #     projections_holes = []
        #     points_holes = []
        #     labels_holes = []
        #     for i in range(len(holes)):
        #         hole = holes[i]
        #         projections_holes.append(hole.bounds_projections)
        #         points_holes.append(self.flatten_points(
        #             hole.bounds))
        #         labels += [i+1] * len(hole.bounds_projections)                
        #     labels = np.array(
        #         [0] * len(projections) + labels)
        #     projections = np.vstack([projections]+projections_holes)
        #     points = np.vstack([points]+points_holes)
            
        # is_points = np.asarray(is_points)
        # A = dict(vertices=plane.projections, holes=[circle.projections])
        # triangles = tr.triangulate(A)
        triangles = Delaunay(projections).simplices
        triangle_medians = np.median(points[triangles], axis=2)
        # if has_holes:
        #     for i in range(len(holes)):
        #         triangles = triangles[
        #             ~np.all(labels[triangles] == i+1, axis=1)]
        
        # has_holes = len(holes) != 0
        # if has_holes:
        triangles_in_holes = np.tile(False, len(triangles))
        for hole in self._holes:
            pass
            
                
        # print(len(triangles))
        for i in idx_intersections_sorted:
            select = []
            for triangle in triangles:
                test = {i-1, i, (i+1) % len(points)}.intersection(triangle)
                select.append(len(test) == 3)
            select = np.array(select)
            triangles = triangles[~select]
        
        # needed to make plane visible from both sides
        triangles = np.vstack([triangles, triangles[:, ::-1]])
        # print(len(triangles))
        
        mesh = TriangleMesh()
        mesh.vertices = Vector3dVector(points)
        mesh.triangles = Vector3iVector(triangles)
        
        if len(self.inlier_colors) > 0:
            # vertex_colors = []
            # for p in self.bounds:
            #     diff = np.linalg.norm(self.inlier_points - p, axis=1)
            #     i = np.where(diff == min(diff))[0][0]
            #     vertex_colors.append(self.inlier_colors[i])
            # mesh.vertex_colors = Vector3dVector(vertex_colors)
            mesh.paint_uniform_color(np.median(self.inlier_colors, axis=0))
        return mesh
    
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
        plane = Plane(np.random.random(4) * scale)
        return plane.get_square_plane(np.random.random() * scale)
    
    @staticmethod
    def fit(points, normals=None):
        """ Gives plane that fits the input points. If the numb
        points : N x 3 arrayer of points is
        higher than the `3`, the fitted shape will return a least squares 
        estimation.
        
        Reference:
            https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        
        Parameters
        ----------
        points : N x 3 array
            N input points 
        normals : N x 3 array
            N normal vectors
        
        Returns
        -------
        PlaneBounded
            Fitted plane.
        """
        
        plane = Plane.fit(points, normals)
        return plane.get_bounded_plane(points)
    
    @staticmethod
    def create_circle(center, normal, radius, resolution=30):
        """ Creates circular plane.
        
        Parameters
        ----------
        center : 3 x 1 array
            Center of circle.
        normal : 3 x 1 array
            Normal vector of plane.
        radius : float
            Radius of circle.
        resolution : int, optional
            Number of points defining circular plane.
        
        Returns
        -------
        PlaneBounded
            Fitted circular plane.
        """
        
        normalize = lambda x:x / np.linalg.norm(x)
        
        center = np.array(center)
        normal = normalize(np.array(normal))
        
        random_axis = normalize(np.random.random(3))
        ex = normalize(np.cross(random_axis, normal))
        ey = normalize(np.cross(normal, ex))
        
        theta = np.linspace(-np.pi, np.pi, resolution)[None].T
        points = center + (np.cos(theta) * ex + np.sin(theta) * ey) * radius
        
        plane = Plane.from_normal_point(normal, center)
        return plane.get_bounded_plane(points)
    
    @staticmethod
    def create_ellipse(center, vx, vy, resolution=30):
        """ Creates elliptical plane from two vectors. The input vectors are 
        interpreted as the two axes of the ellipse, multiplied by their radius.
        
        They must be orthogonal.
        
        Parameters
        ----------
        center : 3 x 1 array
            Center of circle.
        vx : 3 x 1 array
            First axis and multiplied by its semiradius.
        vy : 3 x 1 array
            Second axis and multiplied by its semiradius.
        resolution : int, optional
            Number of points defining circular plane.
        
        Returns
        -------
        PlaneBounded
            Fitted circular plane.
        """
        if np.dot(vx, vy) > 1e-5:
            raise ValueError('Axes must be orthogonal.')
        
        center = np.array(center)
        vx = np.array(vx)
        vy = np.array(vy)
        normal = np.cross(vx, vy)
        normal /= np.linalg.norm(normal)
        
        theta = np.linspace(-np.pi, np.pi, resolution+1)[None].T
        points = center + np.cos(theta) * vx + np.sin(theta) * vy
        
        plane = Plane.from_normal_point(normal, center)
        return plane.get_bounded_plane(points)
  
    @staticmethod  
    def create_box(center=[0,0,0], dimensions=[1, 1, 1]):
        """ Gives list of planes that create, together, a closed box.
        
        Reference:
            https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        
        Parameters
        ----------
        center : array
            Center point of box
        dimensions : array
            Dimensions of box
        
        Returns
        -------
        box : list
            List of planes.
        """
        
        vectors = np.eye(3) * np.array(dimensions) / 2
        center = np.array(center)
        planes = []
        
        for i, j in permutations([0, 1, 2], 2):
            k = 3 - i - j
            sign = int((i-j) * (j-k) * (k-i) / 2)
            v1, v2, v3 = vectors[[i, j, k]]
            
            points = center + np.array([
                + v1 + v2,
                + v1 - v2,
                - v1 - v2,
                - v1 + v2,
                ])
            
            plane = Plane.from_normal_point(v3, center + sign * v3)
            planes.append(plane.get_bounded_plane(points))
            
        return planes
    
    def intersection_bounds(self, other, within_segment=True, eps=1e-3):
        """ Calculates intersection point between bounding lines.
        
        Parameters
        ----------
        other : instance of PlaneBoundeds
            Other line to instersect.
        within_segment : boolean, optional
            If set to False, will suppose lines are infinite. Default: True.
        eps : float, optional
            Acceptable slack added to intervals in order to check if point
            lies on both lines, if 'within_segment' is True. Default: 1e-3.
        
        Returns
        -------
        point or None
            1x3 array containing point.
        """
        if not isinstance(other, PlaneBounded):
            raise ValueError("'other' must be an instance of PlaneBounded.")
            
        if len(self.bounds) == 0 or len(other.bounds) == 0:
            raise ValueError("Both planes must have bounds.")
            
        lines = self.bound_lines
        lines_other = other.bound_lines
        
        points = []
        for l1, l2 in product(lines, lines_other):
            p = l1.point_from_intersection(l2, within_segment=within_segment, eps=eps)
            if p is not None:
                points.append(p)
                
        if len(points) == 0:
            return np.array([])
        else:
            return np.vstack(points)
            
            
            
        
        
        
    
