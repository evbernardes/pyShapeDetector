#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:15:09 2024

@author: ebernardes
"""
import warnings
from itertools import permutations, product, combinations
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
# from scipy.spatial.transform import Rotation
from open3d.geometry import TriangleMesh, AxisAlignedBoundingBox
from open3d.utility import Vector3iVector, Vector3dVector

from pyShapeDetector.utility import (
    fuse_vertices_triangles, 
    planes_ressample_and_triangulate,
    get_triangle_boundary_indexes,
    get_loop_indexes_from_boundary_indexes)
from .primitivebase import Primitive
from .plane import Plane
# from alphashape import alphashape

class PlaneTriangulated(Plane):
    """
    PlaneBounded primitive.

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
    has_inliers
    inlier_points
    inlier_points_flattened
    inlier_normals
    inlier_colors
    inlier_PointCloud
    inlier_PointCloud_flattened
    metricsd
    axis_spherical
    axis_cylindrical
    bbox
    bbox_bounds
    inlier_bbox
    inlier_bbox_bounds

    is_convex
    normal
    dist
    centroid
    holes
    is_hole

    vertices
    triangles
    bounds_or_vertices

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
    __copy_atributes__
    __copy__
    copy
    translate
    rotate
    align
    save
    load
    check_bbox_intersection
    check_inlier_distance
    fuse

    from_normal_dist
    from_normal_point
    add_holes
    remove_hole
    intersect
    get_unbounded_plane
    get_bounded_plane
    get_triangulated_plane
    get_projections
    get_points_from_projections
    get_mesh_alphashape
    get_square_plane
    get_rectangular_vectors_from_inliers
    get_rectangular_plane
    get_square_mesh
    get_rectangular_mesh
    create_circle
    create_ellipse
    create_box

    closest_vertices
    set_vertices_triangles
    from_plane_with_mesh
    get_bounded_planes_from_boundaries
    """
    _name = 'triangulated plane'
    _vertices = np.array([])
    _triangles = np.array([])
    # TODO: maybe set _convex to None as it cannot be known
    _convex = False

    @property
    def surface_area(self):
        """ Surface area of triangulated plane. """
        
        triangle_projections = self.get_projections(self.vertices)[self.triangles]
        diff = np.diff(triangle_projections, axis=1)
        areas = abs(np.cross(diff[:,0,:], diff[:,1,:])) * 0.5
        surface_area = sum(areas)
            
        return surface_area

    @property
    def vertices(self):
        return self._vertices

    @property
    def triangles(self):
        return self._triangles
    
    @property
    def bounds_or_vertices(self):            
        return self.vertices
    
    @property
    def bounds_or_vertices_or_inliers(self):
        if len(self.vertices) > 0:
            return self.vertices
        else:
            return self.inlier_points

    def __init__(self, model, vertices=None, triangles=None,
                 decimals=None):
        """
        Parameters
        ----------
        model : Primitive or list of 4 values
            Shape defining plane
        vertices : array_like, shape (N, 3)
            Vertices of plane triangulation.
        triangles : array_like, shape (N, 3)
            Vertices of plane triangulation.
        decimals : int, optional
            Number of decimal places to round to (default: 0). If
            decimals is negative, it specifies the number of positions to
            the left of the decimal point. Default: None.

        Raises
        ------
        ValueError
            If number of parameters is incompatible with the model of the 
        """
        super().__init__(model, decimals)

        flatten = True
        if vertices is None and triangles is None:
            if isinstance(model, Plane):
                warnings.warn('No input bounds, using inliers mesh.')
                mesh = model.mesh
            else:
                warnings.warn('No input vertices/triangles, returning square plane')
                mesh = self.get_square_plane(1).mesh
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            flatten = False

        elif vertices is not None or triangles is not None:
            pass
        elif vertices is None or triangles is None:
            raise ValueError("Either 'vertices' and 'triangles' should be given, or one of them.")

        # super().__init__(model, decimals)
        self.set_vertices_triangles(vertices, triangles, flatten=flatten)

    @classmethod
    def random(cls, scale=1, decimals=16):
        """ Generates a random shape.

        Parameters
        ----------
        scale : float, optional
            scaling factor for random model values.
        decimals : int, optional
            Number of decimal places to round to (default: 0). If
            decimals is negative, it specifies the number of positions to
            the left of the decimal point.

        Returns
        -------
        PlaneTriangulated
            Random shape.
        """
        model = np.random.random(4) * scale
        plane = Plane(model, decimals=decimals)
        length = np.random.random() * scale
        mesh = plane.get_square_plane(np.round(length, decimals)).mesh
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        return plane.get_triangulated_plane(vertices, triangles)

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
        return PlaneTriangulated.from_bounded_plane(
            plane.get_bounded_plane(points)
        )

    def get_mesh(self, **options):
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
        mesh = TriangleMesh()
        mesh.vertices = Vector3dVector(self.vertices)
        mesh.triangles = Vector3iVector(self.triangles)

        return mesh
    
    def __copy_atributes__(self, shape_original):
        super().__copy_atributes__(shape_original)
        self._vertices = shape_original._vertices.copy()
        self._triangles = shape_original._triangles.copy()

    def translate(self, translation):
        """ Translate the shape.

        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        """
        # Primitive.translate(self, translation)
        super().translate(translation)
        self._vertices = self._vertices + translation

    def rotate(self, rotation):
        """ Rotate the shape.

        Parameters
        ----------
        rotation : 3 x 3 rotation matrix or scipy.spatial.transform.Rotation
            Rotation matrix.
        """
        rotation = self._parse_rotation(rotation)
        super().rotate(rotation)

        self._vertices = rotation.apply(self._vertices)        
                
    # def get_inliers_axis_aligned_bounding_box(self, slack=0, num_sample=15):
    #     """ If the shape includes inlier points, returns the minimum and 
    #     maximum bounds of their bounding box.
        
    #     If no inlier points, use bounds or vertices.
        
    #     Parameters
    #     ----------
    #     slack : float, optional
    #         Expand bounding box in all directions, useful for testing purposes.
    #         Default: 0.
    #     num_sample : int, optional
    #         If no inliers, bounds or vertices found, sample mesh instead.
    #         Default: 15.
            
    #     Returns
    #     -------
    #     tuple of two 3 x 1 arrays
    #         Minimum and maximum bounds of inlier points bounding box.
    #     """
        
    #     if len(self.vertices) > 0:
    #         points = self.vertices
    #     else:
    #         return Primitive.inliers_bounding_box(slack=0, num_sample=15)
        
    #     slack = abs(slack)
    #     min_bound = np.min(points, axis=0)
    #     max_bound = np.max(points, axis=0)
    #     return AxisAlignedBoundingBox(min_bound - slack, max_bound + slack)
    
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
        min_bound = np.min(self.vertices, axis=0)
        max_bound = np.max(self.vertices, axis=0)
        return AxisAlignedBoundingBox(min_bound - slack, max_bound + slack)
                
    @staticmethod
    def fuse(shapes, detector=None, ignore_extra_data=False, line_intersection_eps=1e-3,
             **extra_options):
        """ Find weigthed average of shapes, where the weight is the fitness
        metric.
        
        If a detector is given, use it to compute the metrics of the resulting
        average shapes.
        
        Parameters
        ----------
        shapes : list
            Grouped shapes. All shapes must be of the same type.
        detector : instance of some Detector, optional
            Used to recompute metrics. Default: None.
        ignore_extra_data : boolean, optional
            If True, ignore everything and only fuse model. Default: False.
        line_intersection_eps : float, optional
            Distance for detection of intersection between planes. Default: 0.001.
        force_concave : boolean, optional.
            If True, the fused plane will be concave regardless of inputs.
            Default: True.
        ressample_density : float, optional
            Default: 1.5
        ressample_radius_ratio : float, optional
            Default: 1.2
            
        Returns
        -------
        PlaneBounded
            Averaged PlaneBounded instance.    
        """
        force_concave = extra_options.get('force_concave', True)
        ressample_density = extra_options.get('ressample_density', 1.5)
        ressample_radius_ratio = extra_options.get('ressample_radius_ratio', 1.2)
        
        if len(shapes) == 1:
            return shapes[0]
        elif isinstance(shapes, Primitive):
            return shapes
        
        shape = Plane.fuse(shapes, detector, ignore_extra_data)
        
        is_convex = np.array([shape.is_convex for shape in shapes])
        all_convex = is_convex.all()
        if not all_convex and is_convex.any():
            force_concave = True
            # raise ValueError("If 'force_concave' is False, PlaneBounded "
            #                  "instances should either all be convex or "
            #                  "all non convex.")
        
        if not ignore_extra_data:
            if force_concave:
                vertices, triangles = planes_ressample_and_triangulate(
                    shapes, ressample_density, ressample_radius_ratio, double_triangles=True)
                shape.set_vertices_triangles(vertices, triangles)
                
            elif not all_convex:
                vertices = [shape.vertices for shape in shapes]
                triangles = [shape.triangles for shape in shapes]
                vertices, triangles = fuse_vertices_triangles(vertices, triangles)                                       
                shape.set_vertices_triangles(vertices, triangles)
                
            else:
                bounds = np.vstack([shape.bounds for shape in shapes])
                shape.set_bounds(bounds)
            
                intersections = []
                for plane1, plane2 in combinations(shapes, 2):
                    points = plane1.intersection_bounds(plane2, True, eps=line_intersection_eps)
                    if len(points) > 0:
                        intersections.append(points)
                
                # temporary hack, saving intersections for mesh generation
                if len(intersections) > 0:
                    shape._fusion_intersections = np.vstack(intersections)
        
        return shape
    
    def closest_vertices(self, other_plane, n=1):
        """ Returns n pairs of closest bound points with a second plane.

        Parameters
        ----------            
        other_plane : Plane
            Another plane.
        n : int, optional
            Number of pairs. Default=1.

        Returns
        -------
        closest_points : np.array
            Pairs of points.
        distances : np.array
            Distances for each pair.
        """

        if not isinstance(other_plane, PlaneTriangulated):
            raise ValueError("Only implemented with other instances of "
                             "PlaneBounded.")

        from pyShapeDetector.utility import find_closest_points

        closest_points, distances = find_closest_points(
            self.vertices, other_plane.vertices, n)

        return closest_points, distances

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
        raise RuntimeError("'contains_projections' not implemented for "
                           "PlaneTriangulated instances.")
        
    def set_vertices_triangles(self, vertices, triangles, flatten=True):
        """ Flatten points according to plane model, get projections of 
        flattened points in the model and set desired vertices and triangles.

        Parameters
        ----------
        plane : Plane
            Plane model
        vertices : array_like, shape (N, 3)
            Vertices of plane triangulation.
        triangles : array_like, shape (N, 3)
            Vertices of plane triangulation.
        flatten : bool, optional
            If False, does not flatten points

        """
        self._mesh = None
        vertices = np.asarray(vertices)
        triangles = np.asarray(triangles)
        
        if vertices.shape[1] != 3 or triangles.shape[1] != 3:
            raise ValueError("Invalid shape of 'vertices' and/or 'triangles' "
                             "array.")
        
        if not all([x.is_integer() for x in triangles.flatten()]):
            raise ValueError("All elements of 'triangles' must be integers.")
        
        if (triangles >= len(vertices)).any() or (triangles < 0).any():
            raise ValueError("Each element in 'triangles' should be an integer"
                             " between 0 and len(vertices) - 1.")

        if flatten:
            vertices = self.flatten_points(vertices)
        if np.any(np.isnan(vertices)):
            raise ValueError('NaN found in points')
            
        # self._bounds = np.array([])
        # self._bounds_indices = np.array([])
        self._vertices = vertices
        self._triangles = triangles
        # self._convex = False
        
    @staticmethod
    def from_plane_with_mesh(plane):
        """ Convert plane instance's mesh into PlaneTriangulated instance by

        By copying the vertices and triangles from its mesh.

        Parameters
        ----------
        plane_bounded : PlaneBounded
            Input PlaneBounded instance

        Returns
        -------
        PlaneTriangulated
        """
        if not isinstance(plane, Plane):
            raise TypeError("Can only convert plane meshes into PlaneTriangulated.")
        mesh = plane.mesh
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        return plane.get_triangulated_plane(vertices, triangles)
    
    def get_bounded_planes_from_boundaries(self):
        """ Convert PlaneTriangulated instance into list of non-convex 
        PlaneBounded instances.

        Returns
        -------
        list of PlaneBounded instances
        """
        from .planebounded import PlaneBounded
        
        boundary_indexes = get_triangle_boundary_indexes(
            self.vertices, 
            self.triangles)
        loop_indexes = get_loop_indexes_from_boundary_indexes(boundary_indexes)

        return [PlaneBounded(self.model, self.vertices[loop], convex=False) for loop in loop_indexes]
