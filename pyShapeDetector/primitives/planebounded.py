#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:15:09 2024

@author: ebernardes
"""
import warnings
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
# from scipy.spatial.transform import Rotation
from open3d.geometry import TriangleMesh, AxisAlignedBoundingBox
from open3d.utility import Vector3iVector, Vector3dVector

from pyShapeDetector.utility import (
    get_triangle_surface_areas, 
    # fuse_vertices_triangles, 
    # planes_ressample_and_triangulate,
    triangulate_earclipping,
    # get_triangle_boundary_indexes,
    # get_loop_indexes_from_boundary_indexes
    # find_closest_points,
    # find_closest_points_indices,
    get_triangle_points,
    simplify_loop_with_angle,
    )
# from .primitivebase import Primitive
from .plane import Plane
# from alphashape import alphashape

def _is_clockwise(bounds):
    s = 0
    N = len(bounds)
    for i in range(N):
        point = bounds[i]
        point2 = bounds[(i + 1) % N]
        s += (point2[0] - point[0]) * (point2[1] + point[1])
    return s > 0

class PlaneBounded(Plane):
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
    inlier_mean
    inlier_median
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

    is_convex
    normal
    dist
    centroid
    holes
    is_hole

    is_clockwise
    bounds
    bounds_indices
    bounds_projections
    bound_lines
    bound_LineSet
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
    __put_attributes_in_dict__
    save
    __get_attributes_from_dict__
    load
    check_bbox_intersection
    check_inlier_distance
    fuse

    from_normal_dist
    from_normal_point
    add_holes
    remove_hole
    get_fused_holes
    intersect
    closest_bounds
    get_unbounded_plane
    get_bounded_plane
    get_triangulated_plane
    get_triangulated_plane_from_grid
    get_projections
    get_points_from_projections
    get_mesh_alphashape
    get_polygon_plane
    get_square_plane
    get_rectangular_vectors_from_inliers
    get_rectangular_plane
    get_square_mesh
    get_rectangular_mesh
    create_circle
    create_ellipse
    create_box

    closest_bounds
    contains_projections
    bound_lines_meshes
    set_bounds
    add_bound_points
    intersection_bounds
    simplify_bounds_colinear
    """
    _name = 'bounded plane'
    _bounds_indices = np.array([])
    _bounds = np.array([])
    _bounds_projections = np.array([])
    _convex = True
    _is_clockwise = None

    @property
    def surface_area(self):
        """ Surface area of bounded plane. """
        
        def shoelace(projections):
            # Reference:
            # https://en.wikipedia.org/wiki/Shoelace_formula
            i = np.arange(len(projections))
            x, y = projections.T
            return np.abs(np.sum(x[i-1] * y[i] - x[i] * y[i-1]) * 0.5)
        
        surface_area = shoelace(self.bounds_projections)
        for hole in self.holes:
            surface_area -= shoelace(hole.bounds_projections)
            
        return surface_area
    
    @property
    def is_clockwise(self):
        return self._is_clockwise

    @property
    def bounds(self):
        return self._bounds

    @property
    def bounds_indices(self):
        """ Indices of points corresponding to bounds. """
        # TODO: should take into consideration added bounds
        return self._bounds_indices

    @property
    def bounds_projections(self):
        return self._bounds_projections

    @property
    def bound_lines(self):
        """ Lines defining bounds. """
        from .line import Line
        return Line.from_bounds(self.bounds)
    

    @property
    def bound_LineSet(self):
        """ Lines defining bounds. """
        from .line import Line
        return Line.get_LineSet_from_list(self.bound_lines)
    
    @property
    def bounds_or_vertices(self):            
        return self.bounds
    
    @property
    def bounds_or_vertices_or_inliers(self):
        if len(self.vertices) > 0:
            return self.bounds
        else:
            return self.inlier_points

    def __init__(self, model, bounds=None, convex=True, decimals=None):
        """
        Parameters
        ----------
        model : Primitive or list of 4 values
            Shape defining plane
        bounds : array_like, shape (N, 3), optional
            Points defining bounds.
        convex : bool, optinal
            If True, assumes the bounds are supposed to be convex and use
            ConvexHull. If False, assume bounds are directly given as a loop.
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
        
        # from .planetriangulated import PlaneTriangulated
        
        flatten = True
        if bounds is None:
            if isinstance(model, Plane) and model.has_inliers:
                warnings.warn('No input bounds, using inliers.')
                bounds = model.inlier_points
                convex = False
            # elif isinstance(model, PlaneTriangulated):
            #     print("No input bounds, using PlaneTriangulated's boundary.")
            #     boundary_indexes = get_triangle_boundary_indexes(
            #         self.vertices, 
            #         self.triangles)
            #     loops = get_loop_indexes_from_boundary_indexes(boundary_indexes)
            #     bounds = self.vertices[loops[0]]
            #     flatten = False
            #     convex = False
            else:
                warnings.warn('No input bounds, returning square plane.')
                bounds = self.get_square_plane(1).bounds
                flatten = False
                convex = False

        # super().__init__(model, decimals)
        self.set_bounds(bounds, flatten=flatten, convex=convex)

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
        PlaneBounded
            Random shape.
        """
        model = np.random.random(4) * scale
        plane = Plane(model, decimals=decimals)
        length = np.random.random() * scale
        return plane.get_square_plane(np.round(length, decimals))

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

        # if len(self._fusion_intersections) == 0:
        #     points = self.bounds
        #     projections = self.bounds_projections
        #     # idx_intersections_sorted = []
        # else:
        #     points = np.vstack([self.bounds, self._fusion_intersections])
        #     projections = self.get_projections(points)

        #     angles = projections - projections.mean(axis=0)
        #     angles = np.arctan2(*angles.T) + np.pi
        #     idx = np.argsort(angles)

        #     points = points[idx]
        #     projections = projections[idx]
        projections = self.bounds_projections
        holes = self._holes
        
        if self.is_convex:
            if len(holes) >= 0:
                points_holes = [self.flatten_points(hole.bounds) for hole in holes]
                points = np.vstack([self.bounds]+points_holes)
                projections = self.get_projections(points)
        
            triangles = Delaunay(projections).simplices

            if len(self._holes) > 0:
                triangles_center = get_triangle_points(projections, triangles).mean(axis=1)
                
                for hole in self._holes:
                    inside_hole = np.array(
                        [hole.contains_projections(p, input_is_2D=True).all() for p in triangles_center])
                    triangles = triangles[~inside_hole]
                    triangles_center = triangles_center[~inside_hole]
                
        else:
            if not self.is_clockwise:
                projections = projections[::-1]
            
            if not self.is_hole and len(self.holes) > 0:
                
                from .plane import _fuse_loops
                fused_hole = self.get_fused_holes()
                projections = _fuse_loops(projections, fused_hole.bounds_projections)
            
            points = self.get_points_from_projections(projections)
            triangles = triangulate_earclipping(projections)

        areas = get_triangle_surface_areas(points, triangles)
        triangles = triangles[areas > 0]

        mesh = TriangleMesh()
        mesh.vertices = Vector3dVector(points)
        mesh.triangles = Vector3iVector(triangles)
        
        # if this fails, the ear-clipping triangulation with holes failed
        # try:
        #     np.testing.assert_almost_equal(mesh.get_surface_area(),
        #                                    self.surface_area)
        # except:
        #     print(f"Area diff = {self.surface_area - mesh.get_surface_area()}")
        return mesh

    # def get_mesh(self, **options):
    #     """ Flatten points and creates a simplified mesh of the plane defined
    #     by the points at the borders.

    #     Parameters
    #     ----------
    #     points : N x 3 array
    #         Points corresponding to the fitted shape.

    #     Returns
    #     -------
    #     TriangleMesh
    #         Mesh corresponding to the plane.
    #     """

    #     if len(self._fusion_intersections) == 0:
    #         points = self.bounds
    #         projections = self.bounds_projections
    #         # idx_intersections_sorted = []
    #     else:
    #         points = np.vstack([self.bounds, self._fusion_intersections])
    #         projections = self.get_projections(points)

    #         angles = projections - projections.mean(axis=0)
    #         angles = np.arctan2(*angles.T) + np.pi
    #         idx = np.argsort(angles)

    #         points = points[idx]
    #         projections = projections[idx]

    #         # idx_intersections = list(
    #             # range(len(points) - len(self._fusion_intersections), len(points)))
    #         # idx_intersections_sorted = [
    #             # np.where(i == idx)[0][0] for i in idx_intersections]

    #     holes = self._holes
    #     has_holes = len(holes) != 0
    #     if has_holes:
    #         points_holes = [self.flatten_points(hole.bounds) for hole in holes]
    #         points = np.vstack([points]+points_holes)
    #         projections = self.get_projections(points)
        
    #     if self.is_convex:
    #         triangles = Delaunay(projections).simplices
    #     else:
    #         triangles = triangulate_earclipping(projections)

    #     for hole in self._holes:
    #         inside_hole = np.array(
    #             [hole.contains_projections(p).all() for p in points[triangles]])
    #         triangles = triangles[~inside_hole]

    #     areas = get_triangle_surface_areas(points, triangles)
    #     triangles = triangles[areas > 0]

    #     # needed to make plane visible from both sides
    #     # triangles = np.vstack([triangles, triangles[:, ::-1]])

    #     mesh = TriangleMesh()
    #     mesh.vertices = Vector3dVector(points)
    #     mesh.triangles = Vector3iVector(triangles)

    #     return mesh
    
    def __copy_atributes__(self, shape_original):
        super().__copy_atributes__(shape_original)
        self._bounds_indices = shape_original._bounds_indices.copy()
        self._bounds = shape_original._bounds.copy()
        self._bounds_projections = shape_original._bounds_projections.copy()
        self._is_clockwise = _is_clockwise(self._bounds_projections)
        self._convex = shape_original._convex

    def translate(self, translation):
        """ Translate the shape.

        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        """
        # Primitive.translate(self, translation)
        super().translate(translation)
        self._bounds = self._bounds + translation

    def rotate(self, rotation):
        """ Rotate the shape.

        Parameters
        ----------
        rotation : 3 x 3 rotation matrix or scipy.spatial.transform.Rotation
            Rotation matrix.
        """
        rotation = self._parse_rotation(rotation)
        super().rotate(rotation)

        self._bounds = rotation.apply(self._bounds)    
        
    def __put_attributes_in_dict__(self, data):
        super().__put_attributes_in_dict__(data)
        
        # additional PlaneBounded related data:
        data['bounds'] = self.bounds.tolist()
        data['_fusion_intersections'] = self._fusion_intersections.tolist()
        data['hole_bounds'] = [h.bounds.tolist() for h in self.holes]
        data['convex'] = self.is_convex
        data['hole_convex'] = [h.is_convex for h in self.holes]
        
    def __get_attributes_from_dict__(self, data):
        super().__get_attributes_from_dict__(data)
        
        # additional PlaneBounded related data:
        convex = data.get('convex', True)
        self.set_bounds(data['bounds'], convex=convex)
        self._fusion_intersections = np.array(data['_fusion_intersections'])
        
        hole_bounds = data['hole_bounds']
        try:
            hole_convex = data['hole_convex']
        except KeyError:
            hole_convex = [True] * len(hole_bounds)
            
        holes = []
        for bounds, convex in zip(hole_bounds, hole_convex):
            holes.append(
                PlaneBounded(self.model, bounds, convex=convex))
            
        # no need to remove points, as they were already tested when creating
        # the plane
        self.add_holes(holes, remove_points=False)
                
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
        
    #     if len(self.bounds) > 0:
    #         points = self.bounds
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
        min_bound = np.min(self.bounds, axis=0)
        max_bound = np.max(self.bounds, axis=0)
        return AxisAlignedBoundingBox(min_bound - slack, max_bound + slack)
                
    @staticmethod
    def fuse(shapes, detector=None, ignore_extra_data=False, **extra_options):
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
            
        Returns
        -------
        PlaneBounded
            Averaged PlaneBounded instance.    
        """
        plane_unbounded = Plane.fuse(shapes, detector, ignore_extra_data)
        bounds = np.vstack([s.bounds for s in shapes])
        
        if not np.all([s.is_convex for s in shapes]):
            warnings.warn("Non-convex PlaneBounded instance detected, fused "
                          "plane will be convex.")
            
        shape = PlaneBounded(plane_unbounded.model, bounds)
        if not ignore_extra_data:
            shape.set_inliers(plane_unbounded)
            shape.metrics = plane_unbounded.metrics
            
        return shape
    
    def closest_bounds(self, other_plane, n=1):
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

        if not isinstance(other_plane, PlaneBounded):
            raise ValueError("Only implemented with other instances of "
                             "PlaneBounded.")

        from pyShapeDetector.utility import find_closest_points

        closest_points, distances = find_closest_points(
            self.bounds, other_plane.bounds, n)

        return closest_points, distances

    def contains_projections(self, points, input_is_2D=False):
        """ For each point in points, check if its projection on the plane lies
        inside of the plane's bounds. 

        Parameters
        ----------
        points : N x 3 array (or N x 2 array)
            N input points if input_is_2D is False, or N 2D projections if
            input_is_2D is True.
        input_is_2D : boolean
            If False, calculate projections from points. If True, uses input 
            directly.

        Returns
        -------
        array of booleans
            True for points whose projection lies in plane's bounds
        """
        
        # inside = np.array([True] * len(points))
        inside = []
        points = np.atleast_2d(points)
        if points.shape[1] == 3:
            if input_is_2D:
                raise ValueError("For 3D input points, input_is_2D must be False.")
            projections = self.get_projections(points)
        elif points.shape[1] == 2:
            if not input_is_2D:
                raise ValueError("For 2D input projections, input_is_2D must be True.")
            projections = points
            
        # N = len(self.bounds_projections)
        bounds = self.bounds_projections
        bounds_shifted = np.roll(bounds, -1, axis=0)
        diff = bounds_shifted - bounds
        for projection in projections:
            
            diff1 = projection - bounds
            diff2 = projection - bounds_shifted
            
            with warnings.catch_warnings():
                # Divisions by zero lead to infs that give correct comparisons
                warnings.simplefilter("ignore")
                test = np.logical_and(
                    (diff1[:, 1] < 0) != (diff2[:, 1] < 0),
                    projection[0] < diff[:, 0] * diff1[:, 1] / diff[:, 1] + bounds[:, 0])

            inside.append(np.sum(test) % 2 == 1)
                
        return np.array(inside)
    
    def bound_lines_meshes(self, radius=0.001, color=(0, 0, 0)):
        lines = self.bound_lines
        meshes = [line.get_mesh(radius=radius) for line in lines]
        [mesh.paint_uniform_color(color) for mesh in meshes]
        return meshes
    
    def set_bounds(self, bounds, flatten=True, convex=True):
        """ Flatten points according to plane model, get projections of 
        flattened points in the model and compute its boundary using either 
        the convex hull or alpha shapes.

        Parameters
        ----------
        plane : Plane
            Plane model
        bounds : array_like, shape (N, 3)
            Points corresponding to the fitted shape.
        flatten : bool, optional
            If False, does not flatten points
        convex : bool, optinal
            If True, assumes the bounds are supposed to be convex and use
            ConvexHull. If False, assume bounds are directly given as a loop.

        """
        self._mesh = None
        bounds = np.asarray(bounds)
        
        if bounds.shape[1] != 3 :
            raise ValueError("Invalid shape of 'bounds' array.")
        
        if flatten:
            bounds = self.flatten_points(bounds)
        if np.any(np.isnan(bounds)):
            raise ValueError('NaN found in points')
            
        # self._vertices = np.array([])
        # self._triangles = np.array([])
        # self._convex = True

        projections = self.get_projections(bounds)

        # if method == 'convex':
        if convex:
            chull = ConvexHull(projections)
            self._bounds_indices = chull.vertices
            self._bounds = bounds[chull.vertices]
            self._bounds_projections = projections[chull.vertices]
        else:
            self._bounds_indices = np.array(range(len(bounds)))
            self._bounds = bounds
            self._bounds_projections = projections
            
        self._is_clockwise = _is_clockwise(self._bounds_projections)
        self._convex = convex
        
    def add_bound_points(self, new_bound_points, flatten=True):
        """ Add points to current bounds.

        Parameters
        ----------
        new_bound_points : N x 3 np.array
            New points to be added.
        flatten : bool, optional
            If False, does not flatten points

        """
        if flatten:
            new_bound_points = self.flatten_points(new_bound_points)
        bounds = np.vstack([self.bounds, new_bound_points])
        self.set_bounds(bounds, flatten=False)
        # self.set_bounds(points, flatten, method, alpha)

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

        self._metrics = self._metrics.copy()
        self._color = self._color.copy()
        points = []
        for l1, l2 in product(lines, lines_other):
            p = l1.point_from_intersection(
                l2, within_segment=within_segment, eps=eps)
            if p is not None:
                points.append(p)

        if len(points) == 0:
            return np.array([])
        else:
            return np.vstack(points)
        
    def simplify_bounds_colinear(self, angle_colinear=0, colinear_recursive=True):
        """
        Simplify bounds be removing some if they are colinear (or almost colinear).
        
        Parameters
        ----------
        angle_colinear : float, optional
            Small angle value for assuming two lines are colinear. Default: 0
        colinear_recursive : boolean, optional
            If False, only try to simplify loop once. If True, try to simplify
            it until no more simplification is possible. Default: True.
        """
        indices = simplify_loop_with_angle(
            self.bounds, range(len(self.bounds)), angle_colinear, colinear_recursive)
        
        bounds_new = self.bounds[indices]
        self.set_bounds(bounds_new, flatten=False, convex=self.is_convex)
        