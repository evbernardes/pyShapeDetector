#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
import warnings
from itertools import permutations, product, combinations
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation
from open3d.geometry import AxisAlignedBoundingBox
from pyShapeDetector.geometry import PointCloud, TriangleMesh

from pyShapeDetector.utility import (
    get_rotation_from_axis,
    # get_rectangular_grid,
    # select_grid_points
    )

from .primitivebase import Primitive
# from alphashape import alphashape
# from .line import Line    

def _fuse_loops(loop1, loop2):
    i, j = PointCloud.find_closest_points_indices(loop1, loop2, 1)
    i = i[0]
    j = j[0]
    
    loop2_rolled = np.vstack([
        np.roll(loop2, -j, axis=0), loop2[j]])
    
    loop_full = np.vstack([
        loop2_rolled,
        np.roll(loop1, -i, axis=0),
        loop1[i]])
    
    return loop_full

def _get_rectangular_vertices(v1, v2, eps=1e-8):
    if abs(v1.dot(v2)) > eps:
        raise RuntimeError("Vectors are not orthogonal.")
        
    return np.array([
        -v1 - v2,
        +v1 - v2,
        +v1 + v2,
        -v1 + v2]) / 2
    
class Plane(Primitive):
    """
    Plane primitive.

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
    inliers
    inliers_flattened
    inliers
    inliers_flattened
    has_inliers
    inlier_mean
    inlier_median
    inlier_points
    inlier_points_flattened
    inlier_normals
    inlier_colors
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
    parallel_vectors

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
    group_similar_shapes
    fuse_shape_groups
    fuse_similar_shapes

    from_normal_dist
    from_normal_point
    add_holes
    remove_hole
    get_fused_holes
    intersect
    get_unbounded_plane
    get_bounded_plane
    get_triangulated_plane
    get_bounded_planes_from_grid
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
    cut_with_cylinders
    create_circle
    create_ellipse
    create_box
    get_plane_intersections
    """

    _fit_n_min = 3
    _model_args_n = 4
    _name = 'plane'
    _holes = []
    _rotatable = [0, 1, 2]
    _translatable = []
    _fusion_intersections = np.array([])
    _color = np.array([0, 0, 1])
    _is_hole = False
    _convex = None
    _parallel_vectors = None

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
    def surface_area(self):
        """ For unbounded plane, returns NaN and gives warning """
        warnings.warn('For unbounded planes, the surface area is undefined')
        return float('nan')

    @property
    def volume(self):
        """ Volume of plane, which is zero. """
        return 0

    @property
    def canonical(self):
        """ Return canonical form for testing. """
        shape = self.copy()
        if np.sign(self.dist) < 0:
            shape._model = -self._model
        return shape
    
    @property
    def is_convex(self):
        return self._convex

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
    def holes(self):
        """ Existing holes in plane. """
        return self._holes
    
    @property
    def is_hole(self):
        return self._is_hole
    
    @property
    def parallel_vectors(self):
        if self._parallel_vectors is not None:
            return self._parallel_vectors
        
        elif self.has_inliers:
            self._parallel_vectors = self.get_rectangular_vectors_from_inliers(normalized=True)
            
        else:
            while True:
                random_vector = np.random.random(3)
                if abs(random_vector.dot(self.normal)) > 1e-3:
                    continue
                
            vy = random_vector / np.linalg.norm(random_vector)
            vx = np.cross(vy, self.normal)
            self._parallel_vectors = np.hstack([vx, vy])
            
        return self._parallel_vectors

    def __init__(self, model, decimals=None):
        """
        Parameters
        ----------
        model : Primitive or list of 4 values
            Shape defining plane
        decimals : int, optional
            Number of decimal places to round to (default: 0). If
            decimals is negative, it specifies the number of positions to
            the left of the decimal point. Default: None.

        Raises
        ------
        ValueError
            If number of parameters is incompatible with the model of the 
            primitive.
        """
        if isinstance(model, Primitive):
            shape = model
            model = model.model
            Primitive.__copy_atributes__(self, shape)
        else:
            model = np.array(model)

        norm = np.linalg.norm(model[:3])
        super().__init__(model / norm, decimals)
        self._holes = []

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
    
    def set_inliers(self, points_or_pointcloud, normals=None, colors=None, 
                    flatten=False, color_shape=False):
        """ Set inlier points to shape.
        
        If normals or/and colors are given, they must have the same shape as
        the input points.
        
        Parameters
        ----------
        points_or_pointcloud : N x 3 array or instance of open3d.geometry.PointCloud or Primitive
            Inlier points, pointcloud or shape containing points.
        normals : optional, N x 3 array
            Inlier point normals.
        colors : optional, N x 3 array
            Colors of inlier points.
        flatten : boolean, optional
            If True, flatten inlier points. Default: False.
        color_shape : boolean, optional
            If True, use inliers mean color for shape. Default: False.
        """
        
        super().set_inliers(
            points_or_pointcloud=points_or_pointcloud, 
            normals=None, colors=None,  flatten=False, color_shape=False)
        
        self._parallel_vectors = None
    
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
            
        warnings.warn("Unbounded planes have infinite axis aligned bounding boxes.")
        
        eps = 1e-3
        if np.linalg.norm(self.normal - [1, 0, 0]) < eps:
            idx = 0
        elif np.linalg.norm(self.normal - [0, 1, 0]) < eps:
            idx = 1
        elif np.linalg.norm(self.normal - [0, 0, 1]) < eps:
            idx = 2
        else:
            idx = -1
        
        centroid = self.centroid
        expand = np.array(
            [slack if n == idx else np.inf for n in range(3)])
        return AxisAlignedBoundingBox(centroid - expand, 
                                      centroid + expand)

    def get_mesh(self, **options):
        """ Flatten inliers points and creates a simplified mesh of the plane. If the
        shape has pre-defined inlier points, use them to find borders.
        Otherwise, return square mesh.

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        from .planebounded import PlaneBounded
        
        if not self.has_inliers:
            warnings.warn('No inlier points, returning square plane...')
            return self.get_square_mesh(1)

        bounded_plane = PlaneBounded(self.model, self.inliers_flattened.points)
        # bounded_plane.__copy_atributes__(self)
        mesh = bounded_plane.get_mesh()

        return mesh

    def translate(self, translation, translate_inliers=True):
        """ Translate the shape.

        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        translate_inliers : boolean, optional
            If True, also translate inliers. Default: True.
        """
        super().translate(translation, translate_inliers=translate_inliers)

        centroid = self.centroid + translation
        self._model = Plane.from_normal_point(
            self.normal, centroid).model
            
        if not self.is_hole and len(self.holes) > 0:
            for hole in self.holes:
                hole.translate(translation)

    def rotate(self, rotation, rotate_inliers=True):
        """ Rotate the shape.

        Parameters
        ----------
        rotation : 3 x 3 rotation matrix or scipy.spatial.transform.Rotation
            Rotation matrix.
        rotate_inliers : boolean, optional
            If True, also rotate inliers. Default: True.
        """

        centroid = rotation.apply(self.centroid)
        normal = rotation.apply(self.normal)
        
        rotation = Primitive._parse_rotation(rotation)
        
        # for everything else
        super().rotate(rotation, rotate_inliers=rotate_inliers)
        
        # only for model
        self._model = Plane.from_normal_point(normal, centroid).model
        
        if not self.is_hole and len(self.holes) > 0:
            for hole in self.holes:
                hole.rotate(rotation)

    def __copy_atributes__(self, shape_original):
        super().__copy_atributes__(shape_original)
        self._model = shape_original._model.copy()
        self._fusion_intersections = shape_original._fusion_intersections.copy()
        self._is_hole = shape_original._is_hole
        if not shape_original.is_hole:
            self._holes = [h.copy() for h in shape_original._holes]

    # def bound_lines_meshes(self, radius=0.001, color=(0, 0, 0)):
    #     lines = self.bound_lines
    #     meshes = [line.get_mesh(radius=radius) for line in lines]
    #     [mesh.paint_uniform_color(color) for mesh in meshes]
    #     return meshes

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
        from .planebounded import PlaneBounded
        if remove_points and not isinstance(self, PlaneBounded):
            warnings.warn('Option remove_points only works if plane is bounded.')
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

            inside1 = self.contains_projections(hole.bounds)
            if sum(inside1) < 1:
                warnings.warn('shape does not contain hole')
                continue

            if remove_points:
                # inside1 = self.contains_projections(hole.bounds)
                # if sum(inside1) < 1:
                #     print('shape does not contain hole')
                #     continue
                if sum(~inside1) > 0:
                    intersections = []
                    for l1, l2 in product(hole.bound_lines, self.bound_lines):
                        if (point := l1.point_from_intersection(l2)) is not None:
                            intersections.append(point)
                    bounds = np.vstack([hole.bounds[inside1]]+intersections)
                else:
                    bounds = hole.bounds

                inside2 = hole.contains_projections(self.bounds)
                hole = PlaneBounded(model, bounds, convex=hole.is_convex)
                self._bounds = self._bounds[~inside2]
                self._bounds_projections = self._bounds_projections[~inside2]
            else:
                hole = PlaneBounded(model, hole.bounds, convex=hole.is_convex)
            hole._is_hole = True
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
        
    def get_fused_holes(self):
        """ Fuse bounds of all holes into single PlaneBounded instance.

        Raises
        ------
        ValueError
            When shape has no holes

        Returns
        -------
        PlaneBounded
        """
        if len(self.holes) == 0:
            raise ValueError("Shape has no holes.")
        
        if len(self.holes) == 1:
            # warnings.warn("Shape only has one hole, returning it.")
            bounds = self.holes[0].bounds
            if self.holes[0].is_clockwise:
                bounds = bounds[::-1]
        
        else:
            idx = {}
            holes = self.holes
            for i, j in combinations(range(len(holes)), 2):
                h1 = holes[i]
                h2 = holes[j]
                idx[PointCloud.find_closest_points(h1.bounds_projections, h2.bounds_projections)[1][0]] = (i, j)
                
            distances = np.sort(list(idx.keys()))
            ordered = []
            for d in distances:
                ordered += [i for i in idx[d] if i not in ordered]

            holes_ordered = np.array(holes)[ordered]
            bounds = holes_ordered[0].bounds
            if holes_ordered[0].is_clockwise:
                bounds = bounds[::-1]
                
            for hole in holes_ordered[1:]:
                bounds_new = hole.bounds
                if hole.is_clockwise:
                    bounds_new = bounds_new[::-1]
                bounds = _fuse_loops(bounds, bounds_new)
        
        return self.get_bounded_plane(bounds, convex=False)

    def intersect(self, other_plane, separated=False, intersect_parallel=False,
                  eps_angle=np.deg2rad(0.9), eps_distance=1e-2):
        """ Calculate the line defining the intersection with another planes.

        If separated is True, give two colinear lines, each one fitting the
        projection of one of the planes.

        Parameters
        ----------
        other_plane : instance of Plane of PlaneBounded
            Second plane
        separated : boolean, optional
            If separated is True, give two colinear lines, each one fitting the
            projection of one of the planes.
            Default: False.
        intersect_parallel : boolean, optional
            If True, try to intersect parallel planes too. Default: False.
        eps_angle : float, optional
            Minimal angle (in radians) between normals necessary for detecting
            whether planes are parallel. Default: 0.0017453292519943296
        eps_distance : float, optional
            When planes are parallel, eps_distance is used to detect if the 
            planes are close enough to each other in the dimension aligned
            with their axes. Default: 1e-2.

        Returns
        -------
        Line or None
        """
        from .planebounded import PlaneBounded
        if not isinstance(other_plane, PlaneBounded):
            raise ValueError("Only intersection with other instances of "
                             "PlaneBounded is implemented.")
        from .line import Line

        if not separated:
            return Line.from_plane_intersection(
                self, other_plane, intersect_parallel=intersect_parallel, eps_angle=eps_angle, eps_distance=eps_distance)

        line = Line.from_plane_intersection(
            self, other_plane, intersect_parallel=intersect_parallel, eps_angle=eps_angle, eps_distance=eps_distance, fit_bounds=False)

        line1 = line.get_line_fitted_to_projections(self.bounds)
        line2 = line.get_line_fitted_to_projections(other_plane.bounds)

        return line1, line2
    
    def get_unbounded_plane(self):
        """ Gives unbounded version of plane

        Returns
        -------
        Bounded
            unbounded version of plane.
        """
        plane = Plane(self)
        plane.__copy_atributes__(self)
        return plane

    def get_bounded_plane(self, bounds, convex=True):
        """ Gives bounded version of plane, using input points to define 
        border.

        If points is None, then use vertices or inliers, if they exist.

        Parameters
        ----------
        points : array_like, shape (N, 3).
            Bound points.
        convex : bool, optinal
            If True, assumes the bounds are supposed to be convex and use
            ConvexHull. If False, assume bounds are directly given as a loop.

        Returns
        -------
        PlaneBounded
            Bounded version of plane.
        """
        from .planebounded import PlaneBounded

        plane = PlaneBounded(self.model, bounds, convex)
        Plane.__copy_atributes__(plane, self)
        return plane
    
    def get_triangulated_plane(self, vertices, triangles):
        """ Gives bounded version of plane, using input points to define 
        border.

        Parameters
        ----------
        vertices : array_like, shape (N, 3).
            Vertices for triangles.
        vertices : array_like, shape (N, 3).
            Vertices for triangles.

        Returns
        -------
        PlaneBounded
            Bounded version of plane.
        """
        from .planetriangulated import PlaneTriangulated

        plane = PlaneTriangulated(self.model, vertices, triangles)
        Plane.__copy_atributes__(plane, self)
        return plane
    
    def get_bounded_planes_from_grid(self, 
                                     grid_width, 
                                     max_point_dist=None, 
                                     grid_type = "hexagonal", 
                                     perimeter_multiplier=1,
                                     return_rect_grid=False, 
                                     perimeter_eps=1e-3,
                                     only_inside=False, 
                                     add_boundary=False,
                                     detect_holes=False,
                                     add_inliers=True,
                                     angle_colinear=0,
                                     colinear_recursive=True,
                                     contract_bounds=False,
                                     min_inliers=1):
        """
        Experimental method of triangulating plane with a grid.

        Uses Plane.get_triangulated_plane_from_grid to get a triangulated plane,
        and then detects its bounds to create bounded planes with holes.
        
        See:
            Plane.get_triangulated_plane_from_grid
        
        Parameters
        ----------
        grid_width : float
            Distance between two points in first dimension (and also second 
            dimension for regular grids).
        max_point_dist : float, optional
            Max distance allowed between grid points and inlier points. If not 
            given, same value for "grid_width" will be used.
        grid_type : str, optional
            Type of grid, can be "hexagonal" or "regular". Default: "hexagonal".
        perimeter_multiplier : float, option
            Multiplies expected perimeter to create cutout value. Cannot be 
            smaller than 1. Default: 1.
        return_rect_grid : boolean, optional
            If True, tuple containing rectangular plane and grid. Default: False.
        perimeter_eps : float, option
            Small slack value added to perimeter testing. Default: 1e-3.
        only_inside : boolean, optional
            If True, remove grid points not contained in original convex 
            boundary. Default: False.
        add_boundary : boolean, optional
            If True, add convex boundary points to grid. Default: False.
        detect_holes : boolean, optional
            If True, try to detect holes. Default: False.
        add_inliers : boolean, optional
            If True, add inlier points.
        angle_colinear : float, optional
            Small angle value for assuming two lines are colinear. Default: 0
        colinear_recursive : boolean, optional
            If False, only try to simplify loop once. If True, try to simplify
            it until no more simplification is possible. Default: True.
        contract_bounds : boolean, optional
            If True, contract bounds to closest inlier points. Default: False.
        min_inliers : int, optional
            If add_inliers is True, remove planes with less inliers than this
            value. Default: 1.
            
        Return
        ------
        PlaneTriangulated
            PlaneTriangulated instance from grid

        """
        result = self.get_triangulated_plane_from_grid(
            grid_width=grid_width, 
            max_point_dist=max_point_dist, 
            grid_type=grid_type, 
            perimeter_multiplier=perimeter_multiplier, 
            perimeter_eps=perimeter_eps, 
            return_rect_grid=return_rect_grid,
            only_inside=only_inside,
            add_boundary=add_boundary)
        
        if return_rect_grid:
            plane_triangulated, plane_rect, grid = result
        else:
            plane_triangulated = result
            
        planes = plane_triangulated.get_bounded_planes_from_boundaries(
            detect_holes=detect_holes, 
            add_inliers=add_inliers, 
            angle_colinear=angle_colinear, 
            colinear_recursive=colinear_recursive,
            contract_bounds=contract_bounds,
            min_inliers=min_inliers)
        
        if return_rect_grid:
            return planes, plane_rect, grid
        return planes
        
    def get_triangulated_plane_from_grid(self, grid_width, max_point_dist=None, 
                                         grid_type = "hexagonal", perimeter_multiplier=1,
                                         return_rect_grid=False, perimeter_eps=1e-3,
                                         only_inside=False, add_boundary=False):
        """
        Experimental method of triangulating plane with a grid.
        
        First, find a rectangle that contains all inliers.
        
        Second, create a grid inside this rectangle according to `grid_width`
        and `grid_type` parameters.
        
        Third, remove all grid points that aren't close enough to the inlier 
        points, according to `max_point_dist`.
        
        Finally, triangulate remaining grid points and remove all triangles 
        with perimeter bigger than `perimeter_multiplier * perimeter + perimeter_eps` 
        where `perimeter` is the theoretical perimeter of each triangle 
        depending on the grid type, and `perimeter_eps` is a small slack value.        
        
        Parameters
        ----------
        grid_width : float
            Distance between two points in first dimension (and also second 
            dimension for regular grids).
        max_point_dist : float, optional
            Max distance allowed between grid points and inlier points. If not 
            given, same value for "grid_width" will be used.
        grid_type : str, optional
            Type of grid, can be "hexagonal" or "regular". Default: "hexagonal".
        perimeter_multiplier : float, option
            Multiplies expected perimeter to create cutout value. Cannot be 
            smaller than 1. Default: 1.
        return_rect_grid : boolean, optional
            If True, tuple containing rectangular plane and grid. Default: False.
        perimeter_eps : float, optional
            Small slack value added to perimeter testing. Default: 1e-3.
        only_inside : boolean, optional
            If True, remove grid points not contained in original convex 
            boundary. Default: False.
        add_boundary : boolean, optional
            If True, add convex boundary points to grid. Default: False.
            
        Return
        ------
        PlaneTriangulated
            PlaneTriangulated instance from grid

        """
        
        if not self.has_inliers:
            raise RuntimeError("Plane has no inliers.")
        
        from .planetriangulated import PlaneTriangulated
        
        if perimeter_multiplier < 1:
            raise ValueError("perimeter_multiplier cannot be smaller than 1, "
                             f"got {perimeter_multiplier}")
                             
        if perimeter_eps < 0:
            raise ValueError(f"perimeter_eps has to be non-negative, got {perimeter_eps}")
        
        # Get rectangular plane
        vectors, center = self.get_rectangular_vectors_from_inliers(return_center=True)
        # plane_rect = self.get_rectangular_plane(vectors, center)
        
        # grid inside rectangle and select nearby points
        grid, perimeter = PointCloud.get_rectangular_grid(
            vectors, center, grid_width, grid_type=grid_type, return_perimeter=True)
        
        # flattening, just in case
        grid.points = self.flatten_points(grid.points)
        
        # TODO: slowest thing here is this:
        grid_points_selected = grid.select_nearby_points(self.inlier_points_flattened, max_point_dist).points
        
        if only_inside:
            grid_points_selected = grid_points_selected[self.contains_projections(grid_points_selected)]
            
        if add_boundary:
            chull = ConvexHull(self.get_projections(self.inlier_points))
            grid_points_selected = np.vstack(
                [grid_points_selected, self.inlier_points[chull.vertices]])
        
        # triangulate and remove big triangles
        projections = self.get_projections(grid_points_selected)
        triangles = Delaunay(projections).simplices
        perimeters = TriangleMesh(grid_points_selected, triangles).get_triangle_perimeters()
        select = perimeters < perimeter_multiplier * perimeter + perimeter_eps
        triangles = triangles[select]
        
        plane = PlaneTriangulated(self, grid_points_selected, triangles)
        plane.set_inliers(
            self.inlier_points,
            self.inlier_normals,
            self.inlier_colors,
            )
        
        if return_rect_grid:
            plane_rect = self.get_rectangular_plane(vectors, center)
            return plane, plane_rect, grid
        return plane

    def get_projections(self, points, return_rotation=False):
        """ Get 2D projections of points in plane.

        See: get_points_from_projections

        Parameters
        ----------
        points : array_like, shape (N, 3)
            Points corresponding to the fitted shape.
        return_rotation : boolean, optional
            If True, return rotation matrix. Default: False.

        Returns
        -------
        projections : array_like, shape (N, 2)
            2D projections of boundary points in plane
        """
        if PointCloud.is_instance_or_open3d(points):
            points = np.asarray(points.points)
        
        points = np.asarray(points)
        if points.shape[1] != 3:
            raise ValueError("Input points must be 3D.")
        rot = get_rotation_from_axis([0, 0, 1], self.normal)
        projections = (rot @ points.T).T[:, :2]
        
        if return_rotation:
            return projections, rot
        return projections

    def get_points_from_projections(self, projections, return_rotation=False):
        """ Get 3D points from 2D projections in plane.

        See: get_projections

        Parameters
        ----------
        projections : array_like, shape (N, 2)
            2D projections of boundary points in plane
        return_rotation : boolean, optional
            If True, return rotation matrix. Default: False.

        Returns
        -------
        projections : array_like, shape (N, 3)
            Points corresponding to the fitted shape.
        """
        projections = np.asarray(projections)
        if projections.shape[1] != 2:
            raise ValueError("Input points must be 2D.")
        N = len(projections)

        rot = get_rotation_from_axis([0, 0, 1], self.normal)
        proj_z = (rot @ self.centroid)[2]
        projections_3D = np.vstack([projections.T, np.repeat(proj_z, N)]).T

        points = (rot.T @ projections_3D.T).T
    
        if return_rotation:
            return points, rot.T
        return points

    def get_mesh_alphashape(self, points, alpha=None):
        """ Flatten input points and creates a simplified mesh of the plane 
        using alpha shapes. 

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        from pyShapeDetector.utility import alphashape_2d, new_TriangleMesh
        projections = self.get_projections(points)
        vertices_2d, triangles = alphashape_2d(projections, alpha)
        vertices = self.get_points_from_projections(vertices_2d)
        triangles = np.vstack([triangles, triangles[:, ::-1]])
        return new_TriangleMesh(vertices, triangles)
    
    def get_rectangular_vectors_from_inliers(self, return_center=False, use_PCA=True, normalized=False):
        """ Gives vectors defining a rectangle that roughly contains the plane.
        
        Parameters
        ----------
        return_center : boolean, optional
            If True, return tuple containing both vectors and calculated center.
        use_PCA : boolean, optional
            If True, use PCA to detect vectors (better for rectangles). If False,
            use eigenvectors from covariance matrix. Default: True.
        normalized : boolean, optional
            If True, return normalized vectors. Default: False.

        Returns
        -------
        numpy.array of shape (2, 3)
            Two non unit vectors
        """
        points = self.inlier_points_flattened
        
        center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
        delta = points - center
        
        delta_projection, rot = self.get_projections(delta, return_rotation=True)
        
        if use_PCA:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca.fit(delta_projection)
            v1, v2 = pca.components_
        else:
            cov_matrix = np.cov(delta_projection, rowvar=False)
            _, (v1, v2) = np.linalg.eig(cov_matrix)
        
        v1 = rot.T @ np.hstack([v1, 0])
        v2 = rot.T @ np.hstack([v2, 0])
        

        V1 = 2 * max(abs(delta.dot(v1))) * v1
        V2 = 2 * max(abs(delta.dot(v2))) * v2
        
        if normalized:
            V1 /= np.linalg.norm(V1)
            V2 /= np.linalg.norm(V2)
        
        vectors = np.array([V1, V2])
        
        if return_center:
            return vectors, center
        return vectors
    
    def get_rectangular_plane(self, vectors=None, center=None):
        """ Gives rectangular plane defined two vectors and its center.
        
        Vectors v1 and v2 should not be unit, and instead have lengths equivalent
        to widths of rectangle.
        
        If no vectors are given, calculate them with get_rectangular_vectors_from_inliers.
        
        Parameters
        ----------
        vectors : arraylike of shape (2, 3), optional
            The two orthogonal unit vectors defining the rectangle plane.
        center : arraylike of length 3, optional
            Center of rectangle. If not given, either inliers or centroid are 
            used.

        Returns
        -------
        PlaneBounded
            Rectangular plane
        """
        if vectors is None:                
            vectors, center_rect = self.get_rectangular_vectors_from_inliers(return_center=True)
            if center is None:
                center = center_rect
        
        if center is None:
            if self.has_inliers:
                points = self.inlier_points
                # center = np.median(self.inlier_points, axis=0)
                center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
            else:
                center = self.centroid
                
        V1, V2 = vectors
        vertices = center + _get_rectangular_vertices(V1, V2)
        return self.get_bounded_plane(vertices)
    
    def get_polygon_plane(self, sides, radius, center=None):
        """ Gives plane defined by regular polygon points.

        Parameters
        ----------
        sides : int
            Number of sides of polygon
        radius : float
            Distance from center of polygon to any of its points
        center : arraylike of length 3, optional
            Center of rectangle. If not given, either inliers or centroid are 
            used.

        Returns
        -------
        PlaneBounded
            Polygonal plane
        """
        if sides < 3 or not isinstance(sides, int):
            raise ValueError(f"sides must be an int bigger than 2, got {sides}")
        
        if center is None:
            if self.has_inliers:
                center = np.median(self.inlier_points, axis=0)
            else:
                center = self.centroid
                
        # Calculate the central angle between each vertex
        central_angle = 2 * np.pi / sides
        
        # Initialize a list to store the coordinates of the points
        points = []
        
        x = np.cos(central_angle * np.arange(sides))
        y = np.sin(central_angle * np.arange(sides))
        points = self.get_points_from_projections(radius * np.vstack([x, y]).T)
        return self.get_bounded_plane(center + points)
    
    def get_square_plane(self, length, center=None):
        """ Gives square plane defined by four points.

        Parameters
        ----------
        length : float
            Length of the sides of the square
        center : arraylike of length 3, optional
            Center of rectangle. If not given, either inliers or centroid are 
            used.

        Returns
        -------
        PlaneBounded
            Square plane
        """
        
        # TODO: Replace as an internal call to get_polygon_plane?
        if center is None:
            if self.has_inliers:
                center = np.median(self.inlier_points, axis=0)
            else:
                center = self.centroid
        
        def normalized(x): return x / np.linalg.norm(x)

        if np.isclose(self.normal[1], 1, atol=1e-7):
            v1 = normalized(np.cross(self.normal, [0, 0, 1]))
            v2 = normalized(np.cross(v1, self.normal))
        else:
            v1 = normalized(np.cross([0, 1, 0], self.normal))
            v2 = normalized(np.cross(self.normal, v1))

        vertices = center + _get_rectangular_vertices(length * v1, 
                                                      length * v2)

        return self.get_bounded_plane(vertices)

    def get_square_mesh(self, length, center=None):
        """ Gives a square mesh that fits the plane model.   

        Parameters
        ----------
        length : float
            Length of the sides of the square
        center : arraylike of length 3, optional
            Center of rectangle. If not given, either inliers or centroid are 
            used.

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        return self.get_square_plane(length, center).get_mesh()
    
    def get_rectangular_mesh(self, v1, v2, center=None):
        """ Gives a rectangular mesh that fits the plane model.

        Parameters
        ----------
        v1 : arraylike of length 3
            First vector defining one of the directions, must be orthogonal
            to v2.
        v2 : arraylike of length 3
            Second vector defining one of the directions, must be orthogonal
            to v1.
        center : arraylike of length 3, optional
            Center of rectangle. If not given, either inliers or centroid are 
            used.

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the plane.
        """
        return self.get_rectangular_plane(v1, v2, center).get_mesh()
    
    def cut_with_cylinders(shapes, radius_min, total_cut=False, eps=0):
        """ Isolates planes and cylinders. For every plane and cylinder 
        combination, check if cylinder cuts plane and, if it does, add a hole.
        
        Parameters
        ----------
        shapes : list of shapes
            List containing all shapes.
        radius_min : float
            Only isolates cylinders with radius below this threshold.
        total_cut : boolean, optional
            When True, only accepts cuts when either the top of the bottom 
            completely cuts the plane. Default: False.
        eps : float, optional
            Adds some backlash to top and bottom of cylinder. Default: 0.
        """
        from pyShapeDetector.primitives import Plane, Cylinder
        
        cylinders = [s for s in shapes if isinstance(s, Cylinder) and s.radius < radius_min + eps]
    
        for s in shapes:
            if not isinstance(s, Cylinder):
                warn(f"Ignoring shape of type {type(s)} (not a Cylinder).")
            elif s.radius < radius_min + eps and c.cuts(self, total_cut=total_cut, eps=eps):
                self.add_holes(c.project_to_plane(self))
    
    @classmethod
    def create_circle(cls, center, normal, radius, resolution=30):
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
        Plane
            Circular plane.
        """

        def normalize(x): return x / np.linalg.norm(x)

        center = np.array(center)
        normal = normalize(np.array(normal))

        random_axis = normalize(np.random.random(3))
        ex = normalize(np.cross(random_axis, normal))
        ey = normalize(np.cross(normal, ex))

        theta = np.linspace(-np.pi, np.pi, resolution + 1)[None].T
        points = center + (np.cos(theta) * ex + np.sin(theta) * ey) * radius

        plane_unbounded = Plane.from_normal_point(normal, center)
        plane_unbounded.set_inliers(points)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cls(plane_unbounded)

    @classmethod
    def create_ellipse(cls, center, vx, vy, resolution=30):
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
        Plane
            Elliptical plane.
        """
        center = np.array(center)
        vx = np.array(vx)
        vy = np.array(vy)

        if np.dot(vx, vy) > 1e-5:
            raise ValueError('Axes must be orthogonal.')

        normal = np.cross(vx, vy)
        normal = normal / np.linalg.norm(normal)

        theta = np.linspace(-np.pi, np.pi, resolution+1)[None].T
        points = center + np.cos(theta) * vx + np.sin(theta) * vy

        plane_unbounded = Plane.from_normal_point(normal, center)
        plane_unbounded.set_inliers(points)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cls(plane_unbounded)
    
    @classmethod
    def create_box(cls, center=[0, 0, 0], dimensions=[1, 1, 1]):
        """ Gives list of planes that create, together, a closed box.

        Parameters
        ----------plane_unbounded
        center : array
            Center point of box
        dimensions : array
            Dimensions of box

        Returns
        -------
        box : list
            List of PlaneBounded instances.
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

            plane_unbounded = Plane.from_normal_point(v3, center + sign * v3)
            plane_unbounded.set_inliers(points)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                planes.append(cls(plane_unbounded))

        return planes
    
    @staticmethod
    def get_plane_intersections(
        shapes, bbox_intersection=None, inlier_max_distance=None,
        length_max=None, distance_max=None, ignore=None, 
        intersect_parallel=False, eps_angle=np.deg2rad(5.0), 
        eps_distance=1e-2):
        """ For every possible pair of neighboring bounded planes, calculate their
        intersection and return dictionary of all intersection lines.
        
        Checking bbox_intersection is much quicker than checking inlier_max_distance,
        which is why it is a good idea to check both.
        
        See: group_shape_groups, fuse_shape_groups, glue_nearby_planes
        
        Parameters
        ----------
        shapes : list of shapes
            List containing all shapes.  
        bbox_intersection : float, optional
            Max distance between planes.
        inlier_max_distance : float, optional
            Max distance between points in shapes. If None, ignore this test.
            Default: None.
        length_max : float, optional
            If a value is given, limits lenghts of intersection lines. 
            Default: None.
        distance_max : float, optional
            If a value is given, limits the distance of intersections. 
            Default: None.
        ignore : list of booleans, optional
            If a list of booleans is given, ignore every ith plane in shapes if
            the ith value of 'ignore' is True.
        intersect_parallel : boolean, optional
            If True, try to intersect parallel planes too. Default: False.
        eps_angle : float, optional
            Minimal angle (in radians) between normals necessary for detecting
            whether planes are parallel. Default: 0.08726646259971647 (5 degrees).
        eps_distance : float, optional
            When planes are parallel, eps_distance is used to detect if the 
            planes are close enough to each other in the dimension aligned
            with their axes. Default: 1e-2.
        
        Returns
        -------
        dict
            Dictionary of intersections.
        """
        # if bbox_intersection is None and inlier_max_distance is None:
        #     raise ValueError("bbox_intersection and inlier_max_distance cannot "
        #                      "both be equal to None.")
        
        from pyShapeDetector.primitives import Plane, PlaneBounded, Line
        
        if length_max is not None and length_max <= 0:
            raise ValueError(f"'length_max' must be a positive float, got {length_max}." )    
        
        if ignore is None:
            ignore = [False] * len(shapes)
            
        if len(ignore) != len(shapes):
            raise ValueError("'ignore' must be a list of booleans the size of 'shapes'.")
        
        lines = []
        intersections = {}
        num_shapes = len(shapes)
    
        for i, j in combinations(range(num_shapes), 2):
        
            if not isinstance(shapes[i], Plane) or not isinstance(shapes[j], Plane):
                continue
            
            # if not shapes[i].is_convex or not shapes[j].is_convex:
            #     continue
            
            if ignore[i] or ignore[j]:
                continue
            
            both_bounded = isinstance(shapes[i], PlaneBounded) and isinstance(shapes[j], PlaneBounded)
            
            if both_bounded and not shapes[i].check_bbox_intersection(shapes[j], bbox_intersection):
                continue
            
            if both_bounded and not shapes[i].check_inlier_distance(shapes[j], inlier_max_distance):
                continue
            
            line = Line.from_plane_intersection(
                shapes[i], shapes[j], intersect_parallel=intersect_parallel,
                eps_angle=eps_angle, eps_distance=eps_distance)
            
            if line is None:
                continue
            
            if length_max is not None and line.length > length_max:
                continue
            
            if distance_max is not None:
                # TODO: bounds_or_vertices should be changed for bounds if we are 
                # sure that it will never be implemented for non convex planes
                points = shapes[i].bounds_or_vertices
    
                if min(line.get_distances(points)) > distance_max:
                    continue
                    # continue
                if min(line.get_distances(points)) > distance_max:
                    continue
                    # continue
            
            lines.append(line)            
            intersections[i, j] = line
            
        return intersections