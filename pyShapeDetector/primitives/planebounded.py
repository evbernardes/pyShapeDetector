#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:15:09 2024

@author: ebernardes
"""
import warnings
from itertools import product
import numpy as np
from importlib.util import find_spec
from scipy.spatial import QhullError, ConvexHull, Delaunay

from sklearn.decomposition import PCA

# from scipy.spatial.transform import Rotation
# from open3d.geometry import AxisAlignedBoundingBox
from pyShapeDetector.geometry import (
    PointCloud,
    TriangleMesh,
    AxisAlignedBoundingBox,
    OrientedBoundingBox,
)
from .plane import Plane

from pyShapeDetector.utility import get_rotation_from_axis

if has_mapbox_earcut := find_spec("mapbox_earcut") is not None:
    from mapbox_earcut import triangulate_float32


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
    aabb
    obb

    is_convex
    normal
    dist
    centroid
    holes
    is_hole
    parallel_vectors

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
    get_axis_aligned_bounding_box
    get_oriented_bounding_box
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
    get_rectangular_vectors_from_points
    get_rectangular_plane
    get_square_mesh
    get_rectangular_mesh
    cut_with_cylinders
    create_circle
    create_ellipse
    create_box
    get_plane_intersections

    closest_bounds
    contains_projections
    bound_lines_meshes
    set_bounds
    add_bound_points
    intersection_bounds
    simplify_bounds_colinear
    contract_bounds
    glue_planes_with_intersections
    """

    _name = "bounded plane"
    _bounds_indices = np.array([])
    _bounds = np.array([])
    _bounds_projections = np.array([])
    _convex = True
    _is_clockwise = None

    @property
    def surface_area(self):
        """Surface area of bounded plane."""

        def shoelace(projections):
            # Reference:
            # https://en.wikipedia.org/wiki/Shoelace_formula
            i = np.arange(len(projections))
            x, y = projections.T
            return np.abs(np.sum(x[i - 1] * y[i] - x[i] * y[i - 1]) * 0.5)

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
        """Indices of points corresponding to bounds."""
        # TODO: should take into consideration added bounds
        return self._bounds_indices

    @property
    def bounds_projections(self):
        return self._bounds_projections

    @property
    def bound_lines(self):
        """Lines defining bounds."""
        from .line import Line

        return Line.from_bounds(self.bounds)

    @property
    def bound_LineSet(self):
        """Lines defining bounds."""
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
                warnings.warn("No input bounds, using inliers.")
                bounds = model.inliers.points
                convex = True
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
                warnings.warn("No input bounds, returning square plane.")
                bounds = self.get_square_plane(1).bounds
                flatten = False
                convex = True

        # super().__init__(model, decimals)
        self.set_bounds(bounds, flatten=flatten, convex=convex)

    @classmethod
    def random(cls, scale=1, decimals=16):
        """Generates a random shape.

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
        """Gives plane that fits the input points. If the numb
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

    def get_oriented_bounding_box(self, slack=0):
        """Returns an oriented bounding box of the primitive.

        Parameters
        ----------
        slack : float, optional
            Expand bounding box in all directions, useful for testing purposes.
            Default: 0.

        See: open3d.geometry.get_oriented_bounding_box

        Returns
        -------
        OrientedBoundingBox
        """
        if slack < 0:
            raise ValueError("Slack must be non-negative.")
        oriented_bbox = self.get_rectangular_oriented_bounding_box_from_points()
        return oriented_bbox.expanded(slack)

    def get_mesh(self, **options):
        """Flatten points and creates a simplified mesh of the plane defined
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

        projections = self.bounds_projections
        holes = self._holes

        if self.is_convex:
            if len(holes) >= 0:
                points_holes = [self.flatten_points(hole.bounds) for hole in holes]
                points = np.vstack([self.bounds] + points_holes)
                projections = self.get_projections(points)

            triangles = Delaunay(projections).simplices

            if len(self._holes) > 0:
                triangles_center = projections[triangles].mean(axis=1)

                for hole in self._holes:
                    inside_hole = np.array(
                        [
                            hole.contains_projections(p, input_is_2D=True).all()
                            for p in triangles_center
                        ]
                    )
                    triangles = triangles[~inside_hole]
                    triangles_center = triangles_center[~inside_hole]

        else:
            if has_mapbox_earcut:
                all_points = [self.bounds] + [h.bounds for h in self.holes]
                points = np.vstack(all_points)
                projections = self.get_projections(points)
                rings = [len(self.bounds)]
                for hole in self.holes:
                    rings.append(rings[-1] + len(hole.bounds))

                triangles = triangulate_float32(projections, rings).reshape(-1, 3)

            else:
                warnings.warn(
                    "mapbox_earcut not present, triangulation might not work properly."
                )

                if not self.is_clockwise:
                    projections = projections[::-1]

                if not self.is_hole and len(self.holes) > 0:
                    from .plane import _fuse_loops

                    fused_hole = self.get_fused_holes()
                    projections = _fuse_loops(
                        projections, fused_hole.bounds_projections
                    )

                points = self.get_points_from_projections(projections)
                triangles = TriangleMesh.triangulate_earclipping(projections)

        areas = TriangleMesh(points, triangles).get_triangle_surface_areas()
        triangles = triangles[areas > 0]

        mesh = TriangleMesh(points, triangles)

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

    def translate(self, translation, translate_inliers=True):
        """Translate the shape.

        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        translate_inliers : boolean, optional
            If True, also translate inliers. Default: True.
        """
        # Primitive.translate(self, translation)
        super().translate(translation, translate_inliers=translate_inliers)
        self._bounds = self._bounds + translation

    def rotate(self, rotation, rotate_inliers=True):
        """Rotate the shape.

        Parameters
        ----------
        rotation : 3 x 3 rotation matrix or scipy.spatial.transform.Rotation
            Rotation matrix.
        rotate_inliers : boolean, optional
            If True, also rotate inliers. Default: True.
        """
        rotation = self._parse_rotation(rotation)
        super().rotate(rotation, rotate_inliers=rotate_inliers)

        self._bounds = rotation.apply(self._bounds)

    def __put_attributes_in_dict__(self, data):
        super().__put_attributes_in_dict__(data)

        # additional PlaneBounded related data:
        data["bounds"] = self.bounds.tolist()
        data["_fusion_intersections"] = self._fusion_intersections.tolist()
        data["hole_bounds"] = [h.bounds.tolist() for h in self.holes]
        data["convex"] = self.is_convex
        data["hole_convex"] = [h.is_convex for h in self.holes]

    def __get_attributes_from_dict__(self, data):
        super().__get_attributes_from_dict__(data)

        # additional PlaneBounded related data:
        convex = data.get("convex", True)
        self.set_bounds(data["bounds"], convex=convex)
        self._fusion_intersections = np.array(data["_fusion_intersections"])

        hole_bounds = data["hole_bounds"]
        try:
            hole_convex = data["hole_convex"]
        except KeyError:
            hole_convex = [True] * len(hole_bounds)

        holes = []
        for bounds, convex in zip(hole_bounds, hole_convex):
            holes.append(PlaneBounded(self.model, bounds, convex=convex))

        # no need to remove points, as they were already tested when creating
        # the plane
        self.add_holes(holes, remove_points=False)

    def get_axis_aligned_bounding_box(self, slack=0):
        """Returns an axis-aligned bounding box of the primitive.

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
        """Find weigthed average of shapes, where the weight is the fitness
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
            warnings.warn(
                "Non-convex PlaneBounded instance detected, fused "
                "plane will be convex."
            )

        shape = PlaneBounded(plane_unbounded.model, bounds)
        if not ignore_extra_data:
            shape.set_inliers(plane_unbounded)
            shape.metrics = plane_unbounded.metrics

        return shape

    def get_rectangular_oriented_bounding_box_from_points(
        self, points=None, use_PCA=True
    ):
        """Gives oriented bounding box contains the plane.

        If points are not given, use inliers.

        Parameters
        ----------
        points : Nx3 array, optional
            Points used to find rectangle.
        return_center : boolean, optional
            If True, return tuple containing both vectors and calculated center.
        use_PCA : boolean, optional
            If True, use PCA to detect vectors (better for rectangles). If False,
            use eigenvectors from covariance matrix. Default: True.

        Returns
        -------
        numpy.array of shape (2, 3)
            Two non unit vectors
        """
        if points is None:
            points = self.bounds

        return super().get_rectangular_oriented_bounding_box_from_points(
            points, use_PCA=use_PCA
        )

    def get_rectangular_vectors_from_points(
        self, points=None, return_center=False, use_PCA=True, normalized=False
    ):
        """Gives vectors defining a rectangle that roughly contains the plane.

        If points are not given, use bounds.

        Parameters
        ----------
        points : Nx3 array, optional
            Points used to find rectangle.
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
        if points is None:
            points = self.bounds
            if self.has_inliers:
                points = np.vstack([points, self.inliers.points])

        return super().get_rectangular_vectors_from_points(
            points=points,
            return_center=return_center,
            use_PCA=use_PCA,
            normalized=normalized,
        )

    def closest_bounds(self, other_plane, n=1):
        """Returns n pairs of closest bound points with a second plane.

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
            raise ValueError("Only implemented with other instances of PlaneBounded.")

        closest_points, distances = PointCloud.find_closest_points(
            self.bounds, other_plane.bounds, n
        )

        return closest_points, distances

    def contains_projections(self, points, input_is_2D=False):
        """For each point in points, check if its projection on the plane lies
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
                    projection[0]
                    < diff[:, 0] * diff1[:, 1] / diff[:, 1] + bounds[:, 0],
                )

            inside.append(np.sum(test) % 2 == 1)

        return np.array(inside)

    def bound_lines_meshes(self, radius=0.001, color=(0, 0, 0)):
        lines = self.bound_lines
        meshes = [line.get_mesh(radius=radius) for line in lines]
        [mesh.paint_uniform_color(color) for mesh in meshes]
        return meshes

    def set_bounds(self, bounds, flatten=True, convex=True):
        """Flatten points according to plane model, get projections of
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

        if bounds.shape[1] != 3:
            raise ValueError("Invalid shape of 'bounds' array.")

        if flatten:
            bounds = self.flatten_points(bounds)
        if np.any(np.isnan(bounds)):
            raise ValueError("NaN found in points")

        # self._vertices = np.array([])
        # self._triangles = np.array([])
        # self._convex = True

        projections = self.get_projections(bounds)

        try:
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

        except QhullError:
            warnings.warn(
                "Convex hull failed, bounds probably not valid. "
                "No bounds have been set."
            )

    def add_bound_points(self, new_bound_points, flatten=True):
        """Add points to current bounds.

        Parameters
        ----------
        new_bound_points : N x 3 np.array
            New points to be added.
        flatten : bool, optional
            If False, does not flatten points

        """
        if self.is_convex is not True:
            warn("Non convex plane detected, not implemented yet.")

        else:
            if flatten:
                new_bound_points = self.flatten_points(new_bound_points)
            bounds = np.vstack([self.bounds, new_bound_points])
            self.set_bounds(bounds, flatten=False)

    def intersection_bounds(self, other, within_segment=True, eps=1e-3):
        """Calculates intersection point between bounding lines.

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
            p = l1.point_from_intersection(l2, within_segment=within_segment, eps=eps)
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
        indices = TriangleMesh.simplify_loop_with_angle(
            self.bounds, range(len(self.bounds)), angle_colinear, colinear_recursive
        )

        bounds_new = self.bounds[indices]
        self.set_bounds(bounds_new, flatten=False, convex=self.is_convex)

    def contract_bounds(self, points=None, contract_holes=True):
        """
        Replace each plane bound with the closest point on the input points.
        If no input is used, use internal inlier points.

        Parameters
        ----------
        points : N x 3 array, optional
            Points used to replace bounds. If None is given, use inliers.
        contract_holes : boolean,
        """
        if points is None:
            points = self.inlier_points
            if len(points) == 0:
                raise RuntimeError(
                    "Plane has no inliers, and no input points were given."
                )

        indices = []
        for p in self.bounds:
            indices.append(PointCloud([p]).find_closest_points_indices(points)[1][0])

        indices_unique = []
        for i in indices:
            if i in indices_unique:
                continue
            indices_unique.append(i)

        bounds_new = points[indices_unique]
        self.set_bounds(bounds_new, flatten=False, convex=self.is_convex)

    @staticmethod
    def glue_planes_with_intersections(shapes, intersections, fit_separated=False):
        """Glue shapes using intersections in a dict.

        Also returns dictionary of all intersection lines.

        See: group_shape_groups, fuse_shape_groups, find_plane_intersections, glue_nearby_planes

        Parameters
        ----------
        shapes : list of shapes
            List containing all shapes to be glued.
        intersections : dict
            Dictionary with keys of type `(i, j)` and values of type Primitive.Line.
        fit_separated : bool, optional
            If True, find projections separated for each shape. Default: True.

        Returns
        -------
        list of Line instances
            Only intersections that were actually used
        """

        # new_intersections = {}
        lines = []

        for (i, j), line in intersections.items():
            if not isinstance(shapes[i], PlaneBounded) or not isinstance(
                shapes[j], PlaneBounded
            ):
                # del intersections[i, j]
                continue

            if not shapes[i].is_convex or not shapes[j].is_convex:
                continue

            # new_intersections[i, j] = line
            lines.append(line)

            if fit_separated:
                lines_ij = [
                    line.get_line_fitted_to_projections(shapes[i].bounds),
                    line.get_line_fitted_to_projections(shapes[j].bounds),
                ]

            if not fit_separated:
                projections = np.array(
                    [
                        line.projections_limits_from_points(shapes[i].bounds),
                        line.projections_limits_from_points(shapes[j].bounds),
                    ]
                )

                points = line.points_from_projections(
                    [projections.min(axis=1).max(), projections.max(axis=1).min()]
                )

                lines_ij = [line.get_fitted_to_points(points)] * 2

            for shape, line_ in zip([shapes[i], shapes[j]], lines_ij):
                # line_ = line.get_line_fitted_to_projections(shape.bounds)
                # TODO: add vertices too?
                shape.add_bound_points([line_.beginning, line_.ending])
                shape.add_inliers([line_.beginning, line_.ending])
                # new_points = [line.beginning, line.ending]

        return lines

    # @classmethod
    # def planes_ressample_and_triangulate(cls, planes, density, radius_ratio=None):
    #     """
    #     Experimental method to fuse planes by randomly rampling new points in the
    #     planes to be fused and then finding a triangulation with these new points.

    #     Parameters
    #     ----------
    #     planes : N x 3 array, optional
    #         Points used to replace bounds. If None is given, use inliers.
    #     contract_holes : boolean,
    #     """

    #     if isinstance(planes, PlaneBounded):
    #         planes = [planes]
    #     else:
    #         for plane in planes:
    #             if not isinstance(plane, PlaneBounded):
    #                 raise ValueError("Input must one or more instances of "
    #                                  "PlaneBounded.")

    #     vertices = []
    #     for p in planes:
    #         N = int(np.ceil(density * len(p.inlier_points)))
    #     #     print(N)
    #         vertices.append(p.sample_points_uniformly(N))
    #     vertices = np.vstack(vertices)

    #     # number_of_points = int(self.surface_area * density)
    #     # vertices = [p.sample_points_uniformly(int(len(p.inlier_points) * density)) for p in planes]
    #     # vertices = [p.sample_points_density(density).points for p in planes]
    #     # vertices = np.vstack(vertices)

    #     # from pyShapeDetector.utility import fuse_shape_groups
    #     plane_fused = cls.fuse(planes, ignore_extra_data=True, force_concave=False)

    #     projections = plane_fused.get_projections(vertices)
    #     triangles = Delaunay(projections).simplices

    #     if radius_ratio is not None:
    #         circumradiuses = TriangleMesh(vertices, triangles).get_triangle_circumradius()
    #         triangles = triangles[circumradiuses < radius_ratio * np.mean(circumradiuses)]

    #     if double_triangles:
    #         triangles = np.vstack([triangles, triangles[:, ::-1]])

    #     return cls(vertices, triangles)

    # @classmethod
    # def planes_ressample_and_triangulate_gui(planes, translation_ratio = 0.055,
    #                                      double_triangles=False):

    #     meshes = [plane.get_mesh() for plane in planes]

    #     plane_fused = fuse_shape_groups(
    #         [planes], detector=None, line_intersection_eps=1e-3)[0]

    #     density = 0
    #     all_points = plane_fused.inlier_points_flattened
    #     all_projections = plane_fused.get_projections(all_points)
    #     all_triangles = Delaunay(all_projections).simplices

    #     # all_points = np.vstack(
    #     #     [p.sample_points_uniformly(density * len(p.inlier_points)) for p in planes])
    #     # all_projections = planes[0].get_projections(all_points)
    #     # all_triangles = Delaunay(all_projections).simplices
    #     # vertices = all_points

    #     # perimeters = util.get_triangle_perimeters(all_points, triangles)
    #     circumradius = TriangleMesh(all_points, all_triangles).get_triangle_circumradius()
    #     # mean_circumradius = np.mean(circumradius)

    #     global data
    #     # viss = 0

    #     window_name = """
    #         (W): increase cutout limit.
    #         (S): decrease cutout limit.
    #         (A): add point density.
    #         (D): reduce point density.
    #         """

    #     data = {
    #         'original_vertices': all_points,
    #         'original_triangles': all_triangles,
    #         'density': density,
    #         'mean_circumradius': np.mean(circumradius),
    #         'circumradius': circumradius,
    #         'vertices': all_points,
    #         'triangles': all_triangles,
    #         'radius_ratio': 1,
    #         'color': (1, 0, 0),
    #         'translation': translation_ratio * plane_fused.normal,
    #         'plotted': False}

    #     triangles = data['triangles'][
    #         data['circumradius'] < data['radius_ratio'] * data['mean_circumradius']]
    #     data['current'] = TriangleMesh(all_points, triangles)
    #     data['current'].paint_uniform_color(data['color'])
    #     data['current'].translate(data['translation'])
    #     data['lines'] = TriangleMesh(data['vertices']+data['translation'], data['triangles']).get_triangle_LineSet()

    #     def update_geometries(vis):
    #         global data
    #         print(f"density: {data['density']}, radius_ratio: {data['radius_ratio']}")

    #         triangles = data['triangles'][
    #             data['circumradius'] < data['radius_ratio'] * data['mean_circumradius']]

    #         data['old'] = data['current']
    #         data['current'] = TriangleMesh(data['vertices'], triangles)
    #         data['current'].paint_uniform_color(data['color'])
    #         data['current'].translate(data['translation'])

    #         vis.add_geometry(data['current'])
    #         vis.remove_geometry(data['old'])

    #         return False

    #     def ratio_increase(vis):
    #         global data
    #         data['radius_ratio'] += 0.1
    #         # data['radius_ratio'] *= 2
    #         update_geometries(vis)

    #     def ratio_decrease(vis):
    #         global data
    #         data['radius_ratio'] -= 0.1
    #         # data['radius_ratio'] /= 2
    #         data['radius_ratio'] = max(data['radius_ratio'], 0)
    #         update_geometries(vis)

    #     def update_density(vis):
    #         global data

    #         if data['density'] == 0:
    #             data['vertices'] = data['original_vertices']
    #             data['triangles'] = data['original_triangles']
    #         else:
    #             data['vertices'], data['triangles'] = PlaneBounded.planes_ressample_and_triangulate(
    #                 planes, data['density'])

    #         data['circumradius'] = TriangleMesh(
    #             data['vertices'], data['triangles']).get_triangle_circumradius
    #         data['mean_circumradius'] = np.mean(circumradius)

    #         vis.remove_geometry(data['lines'])
    #         data['lines'] = TriangleMesh(data['vertices']+data['translation'], data['triangles']).get_triangle_LineSet()
    #         vis.add_geometry(data['lines'])

    #     def density_increase(vis):
    #         global data
    #         data['density'] += 0.1
    #         update_density(vis)
    #         update_geometries(vis)

    #     def density_decrease(vis):
    #         global data
    #         data['density'] -= 0.1
    #         data['density'] = max(data['density'], 0.0)
    #         update_density(vis)
    #         update_geometries(vis)

    #     key_to_callback = {}
    #     key_to_callback[ord("W")] = ratio_increase
    #     key_to_callback[ord("S")] = ratio_decrease
    #     key_to_callback[ord("A")] = density_decrease
    #     key_to_callback[ord("D")] = density_increase
    #     # key_to_callback[ord("D")] = remove_last
    #     # key_to_callback[ord("R")] = load_render_option
    #     # key_to_callback[ord(",")] = capture_depth
    #     # key_to_callback[ord(".")] = capture_image

    #     draw_geometries_with_key_callbacks(
    #         meshes+[data['current'], data['lines']],
    #         # [data['current'], data['lines']],
    #         key_to_callback,
    #         window_name = window_name,
    #         # mesh_show_wireframe=True
    #         )

    #     triangles = data['triangles'][
    #         data['circumradius'] < data['radius_ratio'] * data['mean_circumradius']]

    #     if double_triangles:
    #         triangles = np.vstack([triangles, triangles[:, ::-1]])

    #     out_data = {
    #         'density': data['density'],
    #         'radius_ratio': data['radius_ratio'],
    #         'vertices': data['vertices'],
    #         'triangles': triangles}

    #     return out_data
