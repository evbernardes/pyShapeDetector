#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:15:09 2024

@author: ebernardes
"""
import warnings
from itertools import product
import numpy as np
from scipy.spatial import QhullError, ConvexHull, Delaunay

from pyShapeDetector.geometry import PointCloud, TriangleMesh, AxisAlignedBoundingBox
from .plane import Plane, _get_vertices_from_vectors

from pyShapeDetector.utility import check_vertices_clockwise, get_area_with_shoelace

from importlib.util import find_spec

if has_mapbox_earcut := find_spec("mapbox_earcut") is not None:
    from mapbox_earcut import triangulate_float32


def _unflatten(values):
    values = np.array(values)
    if values.ndim == 1:
        values = values.reshape([len(values) // 3, 3])
    return values


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

    is_clockwise
    vertices
    vertices_indices
    vertices_projections
    vertices_lines
    vertices_LineSet

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
    get_obj_description
    fuse
    group_similar_shapes
    fuse_shape_groups
    fuse_similar_shapes

    from_normal_dist
    from_normal_point
    from_vectors_center
    add_holes
    remove_hole
    get_fused_holes
    intersect
    closest_vertices
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

    closest_vertices
    contains_projections
    bound_lines_meshes
    set_vertices
    add_bound_points
    intersection_vertices
    simplify_vertices
    contract_boundary
    glue_planes_with_intersections
    """

    _name = "bounded plane"
    _vertices_indices = np.array([])
    _vertices = np.array([])
    _vertices_projections = np.array([])
    _convex = True
    _is_clockwise = None

    @property
    def surface_area(self):
        """Surface area of bounded plane."""

        surface_area = get_area_with_shoelace(self.vertices_projections)
        for hole in self.holes:
            surface_area -= get_area_with_shoelace(hole.vertices_projections)

        return surface_area

    @property
    def is_clockwise(self):
        return self._is_clockwise

    @property
    def vertices(self):
        return self._vertices

    @property
    def vertices_indices(self):
        """Indices of points corresponding to vertices."""
        # TODO: should take into consideration added vertices
        return self._vertices_indices

    @property
    def vertices_projections(self):
        return self._vertices_projections

    @property
    def vertices_lines(self):
        """Lines defining vertices."""
        from .line import Line

        return Line.from_vertices(self.vertices)

    @property
    def vertices_LineSet(self):
        """Lines defining vertices."""
        from .line import Line

        return Line.get_LineSet_from_list(self.vertices_lines)

    def __init__(self, model, vertices=None, convex=None, decimals=None):
        """
        Parameters
        ----------
        model : Primitive or list of 4 values
            Shape defining plane
        vertices : array_like, shape (N, 3), optional
            Points defining vertices.
        convex : bool, optinal
            If True, assumes the vertices are supposed to be convex and use
            ConvexHull. If False, assume vertices are directly given as a loop.
            Default: None (decide dynamically).
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
        _convex = True

        if vertices is None:
            if isinstance(model, Plane) and hasattr(model, "vertices"):
                vertices = model.vertices
                _convex = model.is_convex

            elif isinstance(model, Plane) and model.has_inliers:
                warnings.warn("No vertices or input vertices, using inliers.")
                vertices = model.inliers.points
                _convex = True

            else:
                warnings.warn("No vertices, input or inliers, returning square plane.")
                vertices = self.get_square_plane(1).vertices
                flatten = False
                _convex = True

        # TODO: check this
        if convex is None:
            convex = _convex

        self.set_vertices(vertices, flatten=flatten, convex=convex)

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
        return PlaneBounded(plane, points, convex=True)

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

        projections = self.vertices_projections
        holes = self._holes

        if self.is_convex:
            if len(holes) >= 0:
                points_holes = [self.flatten_points(hole.vertices) for hole in holes]
                points = np.vstack([self.vertices] + points_holes)
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
                all_points = [self.vertices] + [h.vertices for h in self.holes]
                points = np.vstack(all_points)
                projections = self.get_projections(points)
                rings = [len(self.vertices)]
                for hole in self.holes:
                    rings.append(rings[-1] + len(hole.vertices))

                triangles = triangulate_float32(projections, rings).reshape(-1, 3)

            else:
                warnings.warn(
                    "'mapbox_earcut' module not available, triangulation might not work properly."
                )

                if not self.is_clockwise:
                    projections = projections[::-1]

                if not self.is_hole and len(self.holes) > 0:
                    from .plane import _fuse_loops

                    fused_hole = self.get_fused_holes()
                    projections = _fuse_loops(
                        projections, fused_hole.vertices_projections
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
    #         points = self.vertices
    #         projections = self.vertices_projections
    #         # idx_intersections_sorted = []
    #     else:
    #         points = np.vstack([self.vertices, self._fusion_intersections])
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
    #         points_holes = [self.flatten_points(hole.vertices) for hole in holes]
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
        self._vertices_indices = shape_original.vertices_indices.copy()
        self._vertices = shape_original._vertices.copy()
        self._vertices_projections = shape_original._vertices_projections.copy()
        self._is_clockwise = check_vertices_clockwise(self._vertices_projections)
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
        self._vertices = self._vertices + translation

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

        self._vertices = rotation.apply(self._vertices)

    def __put_attributes_in_dict__(self, data, save_inliers=True):
        super().__put_attributes_in_dict__(data, save_inliers=save_inliers)

        # additional PlaneBounded related data:
        data["vertices"] = self.vertices.flatten().tolist()
        data["_fusion_intersections"] = self._fusion_intersections.tolist()

        if len(self.holes) > 0:
            all_holes_vertices = (
                np.vstack([h.vertices for h in self.holes]).flatten().tolist()
            )
        else:
            all_holes_vertices = []
        data["hole_vertices"] = all_holes_vertices
        data["hole_lengths"] = [len(h.vertices) for h in self.holes]
        data["convex"] = self.is_convex
        data["hole_convex"] = [h.is_convex for h in self.holes]

    def __get_attributes_from_dict__(self, data):
        super().__get_attributes_from_dict__(data)

        # additional PlaneBounded related data:
        convex = data.get("convex", True)
        # Compatibility for when 'vertices' was still called 'bounds':
        try:
            vertices = data["vertices"]
        except KeyError:
            vertices = data["bounds"]
        self.set_vertices(_unflatten(vertices), flatten=False, convex=convex)
        self._fusion_intersections = np.array(data["_fusion_intersections"])

        try:
            hole_vertices = data["hole_vertices"]
        except KeyError:
            hole_vertices = data["hole_bounds"]

        if "file_version" in data is not None and data["file_version"] >= 2:
            hole_vertices = _unflatten(hole_vertices)

            # if hole_vertices.ndim != 1:
            #     raise RuntimeError(
            #         "Invalid hole_vertices shape for file of version {data['file_version']}."
            #     )

            if "hole_lengths" not in data:
                raise RuntimeError(
                    "File of version {data['file_version']} should have 'hole_lengths'."
                )

            hole_vertices_separated = []
            count = 0
            for length in data["hole_lengths"]:
                hole_vertices_separated.append(hole_vertices[count : count + length])
                count += length
            hole_vertices = hole_vertices_separated
        try:
            hole_convex = data["hole_convex"]
        except KeyError:
            hole_convex = [True] * len(hole_vertices)

        holes = []
        for vertices, convex in zip(hole_vertices, hole_convex):
            holes.append(PlaneBounded(self.model, _unflatten(vertices), convex=convex))

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
        min_bound = np.min(self.vertices, axis=0)
        max_bound = np.max(self.vertices, axis=0)
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
        vertices = np.vstack([s.vertices for s in shapes])

        if not np.all([s.is_convex for s in shapes]):
            warnings.warn(
                "Non-convex PlaneBounded instance detected, fused "
                "plane will be convex."
            )

        shape = PlaneBounded(plane_unbounded.model, vertices)

        if not ignore_extra_data:
            shape._inliers = plane_unbounded._inliers
            shape.color = plane_unbounded.color
            shape.metrics = plane_unbounded.metrics

        return shape

    @classmethod
    def from_vectors_center(cls, vectors, center):
        """
        Creates plane from two vectors representing rectangle and center point.

        Parameters
        ----------
        vectors : arraylike of shape (2, 3)
            The two orthogonal unit vectors defining the rectangle plane.
        center : arraylike of length 3
            Center of rectangle.

        Returns
        -------
        Plane
            Generated shape.
        """
        plane = super().from_vectors_center(vectors, center)
        vertices = center + _get_vertices_from_vectors(
            vectors[0], vectors[1], assert_rect=False
        )
        return cls(plane, vertices)

    def closest_vertices(self, other_plane, n=1):
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
            self.vertices, other_plane.vertices, n
        )

        return closest_points, distances

    def contains_projections(self, points, input_is_2D=False):
        """For each point in points, check if its projection on the plane lies
        inside of the plane's vertices.

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
            True for points whose projection lies in plane's vertices
        """

        # inside = np.array([True] * len(points))
        inside = []

        if PointCloud.is_instance_or_open3d(points):
            points = np.asarray(points.points)

        points = np.atleast_2d(points)
        if points.shape[1] == 3:
            if input_is_2D:
                raise ValueError("For 3D input points, input_is_2D must be False.")
            projections = self.get_projections(points)
        elif points.shape[1] == 2:
            if not input_is_2D:
                raise ValueError("For 2D input projections, input_is_2D must be True.")
            projections = points

        # N = len(self.vertices_projections)
        vertices = self.vertices_projections
        vertices_shifted = np.roll(vertices, -1, axis=0)
        diff = vertices_shifted - vertices
        for projection in projections:
            diff1 = projection - vertices
            diff2 = projection - vertices_shifted

            with warnings.catch_warnings():
                # Divisions by zero lead to infs that give correct comparisons
                warnings.simplefilter("ignore")
                test = np.logical_and(
                    (diff1[:, 1] < 0) != (diff2[:, 1] < 0),
                    projection[0]
                    < diff[:, 0] * diff1[:, 1] / diff[:, 1] + vertices[:, 0],
                )

            inside.append(np.sum(test) % 2 == 1)

        return np.array(inside)

    def bound_lines_meshes(self, radius=0.001, color=(0, 0, 0)):
        lines = self.vertices_lines
        meshes = [line.get_mesh(radius=radius) for line in lines]
        [mesh.paint_uniform_color(color) for mesh in meshes]
        return meshes

    def set_vertices(self, vertices, flatten=True, convex=True):
        """Flatten points according to plane model, get projections of
        flattened points in the model and compute its boundary using either
        the convex hull or alpha shapes.

        Parameters
        ----------
        plane : Plane
            Plane model
        vertices : array_like, shape (N, 3)
            Points corresponding to the fitted shape.
        flatten : bool, optional
            If False, does not flatten points. Default: True.
        convex : bool, optinal
            If True, assumes the vertices are supposed to be convex and use
            ConvexHull. If False, assume vertices are directly given as a loop.

        """
        self._mesh = None
        vertices = np.asarray(vertices)

        if vertices.shape[1] != 3:
            raise ValueError("Invalid shape of 'vertices' array.")

        if flatten:
            vertices = self.flatten_points(vertices)
        if np.any(np.isnan(vertices)):
            raise ValueError("NaN found in points")

        # self._vertices = np.array([])
        # self._triangles = np.array([])
        # self._convex = True

        projections = self.get_projections(vertices)

        try:
            if convex:
                chull = ConvexHull(projections)
                self._vertices_indices = chull.vertices
                self._vertices = vertices[chull.vertices]
                self._vertices_projections = projections[chull.vertices]
            else:
                self._vertices_indices = np.array(range(len(vertices)))
                self._vertices = vertices
                self._vertices_projections = projections

            self._is_clockwise = check_vertices_clockwise(self._vertices_projections)
            self._convex = convex

        except QhullError:
            warnings.warn(
                "Convex hull failed, vertices probably not valid. "
                "No vertices have been set."
            )

        if not flatten:
            error = self.get_distances(self.vertices)
            if np.any(error > 1e-7):
                raise RuntimeError("Vertices far away from plane model.")

    def add_bound_points(self, new_vertices_points, flatten=True):
        """Add points to current vertices.

        Parameters
        ----------
        new_vertices_points : N x 3 np.array
            New points to be added.
        flatten : bool, optional
            If False, does not flatten points

        """
        if self.is_convex is not True:
            warnings.warn("Non convex plane detected, not implemented yet.")

        else:
            if flatten:
                new_vertices_points = self.flatten_points(new_vertices_points)
            vertices = np.vstack([self.vertices, new_vertices_points])
            self.set_vertices(vertices, flatten=False)

    def intersection_vertices(self, other, within_segment=True, eps=1e-3):
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

        if len(self.vertices) == 0 or len(other.vertices) == 0:
            raise ValueError("Both planes must have vertices.")

        lines = self.vertices_lines
        lines_other = other.vertices_lines

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

    def simplify_vertices(
        self, angle_colinear=0, min_point_dist=0, max_point_dist=np.inf
    ):
        """
        Simplify vertices be removing some if they are colinear (or almost colinear)
        or distance between them are too small.

        Parameters
        ----------
        angle_colinear : float, optional
            Small angle value for assuming two lines are colinear. Default: 0
        min_point_dist : float, optional
            If the simplified distance is bigger than this value, simplify
            regardless of angle. Default: 0.
        max_point_dist : float, optional
            If the simplified distance is bigger than this value, do not
            simplify. Default: np.inf
        """
        indices = TriangleMesh.simplify_loop(
            self.vertices,
            range(len(self.vertices)),
            angle_colinear=angle_colinear,
            min_point_dist=min_point_dist,
            max_point_dist=max_point_dist,
        )

        vertices_new = self.vertices[indices]
        self.set_vertices(vertices_new, flatten=False, convex=self.is_convex)

    def contract_boundary(self, points=None, contract_holes=True):
        """
        Replace each plane bound with the closest point on the input points.
        If no input is used, use internal inlier points.

        Parameters
        ----------
        points : N x 3 array, optional
            Points used to replace vertices. If None is given, use inliers.
        contract_holes : boolean,
        """
        if points is None:
            points = self.inlier_points
            if len(points) == 0:
                raise RuntimeError(
                    "Plane has no inliers, and no input points were given."
                )

        indices = []
        for p in self.vertices:
            indices.append(PointCloud([p]).find_closest_points_indices(points)[1][0])

        indices_unique = []
        for i in indices:
            if i in indices_unique:
                continue
            indices_unique.append(i)

        vertices_new = points[indices_unique]
        self.set_vertices(vertices_new, flatten=True, convex=self.is_convex)

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
                    line.get_line_fitted_to_projections(shapes[i].vertices),
                    line.get_line_fitted_to_projections(shapes[j].vertices),
                ]

            if not fit_separated:
                projections = np.array(
                    [
                        line.projections_limits_from_points(shapes[i].vertices),
                        line.projections_limits_from_points(shapes[j].vertices),
                    ]
                )

                points = line.points_from_projections(
                    [projections.min(axis=1).max(), projections.max(axis=1).min()]
                )

                lines_ij = [line.get_fitted_to_points(points)] * 2

            for shape, line_ in zip([shapes[i], shapes[j]], lines_ij):
                # line_ = line.get_line_fitted_to_projections(shape.vertices)
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
    #         Points used to replace vertices. If None is given, use inliers.
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
