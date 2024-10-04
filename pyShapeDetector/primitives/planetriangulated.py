#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:15:09 2024

@author: ebernardes
"""
import warnings
from itertools import permutations, product, combinations
import numpy as np

# from scipy.spatial import ConvexHull, Delaunay
# from scipy.spatial.transform import Rotation
# from open3d.geometry import AxisAlignedBoundingBox
from pyShapeDetector.geometry import PointCloud, TriangleMesh, AxisAlignedBoundingBox

# from pyShapeDetector.utility import (
#     # fuse_vertices_triangles,
#     # get_triangle_boundary_indexes,
#     get_loop_indexes_from_boundary_indexes,
#     )
# from .primitivebase import Primitive
from .plane import Plane
from .planebounded import _unflatten


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

    vertices
    triangles

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
    get_unbounded_plane
    get_bounded_plane
    get_triangulated_plane
    get_triangulated_plane_from_grid
    get_triangulated_plane_from_alpha_shape
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
    detect_and_insert_holes

    closest_vertices
    set_vertices_triangles
    from_plane_with_mesh
    get_bounded_planes_from_boundaries
    """

    _name = "triangulated plane"
    _vertices = np.array([])
    _triangles = np.array([])
    # TODO: maybe set _convex to None as it cannot be known
    _convex = False

    @property
    def surface_area(self):
        """Surface area of triangulated plane."""

        triangle_projections = self.get_projections(self.vertices)[self.triangles]
        diff = np.diff(triangle_projections, axis=1)
        areas = abs(np.cross(diff[:, 0, :], diff[:, 1, :])) * 0.5
        surface_area = sum(areas)

        return surface_area

    @property
    def vertices(self):
        return self._vertices

    @property
    def triangles(self):
        return self._triangles

    def __init__(self, model, vertices=None, triangles=None, decimals=None):
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
                mesh = model.mesh
            else:
                warnings.warn("No input vertices/triangles, returning square plane")
                mesh = self.get_square_plane(1).mesh
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            flatten = False

        elif vertices is not None or triangles is not None:
            pass
        elif vertices is None or triangles is None:
            raise ValueError(
                "Either 'vertices' and 'triangles' should be given, or one of them."
            )

        # super().__init__(model, decimals)
        self.set_vertices_triangles(vertices, triangles, flatten=flatten)

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
        return PlaneTriangulated.from_bounded_plane(plane.get_bounded_plane(points))

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
        mesh = TriangleMesh(self.vertices, self.triangles)

        return mesh

    def __copy_atributes__(self, shape_original):
        super().__copy_atributes__(shape_original)
        self._vertices = shape_original._vertices.copy()
        self._triangles = shape_original._triangles.copy()

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
        super().rotate(rotation)

        self._vertices = rotation.apply(self._vertices, rotate_inliers=rotate_inliers)

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

    def __put_attributes_in_dict__(self, data, save_inliers=True):
        super().__put_attributes_in_dict__(data, save_inliers=save_inliers)

        # additional PlaneTriangulated related data:
        data["vertices"] = self.vertices.flatten().tolist()
        data["triangles"] = self.triangles.flatten().tolist()

    def __get_attributes_from_dict__(self, data):
        super().__get_attributes_from_dict__(data)

        # additional PlaneTriangulated related data:
        self.set_vertices_triangles(
            _unflatten(data["vertices"]), _unflatten(data["triangles"])
        )

    @staticmethod
    def fuse(
        shapes,
        detector=None,
        ignore_extra_data=False,
        line_intersection_eps=1e-3,
        **extra_options,
    ):
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
        PlaneTriangulated
            Averaged PlaneTriangulated instance.
        """
        plane_unbounded = Plane.fuse(shapes, detector, ignore_extra_data)

        vertices_list = [plane_unbounded.flatten_points(s.vertices) for s in shapes]
        triangles_list = [s.triangles for s in shapes]

        vertices, triangles = TriangleMesh.fuse_vertices_triangles(
            vertices_list, triangles_list
        )

        shape = PlaneTriangulated(plane_unbounded.model, vertices, triangles)

        if not ignore_extra_data:
            shape._inliers = plane_unbounded._inliers
            shape.color = plane_unbounded.color
            shape.metrics = plane_unbounded.metrics

        return shape

        # force_concave = extra_options.get('force_concave', True)
        # ressample_density = extra_options.get('ressample_density', 1.5)
        # ressample_radius_ratio = extra_options.get('ressample_radius_ratio', 1.2)

        # if len(shapes) == 1:
        #     return shapes[0]
        # elif isinstance(shapes, Primitive):
        #     return shapes

        # shape = Plane.fuse(shapes, detector, ignore_extra_data)

        # is_convex = np.array([shape.is_convex for shape in shapes])
        # all_convex = is_convex.all()
        # if not all_convex and is_convex.any():
        #     force_concave = True
        #     # raise ValueError("If 'force_concave' is False, PlaneBounded "
        #     #                  "instances should either all be convex or "
        #     #                  "all non convex.")

        # if not ignore_extra_data:
        #     if force_concave:
        #         vertices, triangles = planes_ressample_and_triangulate(
        #             shapes, ressample_density, ressample_radius_ratio, double_triangles=False)
        #         shape.set_vertices_triangles(vertices, triangles)

        #     elif not all_convex:
        #         vertices = [shape.vertices for shape in shapes]
        #         triangles = [shape.triangles for shape in shapes]
        #         vertices, triangles = fuse_vertices_triangles(vertices, triangles)
        #         shape.set_vertices_triangles(vertices, triangles)

        #     else:
        #         vertices = np.vstack([shape.vertices for shape in shapes])
        #         shape.set_vertices(vertices)

        #         intersections = []
        #         for plane1, plane2 in combinations(shapes, 2):
        #             points = plane1.intersection_vertices(plane2, True, eps=line_intersection_eps)
        #             if len(points) > 0:
        #                 intersections.append(points)

        #         # temporary hack, saving intersections for mesh generation
        #         if len(intersections) > 0:
        #             shape._fusion_intersections = np.vstack(intersections)

        # return shape

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
        from .planebounded import PlaneBounded

        plane_bounded = PlaneBounded.from_vectors_center(vectors, center)
        return cls(plane_bounded)

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

        if not isinstance(other_plane, PlaneTriangulated):
            raise ValueError("Only implemented with other instances of PlaneBounded.")

        closest_points, distances = PointCloud.find_closest_points(
            self.vertices, other_plane.vertices, n
        )

        return closest_points, distances

    def contains_projections(self, points):
        """For each point in points, check if its projection on the plane lies
        inside of the plane's vertices.

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        array of booleans
            True for points whose projection lies in plane's vertices
        """
        raise RuntimeError(
            "'contains_projections' not implemented for PlaneTriangulated instances."
        )

    def set_vertices_triangles(self, vertices, triangles, flatten=True):
        """Flatten points according to plane model, get projections of
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
        # self._mesh = None
        vertices = np.asarray(vertices).copy()
        triangles = np.asarray(triangles).copy()

        if vertices.shape[1] != 3 or triangles.shape[1] != 3:
            raise ValueError("Invalid shape of 'vertices' and/or 'triangles' array.")

        if not all([x.is_integer() for x in triangles.flatten()]):
            raise ValueError("All elements of 'triangles' must be integers.")

        if (triangles >= len(vertices)).any() or (triangles < 0).any():
            raise ValueError(
                "Each element in 'triangles' should be an integer"
                " between 0 and len(vertices) - 1."
            )

        if flatten:
            vertices = self.flatten_points(vertices)
        if np.any(np.isnan(vertices)):
            raise ValueError("NaN found in points")

        # self._vertices = np.array([])
        # self._vertices_indices = np.array([])
        self._vertices = vertices
        self._triangles = triangles
        # self._mesh = self.get_mesh()
        # self._convex = False

        error = np.linalg.norm(self.get_distances(self.vertices))
        if error > 1e-7:
            raise RuntimeError("Vertices far away from plane model.")

    @staticmethod
    def from_plane_with_mesh(plane):
        """Convert plane instance's mesh into PlaneTriangulated instance by

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

    def get_bounded_planes_from_boundaries(
        self,
        detect_holes=False,
        add_inliers=True,
        angle_colinear=0,
        min_point_dist=0,
        max_point_dist=np.inf,
        contract_boundary=False,
        min_inliers=1,
    ):
        """Convert PlaneTriangulated instance into list of non-convex
        PlaneBounded instances.

        If `detect_holes` is set to True, when planes are detected inside some
        other, they will be added as holes instead.

        If a small positive `angle_colinear` value is given, two subsequent
        boundary lines will be fused into one when they are almost colinear.

        Parameters
        ----------
        detect_holes : boolean, optional
            If True, try to detect holes. Default: False.
        add_inliers : boolean, optional
            If True, add inlier points.
        angle_colinear : float, optional
            Small angle value for assuming two lines are colinear. Default: 0
        min_point_dist : float, optional
            If the simplified distance is bigger than this value, simplify
            regardless of angle. Default: 0.
        max_point_dist : float, optional
            If the simplified distance is bigger than this value, do not
            simplify. Default: np.inf
        contract_boundary : boolean, optional
            If True, contract vertices to closest inlier points. Default: False.
        min_inliers : int, optional
            If add_inliers is True, remove planes with less inliers than this
            value. Default: 1.

        Returns
        -------
        list of PlaneBounded instances
        """
        if not isinstance(min_inliers, int) or (min_inliers < 1):
            raise ValueError(
                f"min_inliers must be a positive integer, got {min_inliers}."
            )

        from .planebounded import PlaneBounded
        from .line import Line

        boundary_indexes = self.mesh.get_triangle_boundary_indexes()

        loop_indexes = TriangleMesh.get_loop_indexes_from_boundary_indexes(
            boundary_indexes
        )

        if angle_colinear is not None:
            for i in range(len(loop_indexes)):
                loop_indexes[i] = Line.get_simplified_loop_indices(
                    self.vertices,
                    angle_colinear=angle_colinear,
                    min_point_dist=min_point_dist,
                    max_point_dist=max_point_dist,
                    loop_indexes=loop_indexes[i],
                )

        planes = [
            PlaneBounded(self, self.vertices[loop], convex=False)
            for loop in loop_indexes
        ]

        if contract_boundary:
            for p in planes:
                p.contract_boundary(self.inliers.points)

        idx = np.argsort([p.surface_area for p in planes])[::-1]
        planes = np.array(planes)[idx].tolist()

        if detect_holes:
            Plane.detect_and_insert_holes(planes)

        if add_inliers and self.has_inliers:
            pcd = self.inliers
            projections = self.get_projections(pcd.points)

            for plane in planes:
                if len(projections) == 0:
                    break
                inside = plane.contains_projections(projections, input_is_2D=True)
                inside_idx = np.where(inside)[0]
                plane.set_inliers(pcd.select_by_index(inside_idx))
                pcd = pcd.select_by_index(inside_idx, invert=True)
                projections = projections[~inside]

            num_inliers = np.array([len(p.inlier_points) for p in planes])
            planes = np.array(planes)[num_inliers > min_inliers].tolist()

        return planes
