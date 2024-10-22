#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:15:09 2024

@author: ebernardes
"""
import warnings
import numpy as np

from pyShapeDetector.geometry import (
    AxisAlignedBoundingBox,
    OrientedBoundingBox,
)
from .plane import Plane, _get_vertices_from_vectors, _get_vx_vy
from pyShapeDetector.utility import get_area_with_shoelace, midrange


class PlaneRectangular(Plane):
    """
    PlaneRectangular primitive.

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
    dimensions
    center
    vertices
    vertices
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
    set_parallel_vectors
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
    detect_and_insert_holes

    contains_projections
    set_parallel_vectors
    set_center
    """

    _name = "rectangular plane"
    _parallel_vectors = None
    _center = None
    _convex = True

    @property
    def surface_area(self):
        """Surface area of rectangular plane."""

        surface_area = np.prod(np.linalg.norm(self.parallel_vectors, axis=1))
        for hole in self.holes:
            surface_area -= get_area_with_shoelace(hole.vertices_projections)

        return surface_area

    @property
    def parallel_vectors(self):
        return self._parallel_vectors

    @property
    def dimensions(self):
        return np.linalg.norm(self.parallel_vectors, axis=1)

    @property
    def center(self):
        return self._center

    @property
    def vertices(self):
        v1, v2 = self.parallel_vectors
        return self.center + _get_vertices_from_vectors(v1, v2, assert_rect=False)

    @property
    def vertices_projections(self):
        return self.get_projections(self.vertices)

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

    def __init__(self, model, vectors=None, center=None, decimals=None):
        """
        Parameters
        ----------
        model : Primitive or list of 4 values
            Shape defining plane
        vectors : arraylike of shape (2, 3), optional
            Vectors ortoghonal to plane's normal. Default: None.
        center : arraylike of shape (1, 3), optional
            Center of plane. Default: None.
        decimals : int, optional
            Number of decimal places to round to (default: 0). If
            decimals is negative, it specifies the number of positions to
            the left of the decimal point. Default: None.

        Raises
        ------
        ValueError
            If number of parameters is incompatible with the model of the plane
        """
        super().__init__(model, decimals)

        if vectors is None or center is None:
            if isinstance(model, PlaneRectangular):
                vectors_ = model._parallel_vectors
                center_ = model._center

            else:
                if isinstance(model, Plane) and hasattr(model, "vertices"):
                    points = model.vertices

                elif isinstance(model, Plane) and model.has_inliers:
                    warnings.warn("No vertices or input vectors/center, using inliers.")
                    points = model.inliers.points

                else:
                    warnings.warn(
                        "No inliers, vertices or input vectors and center, returning square plane, possibly centered at [0, 0, 0]."
                    )
                    points = self.get_square_plane(1).vertices

                vectors_, center_ = self.get_rectangular_vectors_from_points(
                    points=points,
                    return_center=True,
                    use_PCA=True,
                    normalized=False,
                )

            if vectors is None:
                vectors = vectors_

            if center is None:
                center = center_

        self.set_parallel_vectors(vectors)
        self.set_center(center, flatten=True)

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
        unbounded = Plane.random(scale=scale, decimals=decimals)
        vy = np.random.random(3)
        vx = np.cross(vy, unbounded.normal)
        vy = np.cross(unbounded.normal, vx)
        vx *= scale * np.random.random() / np.linalg.norm(vx)
        vy *= scale * np.random.random() / np.linalg.norm(vy)
        center = np.random.random(3) * scale
        return unbounded.get_rectangular_plane([vx, vy], center)

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
        vectors, center = plane.get_rectangular_vectors_from_points(
            points=points, return_center=True, use_PCA=True, normalized=False
        )
        return plane.get_rectangular_plane(vectors, center)

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

        vectors = self.parallel_vectors
        center = self.center
        v_norms = np.linalg.norm(vectors, axis=1)
        v_normalized = vectors / v_norms[np.newaxis].T

        R = np.vstack([v_normalized, np.cross(v_normalized[0], v_normalized[1])])
        extent = np.hstack([v_norms, 0])

        oriented_bbox = OrientedBoundingBox(center=center, R=R.T, extent=extent)
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
        from .planebounded import PlaneBounded

        plane_bounded = PlaneBounded(self.model, self.vertices, convex=True)
        plane_bounded._holes = self._holes
        return plane_bounded.get_mesh(**options)

    def __copy_atributes__(self, shape_original):
        super().__copy_atributes__(shape_original)
        self._parallel_vectors = shape_original._parallel_vectors.copy()
        self._center = shape_original._center.copy()

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
        self._center = self._center + translation

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

        self._center = rotation.apply(self._center)
        self._parallel_vectors = rotation.apply(self._parallel_vectors)

    def __put_attributes_in_dict__(self, data, save_inliers=True):
        super().__put_attributes_in_dict__(data, save_inliers=save_inliers)
        data["parallel_vectors"] = self.parallel_vectors.flatten().tolist()
        data["center"] = self.center.tolist()

    def __get_attributes_from_dict__(self, data):
        super().__get_attributes_from_dict__(data)
        vectors = np.array(data["parallel_vectors"]).reshape((2, 3))
        center = np.array(data["center"])
        self.set_parallel_vectors(vectors)
        self.set_center(center)

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
        from .planebounded import PlaneBounded

        shapes_bounded = [PlaneBounded(shape.model, shape.vertices) for shape in shapes]

        plane_bounded = PlaneBounded.fuse(shapes_bounded, detector, ignore_extra_data)

        vectors, center = plane_bounded.get_rectangular_vectors_from_points(
            return_center=True,
            use_PCA=True,
            normalized=False,
        )

        shape = PlaneRectangular(plane_bounded.model, vectors, center)

        if not ignore_extra_data:
            shape._inliers = plane_bounded._inliers
            # TODO: This is ugly
            shape.color = np.mean([s.color for s in shapes], axis=0)
            shape.metrics = plane_bounded.metrics

        return shape

    @classmethod
    def from_vectors_center(cls, vectors, center=(0, 0, 0)):
        """
        Creates plane from two vectors representing rectangle and center point.

        Parameters
        ----------
        vectors : arraylike of shape (2, 3)
            The two orthogonal unit vectors defining the rectangle plane.
        center : arraylike of length 3, optional.
            Center of rectangle. Default: (0, 0, 0).

        Returns
        -------
        Plane
            Generated shape.
        """
        plane = super().from_vectors_center(vectors, center)
        return cls(plane, vectors, center)

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

        from .planebounded import PlaneBounded

        bounded = PlaneBounded(self.model, self.vertices)
        return bounded.contains_projections(points, input_is_2D)

    def set_parallel_vectors(self, vectors=None):
        """Sets parallel vectors of plane.

        - If vectors are an empty array, removes internal call to it.

        - If no vectors are given as input but the shape has inliers, uses inliers
        to calculate rectangular vectors.

        - If no vectors are given as input and the shape has no inliers, find to random
        vectors forming orthogonal base with normal vector.

        Parameters
        ----------
        vectors : arraylike of shape (2, 3), optional
            Custom vectors defining array. Default: None.

        """
        vectors = np.array(vectors)

        if vectors is not None:
            vectors = np.array(vectors)

            if vectors.shape != (2, 3):
                raise ValueError(
                    f"Parallel vectors must be an arraylike of shape (2, 3), got {vectors}."
                )

            norms = np.linalg.norm(vectors, axis=1)
            if any(abs((vectors / norms[np.newaxis].T).dot(self.normal)) > 1e-5):
                raise ValueError(
                    "Parallel vectors must be orthogonal to normal vector."
                )

            # removing projections to have two ortoghonal vectors
            dot = np.dot(vectors[0], vectors[1])
            if dot > 1e-5:
                warnings.warn(
                    "Parallel vectors are not orthogonal, returning minimum set of orthogonal vectors."
                )
                if norms[0] > norms[1]:
                    vectors[1] = vectors[1] - dot * vectors[0] / (norms[0] ** 2)
                else:
                    vectors[0] = vectors[0] - dot * vectors[1] / (norms[1] ** 2)

            self._parallel_vectors = vectors

        elif self.has_inliers:
            self._parallel_vectors = self.get_rectangular_vectors_from_points(
                normalized=False
            )

        else:
            warnings.warn("No input vectors not inliers, setting orthonormal vectors.")
            vx, vy = _get_vx_vy(self.normal)
            self._parallel_vectors = np.vstack([vx, vy])

        # sanity test
        assert self.parallel_vectors is not None
        v1, v2 = self._parallel_vectors
        assert abs(v1.dot(v2)) < 1e-7

    def set_center(self, center=None, flatten=True):
        """Set center of plane.

        - If center is an empty array, removes it.

        - If no vectors are given as input but the shape has inliers, uses inliers
        to calculate center.

        Parameters
        ----------
        vectors : arraylike of shape (2, 3), optional
            Custom vectors defining array. Default: None.
        flatten : bool, optional
            If False, does not flatten center point. Default: True.

        """
        if center is not None:
            center = np.array(center)
            if center.shape != (3,):
                raise ValueError(
                    f"Center must be an arraylike with length 3, got {center}."
                )

        elif self.has_inliers:
            points = self.inliers.points
            center = midrange(points)

        else:
            warnings.warn("No input center not inliers, setting center to centroid")
            center = self.centroid

        if flatten:
            center = self.flatten_points(center)

        error = self.get_distances(center)
        if error > 1e-7:
            raise RuntimeError(
                "PlaneRectangular instance generated with vertices far away "
                "from plane model, the center is probably incompatible."
            )

        self._center = center
