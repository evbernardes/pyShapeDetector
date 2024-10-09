#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:57:08 2023

@author: ebernardes
"""
from abc import ABC, abstractmethod
import random
import copy
import numpy as np
import open3d as o3d
from open3d.geometry import LineSet, AxisAlignedBoundingBox
from open3d.utility import Vector2iVector, Vector3dVector

from .primitivebase import Primitive
from .cylinder import Cylinder


class Line(Primitive):
    """
    Line primitive.

    Does not define a surface, but implement useful methods for Plane
    intersection.

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

    beginning
    ending
    center
    points
    vector
    axis
    length
    as_LineSet

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
    save
    load
    get_obj_description
    fuse
    group_similar_shapes
    fuse_shape_groups
    fuse_similar_shapes

    from_point_vector
    from_two_points
    from_vertices
    from_plane_intersection
    get_extended
    get_angle
    get_fitted_to_points
    closest_to_line
    get_orthogonal_component
    projections_from_points
    projections_limits_from_points
    points_from_projections
    point_from_intersection
    get_LineSet_from_list
    get_simplified_loop_indices
    check_colinear
    get_segment_intersection
    get_segment_union
    """

    _fit_n_min = 2
    _model_args_n = 6
    _name = "line"
    _translatable = [0, 1, 2]
    _rotatable = [3, 4, 5]
    _color = np.array([0, 0, 0])

    @property
    def surface_area(self):
        """Surface area of primitive"""
        return 0

    @property
    def volume(self):
        """Volume of primitive."""
        return 0

    @property
    def beginning(self):
        """Start point of line."""
        return np.array(self.model[:3])

    @property
    def ending(self):
        """End point of line."""
        return self.beginning + self.vector

    @property
    def center(self):
        """Center point of line."""
        return self.beginning + self.vector / 2

    @property
    def points(self):
        """Start and end points."""
        return np.vstack([self.beginning, self.ending])

    @property
    def vector(self):
        """Vector from beginning point to end point."""
        return np.array(self.model[3:])

    @property
    def axis(self):
        """Unit vector defining axis of line."""
        return self.vector / self.length

    @property
    def length(self):
        """Length of line."""
        return np.linalg.norm(self.vector)

    @property
    def as_LineSet(self):
        line = LineSet(
            Vector3dVector([self.beginning, self.ending]), Vector2iVector([[0, 1]])
        )
        line.paint_uniform_color(self.color)
        return line

    @staticmethod
    def fit(points, normals=None):
        """Not defined for lines.

        Parameters
        ----------
        points : N x 3 array
            N input points
        normals : N x 3 array
            N normal vectors

        Returns
        -------
        None
        """
        raise RuntimeError("Fitting is not defined for lines.")

    def get_signed_distances(self, points):
        """Gives the minimum distance between each point to the line.

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
        distances = np.linalg.norm(np.cross(self.axis, points - self.beginning), axis=1)

        return distances

    def get_normals(self, points):
        """Gives, for each input point, the normal vector of the point closest
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
        normals = self.get_orthogonal_component(points)
        normals /= np.linalg.norm(normals, axis=1)[..., np.newaxis]
        return normals

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

        bbox = self.as_LineSet.get_axis_aligned_bounding_box()
        if slack > 0:
            min_bound = bbox.min_bound - slack
            max_bound = bbox.min_bound + slack
            bbox = AxisAlignedBoundingBox(min_bound, max_bound)
        return bbox

    def get_mesh(self, **options):
        """Creates mesh of the shape.

        Parameters
        ----------
        resolution : int, optional
            Mesh resolution. Default: 5.
        radius_ratio : float, optional
            Ratio between line thickness and line length. Default: 0.001.

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the shape.
        """
        # assert 1 == 0
        resolution = options.get("resolution", 5)
        radius = options.get("radius", 0.01)

        cylinder = Cylinder.from_base_vector_radius(
            # self.beginning, self.vector, self.length * radius_ratio)
            self.beginning,
            self.vector,
            radius,
        )
        return cylinder.get_mesh(resolution=resolution, closed=True)

    @classmethod
    def from_point_vector(cls, beginning, vector):
        """Creates cylinder from center base point, vector and radius as
        separated arguments.

        Parameters
        ----------
        beginning : 3 x 1 array
            First point in line.
        vector : 3 x 1 array
            Direction of line.

        Returns
        -------
        Line
            Generated shape.
        """
        # vector = np.array(ending) - np.array(beginning)
        return cls(list(beginning) + list(vector))

    @classmethod
    def from_two_points(cls, beginning, ending):
        """Creates line from beginning and end points.

        Parameters
        ----------
        beginning : 3 x 1 array
            First point in line.
        ending : 3 x 1 array
            Ending point in line.

        Returns
        -------
        Line
            Generated shape.
        """
        vector = np.array(ending) - np.array(beginning)
        return cls.from_point_vector(beginning, vector)

    @staticmethod
    def from_vertices(vertices):
        """Gives list of lines from input boundary points. Supposes the points
        are ordered in a closed loop.

        vertices : N x 3 array
            N input points

        Returns
        -------
        list of Line instances
            N-1 lines
        """
        num_points = len(vertices)
        if num_points < 2:
            raise ValueError("More than one point needed.")
        lines = []
        for i in range(num_points):
            lines.append(
                Line.from_two_points(vertices[i], vertices[(i + 1) % num_points])
            )
        return lines

    @staticmethod
    def from_plane_intersection(
        plane1,
        plane2,
        fit_vertices=True,
        intersect_parallel=False,
        eps_angle=np.deg2rad(5.0),
        eps_distance=1e-2,
    ):
        """Calculate the line defining the intersection between two planes.

        If the planes are not bounded, give a line of length 1.

        Parameters
        ----------
        plane1 : instance of Plane of PlaneBounded
            First plane
        plane2 : instance of Plane of PlaneBounded
            Second plane
        fit_vertices : boolean, optional
            If True, fits points of line so that it the projections of both
            planes lie within the line. If False, returns a line with length 1.
            Default: True.
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
        Line or None
        """
        axis = np.cross(plane1.normal, plane2.normal)
        norm = np.linalg.norm(axis)
        if abs(np.arcsin(norm)) < eps_angle:
            if not intersect_parallel:
                return None

            dot1 = np.dot(plane1.centroid, plane1.normal)
            dot2 = np.dot(plane2.centroid, plane2.normal)
            if abs(dot1 - dot2) > eps_distance:
                return None

            # closest_points = plane1.closest_vertices(plane2)[0]
            # point = (closest_points[0] + closest_points[1]) / 2
            # axis = np.cross(plane1.vertices.mean(axis=0) - plane2.vertices.mean(axis=0), plane1.normal + plane2.normal)
            # # axis = np.cross(closest_points[1] - closest_points[0], plane1.normal + plane2.normal)
            pairs, _ = plane1.closest_vertices(plane2, 2)
            p1, p2 = np.array(pairs).sum(axis=1) / 2
            # closest_points = plane1.closest_vertices(plane2, 10)
            # pair1 = closest_points[0]
            # pair2 = closest_points[-1]
            # p1 = (pair1[0] + pair1[1]) / 2
            # p2 = (pair2[0] + pair2[1]) / 2

            point = (p1 + p2) / 2
            # point = pair1[0]
            # axis = np.cross(p2 - p1, plane1.normal + plane2.normal)
            axis = np.cross(
                plane1.vertices.mean(axis=0) - plane2.vertices.mean(axis=0),
                plane1.normal + plane2.normal,
            )
            norm = np.linalg.norm(axis)
        else:
            A = np.vstack([plane1.normal, plane2.normal])
            B = -np.vstack([plane1.dist, plane2.dist])
            point = np.linalg.lstsq(A, B, rcond=None)[0].T[0]
        axis /= norm
        line = Line.from_point_vector(point, axis)

        # if fit_vertices and plane1.is_convex and plane1.is_convex:
        #     points = np.vstack([plane1.vertices, plane2.vertices])
        #     line = line.get_fitted_to_points(points)

        if fit_vertices:
            points = []
            if len(p := plane1.vertices) > 0:
                points.append(p)
            if len(p := plane2.vertices) > 0:
                points.append(p)

            if len(points) == 0:
                raise ValueError(
                    "fit_vertices is True, but both planes have no vertices."
                )
            points = np.vstack(points)

            line = line.get_fitted_to_points(points)

        return line

    def get_extended(self, multiplier):
        """Get extended (or shrinked) version of line.

        Parameters
        ----------
        multiplier : float
            Extension value.

        Returns
        -------
        Line
        """
        new_points = self.center + (self.points - self.center) * multiplier
        new_line = Line.from_two_points(*new_points)
        new_line.color = self.color
        return new_line

    def get_angle(self, other_element):
        """Calculates angle between line axis and other element's axis.

        If the other element is not a line, check if it has an element called
        "axis".

        Parameters
        ----------
        other_element : Line, vector or other class instance
            Vector, or element containing an "axis" attribute.

        Returns
        -------
        float
            angle in radians
        """
        if isinstance(other_element, Line):
            axis2 = other_element.axis
        else:
            if hasattr(other_element, "axis"):
                axis2 = other_element.axis
            elif len(other_element) == 3:
                axis2 = np.array(other_element)
                if not np.issubdtype(axis2.dtype, np.number):
                    raise ValueError("other_element is a non-numeric array.")
            else:
                raise ValueError("Invalid other_element.")

            axis2 /= np.linalg.norm(other_element.axis)

        return np.arccos(self.axis.dot(axis2))

    def get_fitted_to_points(self, points):
        """Creates a new line with beginning and end points fitted to
        projections of points.

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        Line
            Generated shape.
        """
        # projections = np.dot(points - self.beginning, self.axis)
        projections = self.projections_from_points(points)
        new_line = self.from_two_points(
            self.points_from_projections(min(projections)),
            self.points_from_projections(max(projections)),
        )
        new_line.color = self.color
        return new_line

    def closest_to_line(self, points):
        """Returns points in line that are the closest to the input
        points.

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        points_closest: N x 3 array
            N points in line
        """
        points = np.asarray(points)
        projection = (points - self.beginning).dot(self.axis)
        return self.beginning + projection[..., np.newaxis] * self.axis

    def get_orthogonal_component(self, points):
        """Removes the axis-aligned components of points.

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        points_orthogonal: N x 3 array
            N points
        """
        points = np.asarray(points)
        delta = points - self.beginning
        return -np.cross(self.axis, np.cross(self.axis, delta))

    def projections_from_points(self, points):
        """Gives projection in line axis.

        See: Line.points_from_projections

        Parameters
        ----------
        points : Nx3 array
            Nx1 array containing normal vectors.

        Returns
        -------

        projections : float
            Target projection.
        """
        points = np.asarray(points)
        if reduce := points.ndim == 1:
            points = points[np.newaxis]

        projections = ((points - self.beginning) * self.axis).sum(axis=1)
        if reduce:
            projections = projections[0]
        return projections

    def projections_limits_from_points(self, points):
        """Gives min and max value for projections of points in line axis.

        See: Line.projections_from_points

        Parameters
        ----------
        points : Nx3 array
            Nx1 array containing normal vectors.

        Returns
        -------

        projections : float
            Target projection.
        """
        projections = self.projections_from_points(points)
        return np.array([projections.min(), projections.max()])

    def points_from_projections(self, projections):
        """Gives point in line whose projection is equal to the input value.

        See: Line.projections_from_points

        Parameters
        ----------
        projections : Nx1 array float
            Target projections.

        Returns
        -------
        points
            Nx3 array points.
        """
        projections = np.asarray(projections)
        if reduce := projections.ndim == 0:
            projections = projections[np.newaxis]

        points = self.beginning + self.axis * projections[:, None]
        if reduce:
            points = points[0]
        return points

    def point_from_intersection(self, other, within_segment=True, eps=1e-3):
        """Calculates intersection point between two lines.

        Parameters
        ----------
        other : instance of Line
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
        if eps <= 0:
            raise ValueError("'eps' must be a sufficiently small, positive value.")

        if not isinstance(other, Line):
            raise ValueError("'other' must be an instance of Line.")

        # dot_vectors = np.dot(self.vector, other.vector)

        if abs(abs(np.dot(self.axis, other.axis)) - 1) < 1e-7:
            return None

        dot_vectors = np.dot(self.vector, other.vector)
        diff = self.beginning - other.beginning

        A = np.array([[self.length**2, -dot_vectors], [-dot_vectors, other.length**2]])

        B = np.array([-self.vector.dot(diff), other.vector.dot(diff)])

        ta, tb = np.linalg.lstsq(A, B, rcond=None)[0]

        # not sure about including this "-1" there:
        if within_segment:
            if not (-eps < ta < 1 + eps) or not (-eps < tb < 1 + eps):
                return None

        pa = self.beginning + self.vector * ta
        pb = other.beginning + other.vector * tb
        if np.linalg.norm(pa - pb) > eps:
            return None

        return (pa + pb) / 2

    @staticmethod
    def get_LineSet_from_list(lines):
        if isinstance(lines, Line):
            lines = [lines]

        points = []
        lines_indices = []
        N = 0
        for line in lines:
            points.append(line.beginning)
            points.append(line.ending)
            lines_indices.append([N, N + 1])
            N += 2

        lineset = LineSet()
        lineset.points = Vector3dVector(points)
        lineset.lines = Vector2iVector(lines_indices)
        lineset.colors = Vector3dVector([line.color for line in lines])
        return lineset

    @classmethod
    def get_simplified_loop_indices(
        cls,
        vertices,
        angle_colinear=0,
        min_point_dist=0,
        max_point_dist=np.inf,
        loop_indexes=None,
    ):
        """Construct lines from a list of vertices that form a closed loop.

        For each consecutive line in boundary points, simplify it if they are
        almost colinear, or too small.

        For example, defining:
            line1 = (vertices[0], vertices[1])
            line2 = (vertices[1], vertices[2])
        If angle(line1, line2) < angle_colinear, then loop_indexes[1] is removed
        from loop_indexes.

        Parameters
        ----------
        vertices : array_like of shape (N, 3)
            List of all points.
        angle_colinear : float, optional
            Small angle value for assuming two lines are colinear
        min_point_dist : float, optional
            If the simplified distance is bigger than this value, simplify
            regardless of angle. Default: 0.
        max_point_dist : float, optional
            If the simplified distance is bigger than this value, do not
            simplify. Default: np.inf
        loop_indexes : list, optional
            Ordered indices defining which points in `vertices` define the loop.

        Returns
        -------
        list
            Indexes of simplified loop.
        """
        if angle_colinear < 0:
            raise ValueError(
                "angle_colinear must be a positive value, " f"got {angle_colinear}"
            )

        if loop_indexes is None:
            loop_vertices = vertices.copy()
        else:
            loop_vertices = vertices[loop_indexes]

        lines = cls.from_vertices(loop_vertices)

        i = 0
        while True:
            if i >= len(lines):
                break

            line = lines[i]

            while True:
                if i + 1 >= len(lines):
                    break

                other = lines[i + 1]
                line_new = Line.from_two_points(line.beginning, other.ending)

                angle_calc = line.get_angle(other)

                if (length := line_new.length) < min_point_dist:
                    pass

                elif angle_calc >= angle_colinear or length > max_point_dist:
                    break

                line = lines[i] = line_new
                del lines[i + 1]

            i += 1

        new_vertices = [line.beginning for line in lines]
        return [np.where(np.all(v == vertices, axis=1))[0][0] for v in new_vertices]

    def check_colinear(self, other_line, distance_eps=1e-8, angle_eps=1e-8):
        """Check if lines are colinear.

        Parameters
        ----------
        other_line : Line
            Other line.
        distance_eps : float, optional
            Distance threshold to decide if line is colinear. Default: 1e-8.
        angle_eps : float, optional
            Angle threshold to decide if line is colinear. Default: 1e-8.

        Returns
        -------
        bool
            True if lines are colinear.
        """
        if not isinstance(other_line, Line):
            raise ValueError(
                f"other_line should be instance of Line, got {type(other_line)}."
            )

        if distance_eps < 0 or angle_eps < 0:
            raise ValueError("distance_eps and angle_eps should be non-negative.")

        distance_test = np.all(self.get_distances(other_line.points) <= distance_eps)
        angle_test = self.axis.dot(other_line.axis) <= np.cos(angle_eps)

        return distance_test and angle_test

    def get_segment_intersection(
        self, colinear_line, distance_eps=1e-8, angle_eps=1e-8
    ):
        """Return line intersection.

        Should only be used with colinear lines.

        Parameters
        ----------
        colinear_line : Line
            Colinear line.
        distance_eps : float, optional
            Distance threshold to decide if line is colinear. Default: 1e-8.
        angle_eps : float, optional
            Angle threshold to decide if line is colinear. Default: 1e-8.

        Returns
        -------
        Line or None
            Line intersection
        """

        if not self.check_colinear(colinear_line, distance_eps, angle_eps):
            raise ValueError("Lines are not colinear.")

        a_start, a_ending = self.projections_limits_from_points(self.points)
        b_start, b_ending = self.projections_limits_from_points(colinear_line.points)

        if (b_start > a_ending) or (a_start > b_ending):
            return None

        interval = [max(a_start, b_start), min(a_ending, b_ending)]
        output_points = self.points_from_projections(interval)
        new_line = Line.from_two_points(*output_points)
        new_line.color = self.color
        return new_line

    def get_segment_union(self, colinear_line, distance_eps=1e-8, angle_eps=1e-8):
        """Return line union.

        Should only be used with colinear lines.

        Parameters
        ----------
        colinear_line : Line
            Colinear line.
        distance_eps : float, optional
            Distance threshold to decide if line is colinear. Default: 1e-8.
        angle_eps : float, optional
            Angle threshold to decide if line is colinear. Default: 1e-8.

        Returns
        -------
        Line
            Line union.
        """

        if not self.check_colinear(colinear_line, distance_eps, angle_eps):
            raise ValueError("Lines are not colinear.")

        points = np.vstack([self.points, colinear_line.points])
        interval = self.projections_limits_from_points(points)
        output_points = self.points_from_projections(interval)
        new_line = Line.from_two_points(*output_points)
        new_line.color = self.color
        return new_line
