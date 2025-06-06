#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
import tempfile
import tarfile
import json
from typing import TYPE_CHECKING
from pathlib import Path
import warnings
from abc import ABC, abstractmethod
from itertools import combinations

if TYPE_CHECKING:
    from pyShapeDetector.geometry import TriangleMesh

import numpy as np

from scipy.spatial.transform import Rotation
from pyShapeDetector import utility, geometry


def _check_distance(shape1, shape2, aabb_intersection, inlier_max_distance):
    if aabb_intersection is not None:
        if not shape1.aabb.intersects(shape2.aabb, distance=aabb_intersection):
            return False

    # Check if there is at least one pair that is close enough
    if inlier_max_distance is not None:
        if not shape1.has_inliers or not shape2.has_inliers:
            warnings.warn("No inliers found, ignoring inlier distance test.")

        elif (
            shape1.inliers.compute_point_cloud_distance(shape2.inliers).min()
            > inlier_max_distance
        ):
            return False

    return True


def _get_partitions_legacy(num_shapes, pairs):
    new_indices = np.array(range(num_shapes))
    for pair, result in pairs.items():
        i, j = pair

        if new_indices[j] != j:
            continue

        if result:
            new_indices[j] = i

    partitions = []
    for index in set(new_indices):
        partition = set([i for i in np.where(new_indices == index)[0]])
        partitions.append(partition)
    return partitions


def _get_partitions(num_shapes, pairs):
    # Step 2: graph-based partitions from pairs
    partitions = []
    added_indices = set()
    for pair, result in pairs.items():
        if pair[0] in added_indices and pair[1] in added_indices:
            if not result:
                continue

            i0 = np.where([pair[0] in p for p in partitions])[0][0]
            partition = partitions.pop(i0)

            # both added in same partitions
            if pair[1] in partition:
                partitions.append(partition)

            # fuse partitions
            else:
                i1 = np.where([pair[1] in p for p in partitions])[0][0]
                partitions[i1] |= partition

        # Check if pair passes the test
        elif result:
            if len(partitions) == 0:
                partitions.append(set(pair))
                added_indices.add(pair[0])
                added_indices.add(pair[1])
            else:
                for partition in partitions:
                    if pair[1] in partition:
                        partition.add(pair[0])
                        added_indices.add(pair[0])
                        break
                    if pair[0] in partition:
                        partition.add(pair[1])
                        added_indices.add(pair[1])
                        # added_to_partition = True
                        break

        else:
            # Add single-element partitions for elements that failed the test
            if pair[0] not in added_indices and pair[1] not in added_indices:
                partitions.append({pair[0]})
                partitions.append({pair[1]})
                added_indices.add(pair[0])
                added_indices.add(pair[1])

    for i in range(num_shapes):
        if i not in added_indices:
            partitions.append({i})

    if (num_test := sum(len(p) for p in partitions)) != num_shapes:
        print(
            f"This shouldn't have happened, implementation error: {num_test} != {num_shapes}"
        )
        assert False

    return partitions


class Primitive(ABC):
    """
    Base class used to represent a geometrical primitive.

    To define a primitive, inherit from this class and define at least the
    following internal attributes:
        `_fit_n_min`
        `_model_args_n`
        `_name`
    And the following methods:
        `get_distances`
        `get_normals`

    The method `get_mesh` can also optionally be implemented to return a : 3 x 3 array
    TriangleMesh instance.

    The properties `surface_area` and `volume` can also be implemented.

    When multiple set of parameters can define the same surface, it might be
    useful to implement the property `canonical` to return the canonical form
    (useful for testing).

    Attributes
    ----------
    dimensions
    is_bounded
    fit_n_min
    model_args_n
    name
    model
    equation
    surface_area
    center
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

    Methods
    -------
    __init__
    __repr__
    __eq__
    _get_bounding_box
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
    transform
    align
    __put_attributes_in_dict__
    save
    __get_attributes_from_dict__
    load
    get_obj_description
    _get_weights_from_shapes
    _fuse_models
    _fuse_extra_data
    fuse
    group_similar_shapes
    fuse_shape_groups
    fuse_similar_shapes
    """

    _metrics = {}
    _color = None
    _mesh = None
    _decimals = None
    _dimensions = None
    _is_bounded = None

    @property
    def dimensions(self):
        """Number of dimensions of geometrical object."""
        if self._dimensions is None:
            print(
                f"'dimensions' not implemented for primitives of type {type(self)}, "
                "returning None."
            )
        return self._dimensions

    @property
    def is_bounded(self):
        """If False, geometric object is unbounded and cannot be visualized entirely."""
        if self._is_bounded is None:
            print(
                f"'is_bounded' not implemented for primitives of type {type(self)}, "
                "returning None."
            )
        return self._is_bounded

    @property
    def fit_n_min(self):
        """Minimum number of points necessary to fit a model."""
        return self._fit_n_min

    @property
    def model_args_n(self):
        """Number of parameters in the model."""
        return self._model_args_n

    @property
    def name(self):
        """Name of primitive."""
        return self._name

    @property
    def model(self):
        return self._model

    @property
    def equation(self):
        """Equation that defines the primitive."""
        raise NotImplementedError(
            f"Equation not implemented for {self.name} primitives."
        )

    @property
    def surface_area(self):
        """Surface area of primitive."""

        raise NotImplementedError(
            "Surface area not implemented for " f"{self.name} primitives."
        )

    @property
    def center(self):
        """Center of primitive."""
        raise NotImplementedError(f"Center not implemented for {self.name} primitives.")

    @property
    def volume(self):
        """Volume of primitive."""
        raise NotImplementedError(f"Volume not implemented for {self.name} primitives.")

    @property
    def canonical(self):
        """Return canonical form for testing."""
        return self.copy()

    def _get_axis_or_vector_or_normal(self):
        if hasattr(self, "axis"):
            return self.axis
        if hasattr(self, "vector"):
            return self.vector
        if hasattr(self, "normal"):
            return self.normal
        raise RuntimeError(f"Primitives of type {self.name} do not have an axis.")

    @property
    def color(self):
        if self._color is None:
            if len(self.inliers.colors) > 0:
                self._color = np.median(self.inliers.colors, axis=0)
            else:
                seed = int(str(abs(hash(self.name)))[:9])
                self._color = np.random.seed(seed)

        return self._color.astype(np.float32)

    @color.setter
    def color(self, new_color):
        new_color = np.asarray(new_color).flatten()
        if new_color.shape != (3,):
            raise ValueError("Invalid input shape.")
        if not np.issubdtype(new_color.dtype, np.number):
            raise ValueError("Input must be numeric array.")
        self._color = new_color.astype(np.float32)

        if self._mesh is not None:
            self._mesh.paint_uniform_color(new_color)

    @property
    def mesh(self) -> "TriangleMesh":
        if self._mesh is None:
            self._mesh = self.get_mesh()
            self._mesh.paint_uniform_color(self.color)
            self._mesh.compute_triangle_normals()

        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh):
        if isinstance(new_mesh, geometry.TriangleMesh) or new_mesh is None:
            self._mesh = new_mesh
        elif isinstance(new_mesh, geometry.TriangleMesh.__open3d_class__):
            self._mesh = geometry.TriangleMesh(new_mesh)

        else:
            raise ValueError(
                "Meshes must be of type TriangleMesh "
                "or open3d.geometry.TriangleMesh."
            )

    @property
    def inliers(self):
        return self._inliers

    @property
    def inliers_flattened(self):
        return geometry.PointCloud.from_points_normals_colors(
            self.inlier_points_flattened, self.inliers.normals, self.inliers.colors
        )

    @property
    def has_inliers(self):
        return len(self.inliers.points) > 0

    @property
    def inlier_mean(self):
        if not self.has_inliers:
            raise ValueError("Shape has no inliers.")
        return self.inliers.points.mean(axis=0)

    @property
    def inlier_median(self):
        if not self.has_inliers:
            raise ValueError("Shape has no inliers.")
        return np.median(self.inliers.points, axis=0)

    @property
    def inlier_points(self):
        """Convenience attribute that can be set to save inlier points"""
        return self.inliers.points

    @property
    def inlier_points_flattened(self):
        """Convenience attribute that can be set to save inlier points"""
        return self.flatten_points(self.inliers.points)

    @property
    def inlier_normals(self):
        """Convenience attribute that can be set to save inlier normals"""
        return self.inliers.normals

    @property
    def inlier_colors(self):
        """Convenience attribute that can be set to save inlier colors"""
        return self.inliers.colors

    @property
    def metrics(self):
        """Convenience attribute that can be set to save shape metrics"""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        if not isinstance(metrics, dict):
            raise ValueError("metrics should be a dict")
        self._metrics = metrics

    @property
    def axis_spherical(self):
        """Get axis in spherical coordinates, if the primitive has an axis."""
        x, y, z = self._get_axis_or_vector_or_normal()
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)
        return np.array([r, theta, phi])

    @property
    def axis_cylindrical(self):
        """Get axis in cylindrical coordinates, if the primitive has an axis."""
        x, y, z = self._get_axis_or_vector_or_normal()
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return np.array([rho, theta, z])

    @property
    def aabb(self):
        axis_aligned_bbox = self.get_axis_aligned_bounding_box()
        axis_aligned_bbox.color = self.color
        return axis_aligned_bbox

    @property
    def obb(self):
        oriented_bbox = self.get_oriented_bounding_box()
        oriented_bbox.color = self.color
        return oriented_bbox

    @property
    def as_open3d(self):
        return self.mesh.as_open3d

    def __repr__(self):
        params = [round(x, 5) for x in self.model]
        return type(self).__name__ + f"({params})"

    def __eq__(self, other_shape):
        # return self.is_similar_to(other_shape, rtol=1e-05, atol=1e-08)
        return (type(self) is type(other_shape)) and np.all(
            self.model == other_shape.model
        )

    @staticmethod
    def _parse_model_decimals(model, decimals):
        if decimals is None:
            return model

        if not isinstance(decimals, int) or decimals < 1:
            raise TypeError("decimals should be a positive integer.")
        return model.round(decimals)

    def __init__(self, model, decimals=None):
        """
        Parameters
        ----------
        model : Plane or list of 4 values
            Parameters defining the shape model
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
            self.__copy_atributes__(shape)
        else:
            model = np.array(model)

        if len(model) != self.model_args_n:
            raise ValueError(
                f"{self.name.capitalize()} primitives take "
                f"{self.model_args_n} elements, got {model}"
            )

        self._model = Primitive._parse_model_decimals(model, decimals)
        self._inliers = geometry.PointCloud()
        self._decimals = decimals

    @classmethod
    def random(cls, scale=1, decimals=16):
        """Generates a random shape.

        see: numpy.array.round

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
        Primitive
            Random shape.
        """
        model = np.random.random(cls._model_args_n) * scale
        return cls(model, decimals=decimals)

    @staticmethod
    @abstractmethod
    def fit(points, normals=None):
        """Gives shape that fits the input points. If the number of points is
        higher than the `_fit_n_min`, the fitted shape will return some kind of
        estimation.

        Moreover, some primitives do not need the normal vectors to fit, while
        others (like cylinders) might benefit from it.

        Actual implementation depends on the type of primitive, m

        Parameters
        ----------
        points : N x 3 array
            N input points
        normals : N x 3 array
            N normal vectors

        Returns
        -------
        Primitive
            Fitted shape.
        """
        pass

    @abstractmethod
    def get_signed_distances(self, points):
        """Gives the minimum distance between each point to the model.

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
        pass

    @utility.accept_one_or_multiple_elements(3)
    def get_distances(self, points):
        """Gives the absolute value of the minimum distance between each point
        to the model.

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        distances
            Nx1 array distances.
        """
        return abs(self.get_signed_distances(points))

    @abstractmethod
    def get_normals(self, points):
        """Gives, for each input point, the normal vector of the point closest
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
        pass

    @utility.accept_one_or_multiple_elements(3, 3)
    def get_angles_cos(self, points, normals):
        """Gives the absolute value of cosines of the angles between the input
        normal vectors and the calculated normal vectors from the input points.

        Parameters
        ----------
        points : N x 3 array
            N input points
        normals : N x 3 array
            N normal vectors

        Raises
        ------
        ValueError
            If number of points and normals are not equal

        Returns
        -------
        angles_cos or None
            Nx1 array with the absolute value of the cosines of the angles, or
            `None` if `normals` is `None`.

        """
        if len(normals) != len(points):
            raise ValueError(
                "Number of elements should all be equal, got "
                f"{len(points)} points and {len(normals)} normals."
            )

        if normals is None:
            return None
        normals = np.asarray(normals)
        normals_from_points = self.get_normals(points)
        angles_cos = np.clip(np.sum(normals * normals_from_points, axis=1), -1, 1)
        return np.abs(angles_cos)

    @utility.accept_one_or_multiple_elements(3, 3)
    def get_angles(self, points, normals):
        """Gives the angles between the input normal vectors and the
        calculated normal vectors from the input points.

        Parameters
        ----------
        points : N x 3 array
            N input points
        normals : N x 3 array
            N normal vectors

        Raises
        ------
        ValueError
            If number of points and normals are not equal

        Returns
        -------
        angles or None
            Nx1 array with the angles, or `None` if `normals` is `None`.

        """
        if normals is None:
            return None

        return np.arccos(self.get_angles_cos(points, normals))

    def get_residuals(self, points, normals):
        """Convenience function returning both distances and angles.

        Parameters
        ----------
        points : N x 3 array
            N input points
        normals : N x 3 array
            N normal vectors

        Raises
        ------
        ValueError
            If number of points and normals are not equal

        Returns
        -------
        tuple
            Tuple containing distances and angles.
        """
        if normals is None:
            return self.get_distances(points), None
        else:
            return self.get_distances(points), self.get_angles(points, normals)

    @utility.accept_one_or_multiple_elements(3)
    def flatten_points(self, points):
        """Stick each point in input to the closest point in shape's surface.

        Parameters
        ----------
        points : N x 3 array
            N input points

        Returns
        -------
        points_flattened : N x 3 array
            N points on the surface

        """
        difference = self.get_signed_distances(points)[
            ..., np.newaxis
        ] * self.get_normals(points)
        points_flattened = points - difference
        return points_flattened

    def flatten_PointCloud(self, pcd):
        """Return new pointcloud with flattened points.

        Parameters
        ----------
        pcd : Open3D.geometry.PointCloud
            Input pointcloud

        Returns
        -------
        Open3D.geometry.PointCloud
            Pointcloud with points flattened

        """
        pcd_flattened = geometry.PointCloud(self.flatten_points(pcd.points))
        pcd_flattened.normals = pcd.normals
        pcd_flattened.colors = pcd.colors
        return pcd_flattened

    def set_inliers(
        self,
        points_or_pointcloud,
        normals=None,
        colors=None,
        flatten=False,
        color_shape=False,
    ):
        """Set inlier points to shape.

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

        if geometry.PointCloud.is_instance_or_open3d(pcd := points_or_pointcloud):
            if normals is not None or colors is not None:
                raise TypeError(
                    "If PointCloud is given as input, normals and "
                    "colors are taken from it, and not accepted "
                    "as input."
                )

            points = pcd.points
            normals = pcd.normals
            colors = pcd.colors

        elif isinstance((shape := points_or_pointcloud), Primitive):
            if normals is not None or colors is not None:
                raise TypeError(
                    "If PointCloud is given as input, normals and "
                    "colors are taken from it, and not accepted "
                    "as input."
                )

            points = shape.inliers.points
            normals = shape.inliers.normals
            colors = shape.inliers.colors
        else:
            points = points_or_pointcloud

        points = utility._set_and_check_3d_array(points, "inlier points")
        num_points = len(points)
        normals = utility._set_and_check_3d_array(normals, "inlier normals", num_points)
        colors = utility._set_and_check_3d_array(colors, "inlier colors", num_points)

        if flatten:
            points = self.flatten_points(points)

        self.inliers.points = points
        if len(normals) > 0:
            self.inliers.normals = normals
        if len(colors) > 0:
            self.inliers.colors = colors
        if color_shape and (len(colors) > 0):
            self.color = np.median(colors, axis=0)

    def add_inliers(self, new_points):
        """Add inlier points to shape.

        If normals or/and colors are given, they must have the same shape as
        the input points.

        If normals or/and colors are given, they must have the same shape as
        the input points.

        Moreover, if original shape has inlier points but no normals or colors,
        trying to add new ones will fail.

        See: set_inliers

        Parameters
        ----------
        new_points : N x 3 array
            New inlier points.
        """
        new_points = utility._set_and_check_3d_array(new_points, "inlier points")
        num_points = len(new_points)

        new_colors = np.repeat(self.color, num_points).reshape(num_points, 3)

        self.inliers.points = np.vstack([self.inliers.points, new_points])

        if len(self.inliers.normals) > 0:
            new_normals = self.get_normals(new_points)
            self.inliers.normals = np.vstack([self.inliers.normals, new_normals])

        if len(self.inliers.colors) > 0:
            new_colors = np.repeat(self.color, num_points).reshape(num_points, 3)
            self.inliers.colors = np.vstack([self.inliers.colors, new_colors])

    def closest_inliers(self, other_shape, n=1):
        """Returns n pairs of closest inlier points with a second shape.

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
        if not isinstance(other_shape, Primitive):
            raise ValueError("other_shape must be a Primitive.")

        closest_points, distances = geometry.PointCloud.find_closest_points(
            self.inliers.points, other_shape.inliers.points, n
        )

        return closest_points, distances

    def inliers_average_dist(self, k=15, leaf_size=40):
        """Calculates the K nearest neighbors of the inlier points and returns
        the average distance between them.

        Parameters
        ----------
        k : positive int, default = 15
            Number of neighbors.

        leaf_size : positive int, default=40
            Number of points at which to switch to brute-force. Changing
            leaf_size will not affect the results of a query, but can
            significantly impact the speed of a query and the memory required
            to store the constructed tree.  The amount of memory needed to
            store the tree scales as approximately n_samples / leaf_size.
            For a specified ``leaf_size``, a leaf node is guaranteed to
            satisfy ``leaf_size <= n_points <= 2 * leaf_size``, except in
            the case that ``n_samples < leaf_size``.

        Returns
        -------
        float
            Average nearest dist.
        """
        return self.inliers.average_nearest_dist(k, leaf_size)

    def get_axis_aligned_bounding_box(
        self, slack: float = 0
    ) -> geometry.AxisAlignedBoundingBox:
        """Returns an axis-aligned bounding box of the primitive.

        Parameters
        ----------
        slack : float, optional
            Expand bounding box in all directions, useful for testing purposes.
            Default: 0.

        See: open3d.geometry.get_axis_aligned_bounding_box

        Returns
        -------
        AxisAlignedBoundingBox
        """
        if slack < 0:
            raise ValueError("Slack must be non-negative.")

        bbox = self.mesh.get_axis_aligned_bounding_box()
        if slack > 0:
            return bbox.expanded(slack)
        return bbox

    def get_oriented_bounding_box(
        self, slack: float = 0
    ) -> geometry.OrientedBoundingBox:
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

        oriented_bbox = self.mesh.get_oriented_bounding_box()
        if slack > 0:
            return oriented_bbox.expanded(slack)
        return oriented_bbox

    def sample_points_uniformly(self, number_of_points=100, use_triangle_normal=False):
        """Sample points from the mesh generated by the shape and then
        returns Numpy array with sampled points from the mesh.

        See: sample_points_density, sample_PointCloud_uniformly,
        sample_PointCloud_density

        Parameters
        ----------
        number_of_points : int, optional
            Number of points that should be uniformly sampled. Default = 100.
        use_triangle_normal : bool, optional
            If True assigns the triangle normals instead of the interpolated
            vertex normals to the returned points. The triangle normals will
            be computed and added to the mesh if necessary. Default = False.

        Returns
        -------
        Numpy array with shape (number_of_points, 3)
            Sampled pointcloud from shape.
        """
        pcd = self.sample_PointCloud_uniformly(number_of_points, use_triangle_normal)
        return np.asarray(pcd.points)

    def sample_points_density(self, density=1, use_triangle_normal=False):
        """Sample points from the mesh generated by the shape and then
        returns Numpy array with sampled points from the mesh.

        See: sample_points_density, sample_PointCloud_uniformly,
        sample_PointCloud_density

        Parameters
        ----------
        density: float, optional
            Ratio between points and surface area. Default: 1.
        use_triangle_normal : bool, optional
            If True assigns the triangle normals instead of the interpolated
            vertex normals to the returned points. The triangle normals will
            be computed and added to the mesh if necessary. Default = False.

        Returns
        -------
        Numpy array with shape (number_of_points, 3)
        """
        pcd = self.sample_PointCloud_density(density, use_triangle_normal)
        return np.asarray(pcd.points)

    def sample_PointCloud_uniformly(
        self, number_of_points=100, use_triangle_normal=False
    ):
        """Sample points from the mesh generated by the shape and then
        returns pointcloud with sampled points from the mesh.

        See: sample_points_uniformly, sample_points_density,
        sample_PointCloud_density

        Parameters
        ----------
        number_of_points : int, optional
            Number of points that should be uniformly sampled. Default = 100.
        use_triangle_normal : bool, optional
            If True assigns the triangle normals instead of the interpolated
            vertex normals to the returned points. The triangle normals will
            be computed and added to the mesh if necessary. Default = False.

        Returns
        -------
        open3d.geometry.PointCloud
            Sampled pointcloud from shape.
        """
        if number_of_points <= 0:
            raise ValueError("Number of points must be a non-negative number.")

        mesh = self.mesh
        pcd = mesh.sample_points_uniformly(number_of_points, use_triangle_normal)
        pcd.normals = self.get_normals(pcd.points)
        return pcd

    def sample_PointCloud_density(self, density=1, use_triangle_normal=False):
        """Sample points from the mesh generated by the shape and then
        returns pointcloud with sampled points from the mesh.

        See: sample_points_uniformly, sample_points_density,
        sample_PointCloud_uniformly

        Parameters
        ----------
        density: float, optional
            Ratio between points and surface area. Default: 1.
        use_triangle_normal : bool, optional
            If True assigns the triangle normals instead of the interpolated
            vertex normals to the returned points. The triangle normals will
            be computed and added to the mesh if necessary. Default = False.

        Returns
        -------
        open3d.geometry.PointCloud
            Sampled pointcloud from shape.
        """
        if density <= 0:
            raise ValueError("Density must be a non-negative number.")

        # mesh = self.get_mesh()
        number_of_points = int(density * self.surface_area)
        return self.sample_PointCloud_uniformly(number_of_points, use_triangle_normal)

    def get_mesh(self, **options):
        """Creates mesh of the shape.

        Parameters
        ----------
        resolution : int, optional
            Resolution parameter for mesh. Default: 30

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the shape.
        """
        raise NotImplementedError(
            "The mesh generating function for "
            f"primitives of type {self.name} has not "
            "been implemented."
        )

    def get_cropped_mesh(self, points=None, eps=1e-3):
        """Creates mesh of the shape and crops it according to points.

        Parameters
        ----------
        points : N x 3 array, optional
            N input points. If points are not given, tries to use inlier points
            of shape.
        eps : float, optional
            Small value for cropping.

        Returns
        -------
        TriangleMesh
            Mesh corresponding to the shape. Default: 1E-3
        """

        if points is None:
            points = self.inliers.points

        if len(points) == 0:
            raise ValueError("No points given, and no inlier points.")

        mesh = self.get_mesh()
        pcd = geometry.PointCloud(self.flatten_points(points))
        bb = pcd.get_axis_aligned_bounding_box()
        bb = geometry.AxisAlignedBoundingBox(
            bb.min_bound - [eps] * 3, bb.max_bound + [eps] * 3
        )
        # return mesh.crop(bb)
        return mesh.clean_crop(bb)

    def is_similar_to(self, other_shape, rtol=1e-02, atol=1e-02):
        """Check if shapes represent same model.

        Parameters
        ----------
        other_shape : Primitive
            Primitive to compare

        Returns
        -------
        Bool
            True if shapes are similar.
        """
        if not isinstance(self, type(other_shape)):
            return False

        compare = np.isclose(
            self.canonical.model, other_shape.canonical.model, rtol=rtol, atol=atol
        )
        return compare.all()

    def __copy_atributes__(self, shape_original):
        try:
            self.set_inliers(
                shape_original.inliers.points,
                shape_original.inliers.normals,
                shape_original.inliers.colors,
            )
        except AttributeError:
            pass

        self._metrics = shape_original._metrics.copy()
        self._color = shape_original._color.copy()
        # So that mesh can be recreated when getting different versions of primitives
        # Example: an unbounded version from a bounded version
        # self._mesh = copy.copy(shape_original._mesh)
        self._decimals = shape_original._decimals

    def __copy__(self):
        """Method for compatibility with copy module"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self.model.copy()
            primitive = type(self)
            shape = primitive(model, decimals=self._decimals)
            shape.__copy_atributes__(self)
        return shape

    def copy(self):
        """Returns copy of shape

        Returns
        -------
        Primitive
            Copied primitive
        """
        return self.__copy__()

    def _translate_points(self, translation):
        """Internal helper function for translation"""
        if len(self.inliers.points) > 0:
            self.inliers.points = self.inliers.points + translation

    def translate(self, translation, translate_inliers=True):
        """Translate the shape.

        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        translate_inliers : boolean, optional
            If True, also translate inliers. Default: True.
        """
        # if not hasattr(self, '_translatable'):
        #     raise NotImplementedError('Shapes of type {shape.name} do not '
        #                               'have an implemented _translatable '
        #                               'attribute')

        if len(self._translatable) != 0:
            self._model[self._translatable] += translation

        if translate_inliers:
            self._translate_points(translation)

        if self._mesh is not None:
            self._mesh.translate(translation)

    def transform(self, transformation_matrix, translate_inliers=True):
        """Apply 4x4 transformation matrix.

        Parameters
        ----------
        translation : 1 x 3 array
            Translation vector.
        translate_inliers : boolean, optional
            If True, also translate inliers. Default: True.
        """
        # if not hasattr(self, '_translatable'):
        #     raise NotImplementedError('Shapes of type {shape.name} do not '
        #                               'have an implemented _translatable '
        #                               'attribute')

        translation = transformation_matrix[:3, 3]
        rotation = transformation_matrix[:3, :3]
        self.rotate(rotation)
        self.translate(translation, translate_inliers=translate_inliers)

    @staticmethod
    def _parse_rotation(rotation):
        """Internal helper function for rotation"""
        if not isinstance(rotation, Rotation):
            rotation = Rotation.from_matrix(rotation)

        if isinstance(rotation, Rotation):
            try:
                length = len(rotation)
                if length[0] != 1:
                    raise ValueError(
                        "Rotation input should contain a single "
                        " rotation but has len(rotation) instead"
                    )
                rotation = rotation[0]

            except TypeError:
                pass

        return rotation

    def _rotate_points_normals(self, rotation):
        """Internal helper function for rotation"""
        if len(self.inliers.points) > 0:
            self.inliers.points = rotation.apply(self.inliers.points)
        if len(self.inliers.normals) > 0:
            self.inliers.normals = rotation.apply(self.inliers.normals)

    def rotate(self, rotation, rotate_inliers=True):
        """Rotate the shape.

        Parameters
        ----------
        rotation : 3 x 3 rotation matrix or scipy.spatial.transform.Rotation
            Rotation matrix.
        rotate_inliers : boolean, optional
            If True, also rotate inliers. Default: True.
        """
        # if not hasattr(self, '_rotatable'):
        #     raise NotImplementedError('Shapes of type {shape.name} do not '
        #                               'have an implemented _rotatable '
        #                               'attribute')

        rotation = Primitive._parse_rotation(rotation)

        if len(self._rotatable) != 0:
            self._model[self._rotatable] = rotation.apply(self.model[self._rotatable])

        if len(self._translatable) != 0:
            self._model[self._translatable] = rotation.apply(
                self.model[self._translatable]
            )

        if rotate_inliers:
            self._rotate_points_normals(rotation)

        if self._mesh is not None:
            self._mesh.rotate(rotation.as_matrix())

    def align(self, axis, possible_attributes=["axis", "vector", "normal"]):
        """Returns aligned

        Parameters
        ----------
        axis : 3 x 1 array
            Axis to which the shape should be aligned.
        possible_attributes : list of strings, optional
            Attribute that should be aligned to axis. If shape has any of
            those, it will be aligned. Otherwise, nothing is done.
            Default: ['axis', 'vector', 'normal']
        """
        for attr in possible_attributes:
            if hasattr(self, attr):
                axis_original = getattr(self, attr)
                rotation = utility.get_rotation_from_axis(axis_original, axis)
                self.rotate(rotation)
                break

    def __put_attributes_in_dict__(self, data, save_inliers=True):
        data["file_version"] = 2
        data["name"] = self.name
        data["model"] = self.model.tolist()
        if save_inliers:
            data["inlier_points"] = self.inliers.points.tolist()
            data["inlier_normals"] = self.inlier_normals.tolist()
            data["inlier_colors"] = self.inlier_colors.tolist()
        if self.color is not None:
            data["color"] = self.color.tolist()

    def save(self, path, save_inliers=None):
        """Saves shape to a file.

        File extension can be:
        - json:
            Saves everything in a single json file, with or without inliers.
        - tar:
            Saves inliers in a separated ply file, everything on a tar file.

        Parameters
        ----------
        path : string of pathlib.Path
            File destination.
        save_inliers : boolean, optional
            Add inliers to JSON file. Ignores this if file is a tar.
            Default: True for tar files and False for json files.
        """

        path = Path(path)
        if path.exists():
            path.unlink()

        if path.suffix == ".json" and save_inliers:
            warnings.warn(
                "Saving inliers in json files is not efficient, consider saving as a tar file."
            )

        if not self.has_inliers:
            save_inliers = False

        if path.suffix == ".json":
            if save_inliers is None:
                save_inliers = False

            json_data = {}
            self.__put_attributes_in_dict__(json_data, save_inliers=save_inliers)
            with open(path, "w") as fp:
                json.dump(json_data, fp, indent=4)

        elif path.suffix == ".tar":
            if save_inliers is None:
                save_inliers = True

            json_data = {}
            self.__put_attributes_in_dict__(json_data, save_inliers=False)

            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a temporary JSON file
                temp_dir = Path(temp_dir)
                json_file_name = "shape.json"
                temp_json_file_path = temp_dir / json_file_name
                with open(temp_json_file_path, "w") as fp:
                    json.dump(json_data, fp, indent=4)

                # Create a temporary copy of the other file
                if save_inliers:
                    temp_inliers_path = temp_dir / "inliers.ply"
                    self.inliers.write_point_cloud(temp_inliers_path)

                # Create the tar file and add both files
                with tarfile.open(path, "w") as tar:
                    tar.add(temp_json_file_path, arcname=json_file_name)
                    if save_inliers:
                        tar.add(temp_inliers_path, arcname="inliers.ply")
        else:
            raise ValueError(
                f"Acceptable extensions are '.tar' and '.json', got {path.suffix}."
            )

    def __get_attributes_from_dict__(self, data):
        try:
            self.inliers.points = np.array(data["inlier_points"])
            self.inliers.normals = np.array(data["inlier_normals"])
            self.inliers.colors = np.array(data["inlier_colors"])
            # print("[Info] Inliers found.")
        except KeyError:
            # print("[Info] No inliers found.")
            pass
        color = data.get("color")
        if color is not None:
            self.color = color

    @staticmethod
    def load(path):
        """Laods shape from JSON file.

        Parameters
        ----------
        path : string of pathlib.Path
            File destination.

        Returns
        -------
        Primitive
            Loaded shape.
        """
        import json
        from pyShapeDetector.primitives import dict_primitives

        extension = Path(path).suffix

        separated_inliers = None

        if extension == ".json":
            with open(path, "r") as f:
                json_data = json.load(f)

        elif extension == ".tar":
            with tempfile.TemporaryDirectory() as temp_dir:
                # Open the tar file
                with tarfile.open(path, "r") as tar:
                    # Extract all files to the temporary directory
                    tar.extractall(path=temp_dir)

                    # Read the JSON file
                    json_path = Path(temp_dir) / "shape.json"
                    with open(json_path, "r") as json_file:
                        json_data = json.load(json_file)

                    inliers_path = Path(temp_dir) / "inliers.ply"
                    if inliers_path.exists():
                        separated_inliers = geometry.PointCloud.read_point_cloud(
                            inliers_path
                        )
        else:
            raise ValueError(
                f"Acceptable extensions are 'tar' and 'json', got {extension}."
            )

        name = json_data["name"]
        primitive = dict_primitives[name]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape = primitive(json_data["model"])

        shape.__get_attributes_from_dict__(json_data)
        if separated_inliers is not None:
            shape.set_inliers(separated_inliers)

        return shape

    def get_obj_description(self, shading="off", mtl="Material", **kwargs):
        """
        Converts the primitive to an OBJ file content string.

        Parameters
        ----------
        shading: str or int
            Shading across polygons is enabled by smoothing groups. Default: "off".
        mtl: str
            Specifies the material name for the element following it.
            The material name matches a named material definition in an external
            .mtl file. Default: "Material"

        Returns
        -------
        str
            The content of the OBJ file as a string.
        """
        name = type(self).__name__
        return utility.mesh_to_obj_description(
            name, self.mesh, shading=shading, mtl=mtl, **kwargs
        )

    @classmethod
    def _get_weights_from_shapes(
        cls, shapes: list["Primitive"], weight_variable: str = "inliers"
    ) -> list[float]:
        """Get weights to correctly calculate the average of shapes.

        `weight_variable` can be:
            "surface_area": uses area of each surface, if all finite.
            "fitness": takes fitness of each fit, if available.
            "volume": uses volume of each surface, if all finite.
            "ones": uses array of ones, equivalent to non-weighted average.
            "inliers": uses number of inliers.

        If the desired weight cannot be used, returns a list of `1`s.

        Parameters
        ----------
        weight_variable : str
            Defines variable used as weight. Can be "ones" (non-weighed averate),
            "surface_area", "fitness", "volume" or "inliers". Default: "inliers".

        Returns
        -------
        list
            Weights for each value.
        """
        valid_types = ["surface_area", "fitness", "volume", "ones", "inliers"]
        if weight_variable not in valid_types:
            raise ValueError(
                f"Valid values for 'weight_variable' are {valid_types}, "
                "got {weight_variable}."
            )

        if weight_variable == "ones":
            return [1] * len(shapes)

        elif weight_variable == "surface_area":
            weights = [shape.surface_area for shape in shapes]

        elif weight_variable == "volume":
            weights = [shape.volume for shape in shapes]

        elif weight_variable == "inliers":
            weights = [len(shape.inliers.points) for shape in shapes]

        elif weight_variable == "fitness":
            try:
                weights = [shape.metrics["fitness"] for shape in shapes]
            except Exception:
                warnings.warn("Fitness not found, returning array of '1's.")
                return [1] * len(shapes)

        else:
            # Should never run
            assert False

        if not np.all(np.isfinite(weights)):
            warnings.warn("Non-finite value found, returning array of '1's.")
            return [1] * len(shapes)

        eps = 1e-7
        if abs(sum(weights)) < eps:
            warnings.warn("Total sum of weights is too small, returning array of '1's.")
            return [1] * len(shapes)

        return weights

    @classmethod
    def _fuse_models(cls, shapes: list["Primitive"], weights: list[float]):
        """Find weighted average of models.

        Parameters
        ----------
        shapes : list
            Grouped shapes. All shapes must be of the same type.
        weights : list
            Weights for average model.
        """
        models = np.vstack([shape.model for shape in shapes])
        return np.average(models, axis=0, weights=weights)

    def _fuse_extra_data(self, shapes: list["Primitive"], detector):
        """Fuse extra data from input shapes into self, like color, inliers and metrics.

        Parameters
        ----------
        shapes : list
            Grouped shapes. All shapes must be of the same type.
        detector : instance of some Detector, optional

        """
        pcd = geometry.PointCloud.fuse_pointclouds([shape.inliers for shape in shapes])
        self.set_inliers(pcd)
        self.color = np.mean([s.color for s in shapes], axis=0)

        if detector is not None:
            num_points = sum([shape.metrics["num_points"] for shape in shapes])
            num_inliers = len(pcd.points)
            distances, angles = self.get_residuals(pcd.points, pcd.normals)
            self.metrics = detector.get_metrics(
                num_points, num_inliers, distances, angles
            )

    @staticmethod
    def fuse(
        shapes: list["Primitive"],
        detector=None,
        ignore_extra_data=False,
        line_intersection_eps=None,
        weight_variable: str = "inliers",
        **extra_options,
    ):
        """Find weigthed average of shapes, where the weight is the fitness
        metric.

        If a detector is given, use it to compute the metrics of the resulting
        average shapes.

        `weight_variable` can be:
            "surface_area": uses area of each surface, if all finite.
            "fitness": takes fitness of each fit, if available.
            "volume": uses volume of each surface, if all finite.
            "ones": uses array of ones, equivalent to non-weighted average.
            "inliers": uses number of inliers.

        Parameters
        ----------
        shapes : list
            Grouped shapes. All shapes must be of the same type.
        detector : instance of some Detector, optional
            Used to recompute metrics. Default: None.
        ignore_extra_data : boolean, optional
            If True, ignore everything and only fuse model. Default: False.
        weight_variable : str
            Defines variable used as weight. Can be "ones" (non-weighed averate),
            "surface_area", "fitness", "volume" or "inliers". Default: "inliers".

        Returns
        -------
        Primitive
            Averaged shape.
        """
        if isinstance(shapes, Primitive):
            return shapes
        elif len(shapes) == 1:
            return shapes[0].copy()

        primitive = type(shapes[0])
        for shape in shapes[1:]:
            if not isinstance(shape, primitive):
                raise ValueError("Shapes in input must all have the same type.")

        weights = primitive._get_weights_from_shapes(
            shapes, weight_variable=weight_variable
        )

        model = primitive._fuse_models(shapes=shapes, weights=weights)

        # Catch warning in case shape is a PlaneBounded
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape = primitive(model)

        if not ignore_extra_data:
            shape._fuse_extra_data(shapes=shapes, detector=detector)

        return shape

    @staticmethod
    def group_similar_shapes(
        shapes,
        rtol=1e-02,
        atol=1e-02,
        aabb_intersection=None,
        inlier_max_distance=None,
        legacy=False,
        return_partitions=False,
    ):
        """Detect shapes with similar model and group.

        See: fuse_shape_groups

        Parameters
        ----------
        shapes : list of shapes
            List containing all shapes.
        rtol : float, optional
            The relative tolerance parameter. Default: 1e-02.
        atol : float, optional
            The absolute tolerance parameter. Default: 1e-02.
        bbox_intersection : float, optional
            Max distance between inlier bounding boxes. If None, ignore this test.
            Default: None.
        inlier_max_distance : float, optional
            Max distance between points in shapes. If None, ignore this test.
            Default: None.
        legacy : bool, optional
            Uses legacy implementation of `group_similar_shapes`. Default: False
        return_partitions :  bool, optional
            If True, return sets defining partitions. Default: False.

        Returns
        -------
        list of lists
            Grouped shapes
        list of sets
            Index partitions defining the shape groups.
        """

        def _test(i, j):
            if not shapes[i].is_similar_to(shapes[j], rtol=rtol, atol=atol):
                return False
            return _check_distance(
                shapes[i], shapes[j], aabb_intersection, inlier_max_distance
            )

        # Step 1: check all pairs
        shape_pairs = combinations(range(len(shapes)), 2)
        pairs = {pair: _test(*pair) for pair in shape_pairs}

        # Step 2: partitions from pairs
        if legacy:
            partitions = _get_partitions_legacy(len(shapes), pairs)
        else:
            # graph-based
            partitions = _get_partitions(len(shapes), pairs)

        # Step 3: get sublists of shapes from partitions
        # shape_groups = [[shapes[i] for i in partition] for partition in partitions]

        shape_groups = []
        for partition in partitions:
            group = [shapes[i] for i in partition]
            shape_groups.append(group)
        if return_partitions:
            return shape_groups, partitions
        return shape_groups

    @staticmethod
    def fuse_shape_groups(shapes_lists, **fuse_options):
        """Find weigthed average of shapes, where the weight is the fitness
        metric.

        If a detector is given, use it to compute the metrics of the resulting
        average shapes.

        See: group_similar_shapes

        Parameters
        ----------
        shapes_lists : list of list shapes
            Grouped shapes.
        detector : instance of some Detector, optional
            Used to recompute metrics.
        ignore_extra_data : boolean, optional
            If True, ignore everything and only fuse model. Default: False.
        line_intersection_eps : float, optional
            Distance for detection of intersection between planes. Default: 0.001.

        Extra parameters for PlaneBounded
        ---------------------------------
        force_concave : boolean, optional.
            If True, the fused plane will be concave regardless of inputs.
            Default: True.
        ressample_density : float, optional
            Default: 1.5
        ressample_radius_ratio : float, optional
            Default: 1.2

        Returns
        -------
        list
            Average shapes.
        """
        fused_shapes = []
        for sublist in shapes_lists:
            primitive = type(sublist[0])
            fused_shape = primitive.fuse(sublist, **fuse_options)

            num_points = sum(len(s.inlier_points) for s in sublist)
            if num_points != len(fused_shape.inlier_points):
                pass

            fused_shapes.append(fused_shape)

        return fused_shapes

    @staticmethod
    def fuse_similar_shapes(
        shapes,
        rtol=1e-02,
        atol=1e-02,
        bbox_intersection=None,
        inlier_max_distance=None,
        legacy=False,
        **fuse_options,
    ):
        """Detect shapes with similar model and fuse them.

        If a detector is given, use it to compute the metrics of the resulting
        average shapes.

        For extra fuse options, see: fuse_shape_groups

        See: group_shape_groups

        Parameters
        ----------
        shapes : list of shapes
            List containing all shapes.
        rtol : float, optional
            The relative tolerance parameter. Default: 1e-02.
        atol : float, optional
            The absolute tolerance parameter. Default: 1e-02.
        bbox_intersection : float, optional
            Max distance between inlier bounding boxes. If None, ignore this test.
            Default: None.
        inlier_max_distance : float, optional
            Max distance between points in shapes. If None, ignore this test.
            Default: None.

        Returns
        -------
        list
            Average shapes.
        """
        partitions = Primitive.group_similar_shapes(
            shapes, rtol, atol, bbox_intersection, inlier_max_distance, legacy
        )
        shape_groups = [[shapes[i] for i in partition] for partition in partitions]

        return Primitive.fuse_shape_groups(shape_groups, **fuse_options)
