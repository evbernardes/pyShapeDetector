#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:26:20 2024

@author: ebernardes
"""
import itertools
from pathlib import Path
import numpy as np
import multiprocessing
import time
import warnings

from open3d.geometry import PointCloud as open3d_PointCloud
from open3d.geometry import KDTreeFlann
from open3d import io

from .axis_aligned_bounding_box import AxisAlignedBoundingBox
from .oriented_bounding_box import OrientedBoundingBox

from scipy.spatial.distance import cdist

import h5py
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from pyShapeDetector.utility import rgb_to_cielab, cielab_to_rgb, parallelize
from .open3d_geometry import link_to_open3d_geometry, Open3D_Geometry


@link_to_open3d_geometry(open3d_PointCloud)
class PointCloud(Open3D_Geometry):
    """
    PointCloud class that uses Open3D.geometry.PointCloud internally.

    Almost every method and property are automatically copied and decorated.

    Attributes
    ----------
    volume
    curvature
    has_curvature
    colors_cielab

    Methods
    -------
    from_points_normals_colors
    distribute_to_closest
    estimate_curvature
    fuse_pointclouds
    average_nearest_dist
    write_point_cloud
    read_point_cloud
    split
    split_in_half
    split_until_small
    separate_with_labels
    segment_by_position
    segment_kmeans_colors
    segment_dbscan
    segment_curvature_threshold
    segment_with_region_growing
    find_closest_points_indices
    find_closest_points
    """

    _curvature = np.empty(0)

    @property
    def volume(self):
        return np.product(self.get_oriented_bounding_box().extent)

    @property
    def curvature(self):
        return self._curvature

    @curvature.setter
    def curvature(self, values):
        values = np.array(values)
        if values.ndim != 1:
            raise ValueError(
                "Curvature values must be one-dimensional ",
                f"got array of shape {values.shape}.",
            )
        self._curvature = values

    # @property
    def has_curvature(self):
        return len(self.curvature) > 0

    @property
    def colors_cielab(self):
        return rgb_to_cielab(self.colors.copy())

    @colors_cielab.setter
    def colors_cielab(self, lab):
        self.colors = cielab_to_rgb(lab)

    def from_points_normals_colors(element, normals=[], colors=[]):
        """Creates PointCloud instance from points, normals or colors.

        Parameters
        ----------
        element : instance Primitive, PointCloud or numpy.array
            Shape with inliers or Points of the PointCloud.
        normals : numpy.array, optional
            Normals for each point of the PointCloud, must have same length as
            points. Default: None.
        colors : numpy.array, optional
            Colors for each point of the PointCloud, must have same length as
            points. Default: None.

        Returns
        -------
        Open3d.geometry.PointCloud
            PointCloud defined by the inputs.
        """
        from pyShapeDetector.primitives import Primitive

        if isinstance((shape := element), Primitive):
            points = shape.inlier_points
            if normals is None:
                normals = shape.inliers.normals
            if colors is None:
                colors = shape.inliers.colors
        else:
            points = element

        pcd = PointCloud()
        pcd.points = points
        pcd.normals = normals
        pcd.colors = colors
        return pcd

    def get_oriented_bounding_box(self):
        try:
            oriented_bbox = self.get_minimal_oriented_bounding_box()

        except RuntimeError as e:
            if "Qhull precision error" in str(e):
                pca = PCA(n_components=3)
                pca.fit(self.points)
                # extent = pca.explained_variance_
                extent = 2 * np.array(
                    [
                        max(abs((self.points - pca.mean_).dot(v)))
                        for v in pca.components_
                    ]
                )
                oriented_bbox = OrientedBoundingBox(
                    center=pca.mean_, R=pca.components_.T, extent=extent
                )
                warnings.warn("PointCloud has no extent in one of its dimensions.")

            else:
                raise e

        return oriented_bbox

    def distribute_to_closest(self, pcds):
        """Add each point to the closest of the input pointclouds.

        Parameters
        ----------
        pcds : list of pointclouds
            List of pointclouds to distribute.
        """

        distances = np.vstack([self.compute_point_cloud_distance(pcd) for pcd in pcds])
        labels = np.argmin(distances, axis=0)
        pcds_separated = self.separate_with_labels(labels)
        for pcd, distributed in zip(pcds, pcds_separated):
            pcd._open3d += distributed._open3d

    def estimate_curvature(self, k=15, cores=10):
        """Estimate curvature of points by getting the mean value of angles between
        the neighbors.

        Parameters
        ----------
        k : positive int, default = 15
            Number of neighbors.
        cores : positive int, default = 10
            Number of cores used to paralellize the computation.

        """

        if not self.has_normals():
            raise RuntimeError("Pointcloud has no normals, call estimate_normals.")

        points = self.points
        normals = self.normals
        from scipy.spatial import KDTree

        tree = KDTree(points)
        # curvature = np.zeros(len(points))

        @parallelize(cores)
        def _get_normals(indices):
            curvature = np.empty(len(indices))
            j = 0
            for i in indices:
                _, idx = tree.query(points[i], k=k)
                neighbors = normals[idx]
                # angles = np.arccos(np.clip(np.dot(neighbors, normals[i]), -1.0, 1.0))
                dot_products = abs(np.dot(neighbors, normals[i]))
                dot_mean = np.clip(dot_products.mean(), 0.0, 1.0)
                # angles = np.arccos(np.clip(np.dot(neighbors, normals[i]), -1.0, 1.0))
                # curvature[j] = np.mean(angles)
                curvature[j] = np.arccos(dot_mean)
                j += 1
            return curvature

        self.curvature = _get_normals(np.arange(len(points)))

    def average_nearest_dist(self, k=15, leaf_size=40, split=1):
        """Calculates the K nearest neighbors and returns the average distance
        between them.

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
        split : positive int, optional
            If a value bigger than 1 is given, the pointcloud will be split
            in sections and the average dist will be taken from the biggest
            split. Default: 1.

        Returns
        -------
        float
            Average nearest dist.
        """

        if not isinstance(split, int) or split < 1:
            raise ValueError(f"split must be a positive integer, got {split}.")

        if split > 1:
            pcds = self.split(split)
            pcd = pcds[np.argmax([len(p.points) for p in pcds])]
        else:
            pcd = self

        tree = KDTree(pcd.points, leaf_size=leaf_size)
        nearest_dist, nearest_ind = tree.query(pcd.points, k=k)
        return np.mean(nearest_dist[:, 1:])

    def write_point_cloud(self, filepath, **options):
        """Write pointcloud to file.

        Internal call to Open3D.io.write_point_cloud

        Parameters
        ----------
        filepath : string or instance of pathlib.Path
            File to be loaded

        pointcloud : Open3D.geometry.PointCloud
            PointCloud to be saved

        See: Open3D.io.write_point_cloud
        """
        if isinstance(filepath, Path):
            filepath = filepath.as_posix()
        io.write_point_cloud(filepath, self.as_open3d, **options)

    @classmethod
    def read_point_cloud(cls, filepath, down_sample=None, estimate_normals=False):
        """Read file to pointcloud. Can also read .h5 files from traceparts
        database.

        Internal call to Open3D.io.read_point_cloud.

        Parameters
        ----------
        filepath : string or instance of pathlib.Path
            File to be loaded

        down_sample : int, optional
            If not None, downsample pointcloud uniformly. The sample is performed
            in the order of the points with the 0-th point always chosen,
            not at random. Default: None.
            See: open3d.geometry.PointCloud.uniform_down_sample

        estimate_normals : boolean, optional
            If True, compute the normals of a point cloud. Normals are oriented
            with respect to the input point cloud if normals exist. Defdault: False.
            See: open3d.geometry.PointCloud.estimate_normals

        Returns
        -------
        PointCloud
            Loaded point cloud.

        See: Open3D.io.read_point_cloud
        """
        if isinstance(filepath, Path):
            filename = filepath.as_posix()
        else:
            filename = filepath
            filepath = Path(filename)

        # print(filename)

        if filepath.suffix == ".h5":
            f = h5py.File(filepath, "r")
            labels = np.asarray(f["gt_labels"])
            points = f["gt_points"]
            normals = f["gt_normals"]
            pcds = []
            for label in set(labels):
                idx = label == labels
                pcd = PointCloud()
                pcd.points = points[idx]
                pcd.normals = normals[idx]
                pcds.append(pcd)
            if len(pcds) == 1:
                pcds = pcds[0]
            f.close()

        else:
            pcds = PointCloud(io.read_point_cloud(filename))

        if down_sample is not None:
            try:
                pcds = [pcd.uniform_down_sample(down_sample) for pcd in pcds]
            except TypeError:
                pcds = pcds.uniform_down_sample(down_sample)

        if estimate_normals:
            try:
                [pcd.estimate_normals() for pcd in pcds]
            except TypeError:
                pcds.estimate_normals()

        return pcds

    def split(self, num_boxes, dim=None, return_only_indices=False):
        """Split the bounding box of the pointcloud in multiple sub boxes and
        return a list of sub pointclouds.

        Parameters
        ----------
        num_boxes : int
            Number of sub-boxes.
        dim : int, optional
            Dimension that should be divided. If not given, will be chosen as the
            largest dimension. Default: None.
        return_only_indices : boolean, optional
            If True, returns indices instead of pointclouds. This is much
            slower for bigger pointclouds. Default: False.

        Returns
        -------
        list
            Divided pointclouds
        """
        suaabbes = self.get_axis_aligned_bounding_box().split(num_boxes, dim)

        if return_only_indices:
            return [
                aabb.get_point_indices_within_bounding_box(self.points)
                for aabb in suaabbes
            ]
        else:
            return [self.crop(aabb) for aabb in suaabbes]

    # def split(self, num_boxes, dim=None, return_only_indices=False):
    #     """ Split the bounding box of the pointcloud in multiple sub boxes and
    #     return a list of sub pointclouds.

    #     Parameters
    #     ----------
    #     num_boxes : int
    #         Number of sub-boxes.
    #     dim : int, optional
    #         Dimension that should be divided. If not given, will be chosen as the
    #         largest dimension. Default: None.
    #     return_only_indices : boolean, optional
    #         If True, returns indices instead of pointclouds. Default: False.

    #     Returns
    #     -------
    #     list
    #         Divided pointclouds
    #     """
    #     suaabbes = self.get_axis_aligned_bounding_box().split(num_boxes, dim)
    #     points = self.points
    #     available_indices = set(range(len(self.points)))
    #     subsets = []
    #     for aabb in suaabbes:

    #         indices = aabb.get_point_indices_within_bounding_box(points)
    #         indices_set = set(indices)
    #         for subset in subsets:
    #             indices_set = indices_set - subset

    #         subsets.append(indices_set)
    #         available_indices = available_indices - indices_set

    #         if len(available_indices) == 0:
    #             break

    #     if len(available_indices) != 0:
    #         subsets[-1] = subsets.union(available_indices)

    #     if return_only_indices:
    #         return subsets

    #     return [self.select_by_index(list(subset)) for subset in subsets]

    def split_in_half(self, resolution=30):
        """Divide pointcloud in two sub clouds, each one occupying roughly the
        same volume.

        Parameters
        ----------
        resolution : int, optional
            Number of subdivisions done before joining them. Default: 5.

        Returns
        -------
        tuple
            Subdivided pointclouds
        """
        if resolution < 2:
            raise ValueError(f"Resolution must be at least 2, got {resolution}.")

        pcds = self.split(resolution)
        test = np.cumsum([len(pcd.points) for pcd in pcds]) - len(self.points) / 2

        left = PointCloud()
        right = PointCloud()

        for i, pcd in enumerate(pcds):
            if test[i] < 0:
                left += pcd
            else:
                right += pcd

        return left, right

    def split_until_small(self, max_points=1000000, resolution=30, debug=False):
        """Recursively divide pointcloud in two, until each pointcloud has
        less points than `max_points`.

        Parameters
        ----------
        max_points : int, optional
            Max number of points in each output pointcloud.
        resolution : int, optional
            Number of subdivisions done before joining them. Default: 5.

        Returns
        -------
        tuple
            Subdivided pointclouds
        """
        if resolution > 500:
            warnings.warn("Resolution too high, returning cannot go further.")
            return [self]

        if len(self.points) <= max_points:
            return [self]

        pcds = self.split_in_half()
        if np.any(np.array([len(p.points) for p in pcds]) == 0):
            old_res = resolution
            resolution = int(1.1 * resolution)
            warnings.warn(
                f"Subarray with points detected, "
                f"resolution changing from {old_res} to {resolution}."
            )
        # print(f'{len(self.points)} to {[len(p.points) for p in pcds]}')

        pcds = pcds[0].split_until_small(max_points, resolution, debug) + pcds[
            1
        ].split_until_small(max_points, resolution, debug)

        return [pcd for pcd in pcds if len(pcd.points) != 0]

    def separate_with_labels(self, labels, return_ungroupped=False):
        """Separate pcd with labels.

        Each distinct value of `labels` correspond to a separate cluster.

        Parameters
        ----------
        labels : list or array
            List or one-dimensional array with same length as the pcd's number
            of points.
        return_ungroupped : boolean, optional
            If True, add ungroupped points to list. Default: False.

        Returns
        -------
        list
            Separated pointclouds.
        """
        labels = np.array(labels)

        pcd_separated = []
        for label in set(labels) - {-1}:
            idx = np.where(labels == label)[0]
            pcd_ = self.select_by_index(idx)
            pcd_separated.append(pcd_)

        if return_ungroupped:
            idx = np.where(labels == -1)[0]
            pcd_separated.append(self.select_by_index(idx))

        return pcd_separated

    def segment_by_position(self, shape, min_points=1):
        """Uniformly divide pcd into different subsets based purely on position.

        Parameters
        ----------
        shape : list or tuple of 3 elements
            Defines how many subdividions along the x, y and z axes respectively.
        min_points : int, optional
            Minimum number of points to form a cluster. Default: 1

        Returns
        -------
        list
            Subdivided pointclouds
        """
        if min_points < 1 or not isinstance(min_points, int):
            raise ValueError(f"min_points must be positive integer, got {min_points}")

        if len(shape) != 3:
            raise ValueError(f"shapes must have 3 elements, got {shape}")

        for n in shape:
            if n < 1 or not isinstance(n, int):
                raise ValueError(
                    f"shapes must only contain positive integers, got {shape}"
                )

        aabb = self.get_axis_aligned_bounding_box()
        min_bound = aabb.min_bound
        max_bound = aabb.max_bound
        delta = (max_bound - min_bound) / shape
        pcds = []
        for ix, iy, iz in itertools.product(*[range(n) for n in shape]):
            min_bound_sub = min_bound + delta * (ix, iy, iz)
            max_bound_sub = min_bound + delta * (ix + 1, iy + 1, iz + 1)
            aabb_sub = AxisAlignedBoundingBox(min_bound_sub, max_bound_sub)
            pcd_sub = self.crop(aabb_sub)
            if len(pcd_sub.points) >= min_points:
                pcds.append(pcd_sub)
        return pcds

    def segment_kmeans_colors(self, n_clusters=2, n_init="auto", **options):
        """Segment pointcloud according to the colors by using KMeans.

        For other options, see:
            sklearn.cluster._kmeans.KMeans

        Parameters
        ----------
        n_clusters : int, optional
          The number of clusters to form as well as the number of
          centroids to generate. Default=2.
        n_init : 'auto' or int,
            Number of times the k-means algorithm is run with different centroid
            seeds. The final results is the best output of `n_init` consecutive runs
            in terms of inertia. Several runs are recommended for sparse
            high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

            When `n_init='auto'`, the number of runs depends on the value of init:
            10 if using `init='random'` or `init` is a callable;
            1 if using `init='k-means++'` or `init` is an array-like.
            Default = 'auto'.

        Returns
        -------
        list
            Segmented pointcloud clusters.
        """
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, **options).fit(
            self.colors_cielab
        )
        return self.separate_with_labels(kmeans.labels_)

    def segment_dbscan(
        self, eps, min_points=1, print_progress=False, color_based=False
    ):
        """Get PointCloud points, label it according to Open3D's cluster_dbscan
        implementation and then return a list of segmented pointclouds.

        Parameters
        ----------
        eps : float
            Density parameter that is used to find neighbouring points.
        min_points : int, optional
            Minimum number of points to form a cluster. Default: 1
        print_progress : bool, optional
            If true the progress is visualized in the console. Default=False

        Returns
        -------
        list
            Segmented pointcloud clusters.
        """
        if min_points < 1 or not isinstance(min_points, int):
            raise ValueError(f"min_points must be positive integer, got {min_points}")

        if color_based:
            pcd_clustering = PointCloud(self.colors_cielab)
        else:
            pcd_clustering = self

        labels = pcd_clustering.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=print_progress
        )

        return self.separate_with_labels(labels)

    def segment_curvature_threshold(self, std_ratio=0.1, distance_threshold=0):
        """Remove borders by separating points with high and low curvature.

        The cutoff threshold used to separate points of high and low curvature
        is defined as:
            threshold = mean + std * str_ratio

        Where the mean and str are calculated from the pointcloud's curvature
        values.

        Parameters
        ----------
        std_ratio : float, optional
            Defines the cutoff threshold. Default: 0.1.

        print_progress : bool, optional
            If true the progress is visualized in the console. Default=False

        distance_threshold : float, optional
            If bigger than 0, also assumes points close to high curvature
            points as high curvature. Default: 0.

        Returns
        -------
        list
            Segmented pointclouds.
        """

        if distance_threshold < 0:
            raise ValueError(
                "distance_threshold must be a non-negative number, got {distance_threshold}"
            )

        if not self.has_curvature:
            warnings.warn("PointCloud does not have curvature, calculating...")
            self.estimate_curvature()

        mean = np.mean(self.curvature)
        std = np.std(self.curvature)

        threshold = mean + std * std_ratio

        indices = np.where(self.curvature < threshold)[0]
        pcd_low = self.select_by_index(indices)
        pcd_high = self.select_by_index(indices, invert=True)

        if distance_threshold > 0:
            is_close = (
                pcd_low.compute_point_cloud_distance(pcd_high) <= distance_threshold
            )
            new_indices = np.where(is_close)[0]
            pcd_high += pcd_low.select_by_index(new_indices)
            pcd_low = pcd_low.select_by_index(new_indices, invert=True)

        return pcd_low, pcd_high

    def segment_with_region_growing(
        self,
        residuals=None,
        mode="knn",
        k=20,
        radius=0,
        k_retest=10,
        threshold_angle=np.radians(10),
        min_points=10,
        cores=1,
        seeds_max=150,
        debug=False,
        old_divide=False,
    ):
        """Segment point cloud into multiple sub clouds according to their
        curvature by analying normals of neighboring points.

        Parameters
        ----------
        pcd : PointCloud
            Input point cloud.
        residuals : array, optional
            Array of residuals used to find the next random region, by selecting
            the unlabeled point with the least residual.
        mode : string, optional
            Decides mode of search. If 'knn', calculates the K nearest neighbors
            for testing. If 'radius', calculates points under a radius. Default:
            'knn'.
        k : int, optional
            Number of neighbors points to be tested at each time, if mode == 'knn'.
            Default: 20.
        radius : float, optional
            Radius to find neighbors, if mode == 'radius'.
        k_retest : int, optional
            Number of points after test to be added to seedlist. Default: 10.
        threshold_angle : float, optional
            Maximum angle (in radians) for neighboring points to be considered in
            the same region. Default = 0.17453 (10 degrees).
        min_points : positive int, optional
            Minimum of points necessary for each grouping. If bigger than 0, then
            every grouping smaller than the given value will be degrouped and added
            a last grouping.
        seeds_max : positive int, optional
            Max number of seeds in seedlist. Default: 150.
        cores : positive int, optional
            Number of cores. Default: 1.
        debug : boolean, optional
            If True, print information.

        Returns
        -------
        list
            Segmented pointclouds
        """
        if mode.lower() not in ["knn", "radius"]:
            raise ValueError(f"mode can be 'knn' or 'radius', got {mode}.")
        mode = mode.lower()

        if mode == "knn" and (k < 1 or not isinstance(k, int)):
            raise ValueError(
                "When mode == 'knn', k must be a positive int. " f"Got {k}."
            )

        if mode == "radius" and radius <= 0:
            raise ValueError(
                "When mode == 'radius', radius must be positive. " f"Got {radius}."
            )

        if residuals is None and len(self.curvature) == len(self.points):
            residuals = self.curvature

        def _get_labels_region_growing(
            pcd, residuals, k, k_retest, cos_thr, min_points, debug, i=None, data=None
        ):
            """Worker function for region growing segmentation"""
            if debug and i is not None:
                print(f"Process {i+1} entering...")

            pcd_tree = KDTreeFlann(pcd.as_open3d)
            num_points = len(pcd.points)
            labels = np.repeat(0, num_points)
            normals = np.asarray(pcd.normals)
            seedlist = []
            usedseeds = set()
            label = 0
            while sum(labels == 0) > 0:
                # if debug and i is None:
                if debug:
                    print(
                        f"{sum(labels == 0)} points to go, {len(seedlist)} in seed list"
                    )
                # available_residuals = residuals[labels == 0]
                # res_diff = abs(available_residuals - max(available_residuals) * rth_ratio)
                # rth_idx = np.where(res_diff == min(res_diff))[0][0]
                # rth = residuals[rth_idx]
                try:
                    seed = seedlist.pop()

                except IndexError:
                    label += 1
                    if residuals is None:
                        seed = np.where(labels == 0)[0][0]
                    else:
                        min_residuals = min(residuals[labels == 0])
                        seed = np.where(residuals == min_residuals)[0][0]
                    labels[seed] = label

                point = pcd.points[seed]
                normal = pcd.normals[seed]
                usedseeds.add(seed)

                if mode == "knn":
                    _, idx, b = pcd_tree.search_knn_vector_3d(point, k)
                elif mode == "radius":
                    _, idx, b = pcd_tree.search_radius_vector_3d(point, radius)
                else:
                    raise RuntimeError("This should never happen.")

                idx = np.array(idx)
                normals_neighbors = normals[idx]
                is_neighbor = np.abs(normals_neighbors.dot(normal)) > cos_thr
                idx = idx[is_neighbor]

                labels[idx] = label

                if k_retest > len(idx):
                    idx = idx[-k_retest - 1 : -1 : 1]
                idx = list(set(idx) - usedseeds)
                # if debug and i is None:
                if debug:
                    print(f"-- adding {len(idx)} points")

                if len(idx) > 0:
                    seedlist += idx
                    if len(seedlist) > seeds_max:
                        seedlist = seedlist[-1 - seeds_max : -1]

            if data is not None:
                # data = queue.get()
                data[i] = labels
                # data = {i: list(labels)}
                # queue.put(data)
                # queue.put([i]+list(labels))
                if debug:
                    print(f"Process {i+1} finished!")

            if debug and i is not None:
                print(f"Process {i+1} returning, {len(labels)} points labeled...")
            return labels

        def _separate_with_labels(pcd, labels):
            labels = np.asarray(labels)
            pcds_segmented = []
            pcd_rest = PointCloud()
            for label in set(labels):
                idx = np.where(labels == label)[0]
                pcd_group = pcd.select_by_index(idx)
                if len(idx) >= min_points:
                    pcds_segmented.append(pcd_group)
                else:
                    pcd_rest += pcd_group

            num_ungroupped = len(pcd_rest.points)
            if num_ungroupped > 0:
                pcds_segmented.append(pcd_rest)

            return pcds_segmented, num_ungroupped

        if min_points < 0 or not isinstance(min_points, int):
            raise ValueError(
                "min_points must be a non-negative int, got " f"{min_points}"
            )

        if cores < 1 or not isinstance(cores, int):
            raise ValueError("cores must be a positive int, got " f"{cores}")

        if cores > multiprocessing.cpu_count():
            warnings.warn(
                f"Only {multiprocessing.cpu_count()} available, {cores} required."
                " limiting to max availability."
            )
            cores = multiprocessing.cpu_count()

        if k_retest > k:
            raise ValueError(
                "k_retest must be smaller than k, got "
                f"got {k_retest} and {k} respectively"
            )

        cos_thr = np.cos(threshold_angle)

        time_start = time.time()

        if cores == 1:
            if debug == 1:
                print("For verbose debug, set debug = 2.")
            labels = _get_labels_region_growing(
                self, residuals, k, k_retest, cos_thr, min_points, debug=debug >= 2
            )

            pcds_segmented, num_ungroupped = _separate_with_labels(self, labels)

        else:
            if debug == 1:
                print("For verbose debug, set debug = 2.")

            if old_divide:
                a = int(np.ceil(cores ** (1 / 3)))
                b = int(np.sqrt(cores / a))
                c = int(cores / (a * b))
                if a * b * c > cores:
                    raise RuntimeError(
                        "Recomputed number of cores bigger than the "
                        "required number, this should never happen"
                    )
                cores = a * b * c
                pcds = self.segment_by_position([a, b, c], min_points=min_points)
                cores = len(pcds)

            else:
                pcds = [self]
                while len(pcds) < cores:
                    lengths = np.array([len(p.points) for p in pcds])
                    i = np.where(lengths == max(lengths))[0][0]
                    pcds += list(pcds.pop(i).split_in_half())

            if debug:
                print(f"Starting parallel with {cores} cores.")

            # Create processes and queues
            manager = multiprocessing.Manager()
            data = manager.dict()
            processes = [
                multiprocessing.Process(
                    target=_get_labels_region_growing,
                    args=(
                        pcds[i],
                        None,
                        k,
                        k_retest,
                        cos_thr,
                        min_points,
                        debug >= 2,
                        i,
                        data,
                    ),
                )
                for i in range(cores)
            ]

            # Start all processes
            # for process in [processes[1], processes[2], processes[3]]:
            for process in processes:
                process.start()
            if debug:
                print(f"All {cores} processes created, initializing...")

            # Wait until all processes are finished
            for process in processes:
                process.join()

            assert len(data) == cores

            if debug:
                print(f"All {cores} processes finished!")

            # data = {}
            # while(True):
            #     try:
            #         data = data | queue.get(False)
            #         # data[labels[0]] = labels[1:]
            #     except Empty:
            #         break

            # if debug:
            # print(f'Dict assembled!')

            pcds_segmented = []
            num_ungroupped = 0
            for i in data:
                ret = _separate_with_labels(pcds[i], data[i])
                pcds_segmented += ret[0]
                num_ungroupped += ret[1]

        if debug:
            m, s = divmod(time.time() - time_start, 60)
            h, m = divmod(m, 60)
            print(f"\n{len(pcds_segmented)} point clouds found")
            print(f"Algorithm took {m} minutes and {int(s)} seconds")
            print(f"{num_ungroupped} ungroupped points")

        return pcds_segmented

    def _get_points(points):
        if PointCloud.is_instance_or_open3d(points):
            return np.asarray(points.points)
        elif isinstance(points, np.ndarray):
            return points

        assert False  # shouldn't happen

    def find_closest_points_indices(self, other_points, n=1):
        """Finds pairs of closest points and returns indices.

        Parameters
        ----------
        other_points : N x 3 array or instance of PointCloud
            First pointcloud.
        n : int, optional
            Number of pairs. Default=1.

        Returns
        -------
        list of n tuples of indices
            Pairs of close points.
        """
        points1 = PointCloud._get_points(self)
        points2 = PointCloud._get_points(other_points)

        distances = cdist(points1, points2)
        min_distance_indices = np.unravel_index(
            np.argpartition(distances, n, axis=None)[:n], distances.shape
        )

        return min_distance_indices

    def find_closest_points(self, other_points, n=1):
        """Fuses list of pointclouds into a single open3d.geometry.PointCloud
        instance.

        Parameters
        ----------
        other_points : N x 3 array or instance of PointCloud
            First pointcloud.
        n : int, optional
            Number of pairs. Default=1.

        Returns
        -------
        list of n tuples of 2 points
            Pairs of close points.
        np.array
            Distances for each pair.
        """
        points1 = PointCloud._get_points(self)
        points2 = PointCloud._get_points(other_points)

        distances = cdist(points1, points2)
        min_distance_indices = np.unravel_index(
            np.argpartition(distances, n, axis=None)[:n], distances.shape
        )

        closest_points = []
        min_distances = []
        for i in range(n):
            closest_points.append(
                (
                    points1[min_distance_indices[0][i]],
                    points2[min_distance_indices[1][i]],
                )
            )
            min_distances.append(
                distances[min_distance_indices[0][i], min_distance_indices[1][i]]
            )

        return closest_points, min_distances

    def select_nearby_points(self, pcd, max_distance, cores=6):
        """
        Return PointCloud containing points that are close enough to at
        least some point in the inlier_points points array.

        Attention: Consider flattening both "grid" and "inlier_points" to the
        desired surface.

        See: get_rectangular_grid

        Parameters
        ----------
        pcd : PointCloud or numpy array
            PointCloud or Array of points (most likely, inlier points).
        max_distance : float
            Max distance allowed between grid points and inlier points.

        Returns
        -------
        numpy array
            Selected points.
        """
        if not isinstance(pcd, PointCloud):
            pcd = PointCloud(pcd)

        distance = self.compute_point_cloud_distance(pcd)
        selected_idxs = np.where(distance <= max_distance)[0]
        return self.select_by_index(selected_idxs)

    @classmethod
    def get_rectangular_grid(
        cls, vectors, center, grid_width, return_perimeter=False, grid_type="hexagonal"
    ):
        """Gives rectangular grid defined two vectors and its center.

        Vectors v1 and v2 should not be unit, and instead have lengths equivalent
        to widths of rectangle.

        Available grid types: "regular" and "hexagonal".

        If `return_perimeter` is set to True, also return the expected perimeter
        of each triangle.

        Parameters
        ----------
        vectors : arraylike of shape (2, 3)
            The two orthogonal unit vectors defining the rectangle plane.
        center : arraylike of length 3, optional
            Center of rectangle. If not given, either inliers or centroid are
            used.
        grid_width : float
            Distance between two points in first dimension (and also second
            dimension for regular grids).
        return_perimeter : boolean, optional
            If True, return tuple containing both grid and calculated perimeter.
            Default: False.
        grid_type : str, optional
            Type of grid, can be "hexagonal" or "regular". Default: "hexagonal".

        See also:
            select_grid_points

        Returns
        -------
        PointCloud
            Grid points

        float
            Perimeter of one triangle
        """
        if grid_type not in ["regular", "hexagonal"]:
            raise ValueError(
                "Possible grid types are 'regular' and 'hexagonal', "
                f"got '{grid_type}'."
            )
        eps = 1e-8
        lengths = np.linalg.norm(vectors, axis=1)

        # get unit vectors
        vx, vy = vectors / lengths[:, np.newaxis]

        n = []
        for length in lengths:
            ratio = length / grid_width
            n.append(int(np.floor(ratio)) + int(ratio % 1 > eps))

        def get_range(length, width):
            array = np.arange(stop=length, step=width) - length / 2
            # TODO: why? why is it failing?
            # assert abs(width - (array[1] - array[0])) < eps

            # adding last point if needed
            if (length / width) % 1 > eps:
                # array = np.hstack([array, array[-1] + grid_width])
                array = np.hstack([array, length / 2])

            return array

        if grid_type == "regular":
            array_x = get_range(lengths[0], grid_width)
            array_y = get_range(lengths[1], grid_width)

            x_ = vx * array_x[np.newaxis].T
            y_ = vy * array_y[np.newaxis].T
            # grid_lines = [px + py for px, py in product(x_, y_)]
            grid_lines = [x_ + py for py in y_]
            grid = np.vstack(grid_lines)

            perimeter = (2 + np.sqrt(2)) * grid_width

        elif grid_type == "hexagonal":
            h = grid_width * np.sqrt(3) / 2

            array_x = get_range(lengths[0], grid_width)
            array_y = get_range(lengths[1], h)

            x_ = vx * array_x[np.newaxis].T
            y_ = vy * array_y[np.newaxis].T
            # grid_lines = [px + py for px, py in product(x_, y_)]
            grid_lines = [x_ + py for py in y_]
            for i in range(len(grid_lines)):
                if i % 2 == 1:
                    grid_lines[i] += vx * grid_width / 2
                    grid_lines[i] = np.vstack(
                        [-vx * lengths[0] / 2, grid_lines[i][:-1]]
                    )

            perimeter = 3 * grid_width

        grid = np.vstack(grid_lines)
        pcd = cls(grid + center)

        if return_perimeter:
            return pcd, perimeter
        return pcd

    @classmethod
    def fuse_pointclouds(cls, pcds):
        """Fuses list of pointclouds into a single PointCloud instance.

        Parameters
        ----------
        pcds : list
            Input pointclouds.

        Returns
        -------
        PointCloud
            Fused point cloud.
        """
        pcd_full = open3d_PointCloud()

        if len(pcds) == 0:
            return pcd_full

        curvatures = []
        for pcd in pcds:
            pcd_full += pcd.as_open3d
            curvatures.append(pcd.curvature)
        pcd_full = cls(pcd_full)
        pcd_full.curvature = np.hstack(curvatures)
        return pcd_full
