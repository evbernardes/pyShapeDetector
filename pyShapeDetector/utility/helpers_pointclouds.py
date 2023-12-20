#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions that act on pointclouds.

Created on Tue Nov 14 16:40:48 2023

@author: ebernardes
"""
from warnings import warn
import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import h5py
from pathlib import Path
import itertools
# import copy
import open3d as o3d
from open3d.geometry import PointCloud, AxisAlignedBoundingBox
from open3d.utility import Vector3dVector
from sklearn.neighbors import KDTree
import multiprocessing

def average_nearest_dist(points, k=15, leaf_size=40):
    """ Calculates the K nearest neighbors and returns the average distance 
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
    
    Returns
    -------
    PointCloud
        Loaded point cloud.
    """
    tree = KDTree(points, leaf_size=leaf_size)
    nearest_dist, nearest_ind = tree.query(points, k=k)
    return np.mean(nearest_dist[:, 1:])

def read_point_cloud(filepath):
    """ Read file to pointcloud. Can also read .h5 files from traceparts 
    database.
    
    Parameters
    ----------
    filepath : string or instance of pathlib.Path
        File to be loaded
        
    Returns
    -------
    PointCloud
        Loaded point cloud.
    """
    if isinstance(filepath, Path):
        filename = filepath.as_posix()
    else:
        filename = filepath
        filepath = Path(filename)

    print(filename)

    if filepath.suffix == '.h5':
        f = h5py.File(filepath, 'r')
        labels = np.asarray(f['gt_labels'])
        points = f['gt_points']
        normals = f['gt_normals']
        pcds = []
        for label in set(labels):
            idx = label == labels
            pcd = PointCloud()
            pcd.points = Vector3dVector(points[idx])
            pcd.normals = Vector3dVector(normals[idx])
            pcds.append(pcd)
        if len(pcds) == 1:
            pcds = pcds[0]
        f.close()

    else:
        pcds = o3d.io.read_point_cloud(filename)

    return pcds

def paint_random(elements):
    """ Paint each pointcloud/mesh with a different random color.
    
    Parameters
    ----------
    elements : list of geomery elements
        Elements to be painted
    """
    if not isinstance(elements, list):
        elements.paint_uniform_color(np.random.random(3))
    else:
        for element in elements:
            element.paint_uniform_color(np.random.random(3))
            
    
def segment_dbscan(pcd, eps, min_points=1, print_progress=False, colors=False):
    """ Read file to pointcloud, label it according to Open3D's cluster_dbscan
    implementation and then return a list of segmented pointclouds.
    
    Parameters
    ----------
    pcd : instance of Pointcloud
        Point cloud to be segmented
    eps : float
        Density parameter that is used to find neighbouring points.
    min_points : int, optional
        Minimum number of points to form a cluster. Default: 1
    print_progress : bool, optional
        If true the progress is visualized in the console. Default=False
    colors : bool, optional
        If true each cluster is uniformly painted with a different color. 
        Default=False
        
    Returns
    -------
    list
        Segmented pointcloud clusters.
    """
    if min_points < 1 or not isinstance(min_points, int):
        raise ValueError(
            f'min_points must be positive integer, got {min_points}')
        
    labels = pcd.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=print_progress)

    labels = np.array(labels)
    # max_label = labels.max()
    pcd_segmented = copy.copy(pcd)
    # print(f"\nPoint cloud has {len(set(labels))} clusters!\n")
    if colors:
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd_segmented.colors = Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd_segmented])

    pcds_segmented = []
    for label in set(labels):
        idx = np.where(labels == label)[0]
        pcds_segmented.append(pcd.select_by_index(idx))
        
    return pcds_segmented

def segment_by_position(pcd, shape, min_points=1):
    """ Uniformly divide pcd into different subsets based purely on position.
    
    Parameters
    ----------
    pcd : instance of Pointcloud
        Point cloud to be segmented
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
        raise ValueError(
            f'min_points must be positive integer, got {min_points}')
    
    if len(shape) != 3:
        raise ValueError(f'shapes must have 3 elements, got {shape}')
        
    for n in shape:
        if n < 1 or not isinstance(n, int):
            raise ValueError(
                f'shapes must only contain positive integers, got {shape}')
    
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    delta = (max_bound - min_bound) / shape
    pcds = []
    for ix, iy, iz in itertools.product(*[range(n) for n in shape]):
        min_bound_sub = min_bound + delta * (ix, iy, iz)
        max_bound_sub = min_bound + delta * (ix+1, iy+1, iz+1)
        bbox_sub = AxisAlignedBoundingBox(min_bound_sub, max_bound_sub)
        pcd_sub = pcd.crop(bbox_sub)
        if len(pcd_sub.points) >= min_points:
            pcds.append(pcd_sub)
    return pcds
            
def _segment_with_region_growing_worker(
        pcd, residuals, k, k_retest, cos_thr, min_points, debug):
    """ Worker function for region growing segmentation"""
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    num_points = len(pcd.points)
    labels = np.repeat(0, num_points)
    normals = np.asarray(pcd.normals)
    seedlist = []
    usedseeds = set()
    
    label = 0
    
    while sum(labels == 0) > 0:
        if debug:
            print(f'{sum(labels == 0)} points to go')
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
            
        _, idx, b = pcd_tree.search_knn_vector_3d(point, k)
        idx = np.array(idx)
        normals_neighbors = normals[idx]
        is_neighbor = np.abs(normals_neighbors.dot(normal)) > cos_thr
        idx = idx[is_neighbor]
        
        labels[idx] = label
        
        if k_retest > len(idx):
            idx = idx[-k_retest-1:-1:1]
        idx = list(set(idx) - usedseeds)
        if debug:
            print(f'-- adding {len(idx)} points')
        seedlist += idx
        
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

def segment_with_region_growing(pcd, residuals=None, k=20, k_retest=10,
                                threshold_angle=np.radians(10), min_points=10,
                                cores=1, debug=False):
    """ Segment point cloud into multiple sub clouds according to their 
    curvature by analying normals of neighboring points.
    
    Parameters
    ----------
    pcd : PointCloud
        Input point cloud.
    residuals : array, optional
        Array of residuals used to find the next random region, by selecting
        the unlabeled point with the least residual.
    k : int, optional
        Number of neighbors points to be tested at each time. Default: 20.
    k_retest : int, optional
        Number of points after test to be added to seedlist. Default: 10.
    threshold_angle : float, optional
        Maximum angle (in radians) for neighboring points to be considered in
        the same region. Default = 0.17453 (10 degrees).
    min_points : positive int, optional
        Minimum of points necessary for each grouping. If bigger than 0, then
        every grouping smaller than the given value will be degrouped and added
        a last grouping.
    cores : positive int, optional
        Number of cores. Default: 1.
    debug : boolean, optional
        If True, print information.
        
    Returns
    -------
    list
        Segmented pointclouds
    """
    if min_points < 0 or not isinstance(min_points, int):
        raise ValueError('min_points must be a non-negative int, got '
                         f'{min_points}')
                         
    if cores < 1 or not isinstance(cores, int):
        raise ValueError('cores must be a positive int, got '
                         f'{cores}')
        
    if cores > multiprocessing.cpu_count():
        warn(f'Only {multiprocessing.cpu_count()} available, {cores} required.'
              ' limiting to max availability.')
        cores = multiprocessing.cpu_count()
    
    if k_retest > k:
        raise ValueError('k_retest must be smaller than k, got '
                         f'got {k_retest} and {k} respectively')
        
    cos_thr = np.cos(threshold_angle)
        
    time_start = time.time()
    
    pcds_segmented, num_ungroupped = _segment_with_region_growing_worker(
            pcd, residuals, k, k_retest, cos_thr, min_points, debug)
    
    if debug:
        m, s = divmod(time.time() - time_start, 60)
        h, m = divmod(m, 60)
        print(f'\n{len(pcds_segmented)} point clouds found')
        print(f'Algorithm took {m} minutes and {s} seconds')
        print(f'{num_ungroupped} ungroupped points')
        
    return pcds_segmented
