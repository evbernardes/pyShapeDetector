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
from queue import Empty

def fuse_pointclouds(pcds):
    """ Fuses list of pointclouds into a single open3d.geometry.PointCloud 
    instance.
    
    Parameters
    ----------
    pcds : list
        Input pointclouds.
    
    Returns
    -------
    PointCloud
        Fused point cloud.
    """
    pcd_full = PointCloud()
    for pcd in pcds:
        pcd_full += pcd
    return pcd_full

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
    # pcd_segmented = copy.copy(pcd)
    # print(f"\nPoint cloud has {len(set(labels))} clusters!\n")
    if colors:
        pcd = copy.copy(pcd)
        max_label = max(labels)
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd_segmented])

    pcds_segmented = []
    for label in set(labels) - {-1}:
        idx = np.where(labels == label)[0]
        pcd_ = pcd.select_by_index(idx)
        # if len(pcd_.points) >= min_points:
        pcds_segmented.append(pcd_)
        
    return pcds_segmented

def separate_pointcloud_in_two(pcd):
    """ Divide pointcloud in two sub clouds, each one occupying the same 
    volume.
    
    Parameters
    ----------
    pcd : instance of Pointcloud
        Point cloud to be divided
        
    Returns
    -------
    tuple
        Subdivided pointclouds
    """
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    delta = (max_bound - min_bound)
    # i = np.where(delta == max(delta))[0][0]
    
    delta[np.where(delta != max(delta))[0]] = 0
    pcd_sub_1 = pcd.crop(
        AxisAlignedBoundingBox(min_bound, max_bound - delta/2))
    pcd_sub_2 = pcd.crop(
        AxisAlignedBoundingBox(min_bound + delta/2, max_bound))

    return pcd_sub_1, pcd_sub_2

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

def segment_with_region_growing(pcd, residuals=None, mode='knn', k=20, radius=0,
                                k_retest=10, threshold_angle=np.radians(10), 
                                min_points=10, cores=1, seeds_max = 150, 
                                debug=False, old_divide=False):
    """ Segment point cloud into multiple sub clouds according to their 
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
    if mode.lower() not in ['knn', 'radius']:
        raise ValueError(f"mode can be 'knn' or 'radius', got {mode}.")
    mode = mode.lower()
    
    if mode == 'knn' and (k < 1 or not isinstance(k, int)):
        raise ValueError("When mode == 'knn', k must be a positive int. "
                         f"Got {k}.")
                         
    if mode == 'radius' and radius <= 0:
        raise ValueError("When mode == 'radius', radius must be positive. "
                         f"Got {radius}.")
    
    def _get_labels_region_growing(
            pcd, residuals, k, k_retest, cos_thr, min_points, debug, i=None, data=None):
        """ Worker function for region growing segmentation"""
        if debug and i is not None:
            print(f'Process {i+1} entering...')
        
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        num_points = len(pcd.points)
        labels = np.repeat(0, num_points)
        normals = np.asarray(pcd.normals)
        seedlist = []
        usedseeds = set()
        label = 0
        while sum(labels == 0) > 0:
            # if debug and i is None:
            if debug:
                print(f'{sum(labels == 0)} points to go, {len(seedlist)} in seed list')
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
            
            if mode == 'knn':
                _, idx, b = pcd_tree.search_knn_vector_3d(point, k)
            elif mode == 'radius':
                _, idx, b = pcd_tree.search_radius_vector_3d(point, radius)
            else:
                raise RuntimeError("This should never happen.")
                
            idx = np.array(idx)
            normals_neighbors = normals[idx]
            is_neighbor = np.abs(normals_neighbors.dot(normal)) > cos_thr
            idx = idx[is_neighbor]
            
            labels[idx] = label
            
            if k_retest > len(idx):
                idx = idx[-k_retest-1:-1:1]
            idx = list(set(idx) - usedseeds)
            # if debug and i is None:
            if debug:
                print(f'-- adding {len(idx)} points')
            
            if len(idx) > 0:
                seedlist += idx
                if len(seedlist) > seeds_max:
                    seedlist = seedlist[-1-seeds_max:-1]
        
        if data is not None:
            # data = queue.get()
            data[i] = labels
            # data = {i: list(labels)}
            # queue.put(data)
            # queue.put([i]+list(labels))
            if debug:
                print(f'Process {i+1} finished!')

        if debug:
            print(f'Process {i+1} returning, {len(labels)} points labeled...')
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
    
    if cores == 1:
        if debug == 1:
            print('For verbose debug, set debug = 2.')
        labels = _get_labels_region_growing(
                pcd, residuals, k, k_retest, cos_thr, min_points, debug=debug > 2)
        
        pcds_segmented, num_ungroupped = _separate_with_labels(pcd, labels)
        
    else:
            
        if debug == 1:
            print('For verbose debug, set debug = 2.')
            
        if old_divide:
            a = int(np.ceil(cores ** (1/3)))
            b = int(np.sqrt(cores / a))
            c = int(cores / (a * b))
            if a * b * c > cores:
                raise RuntimeError('Recomputed number of cores bigger than the '
                                    'required number, this should never happen')
            cores = a * b * c
            pcds = segment_by_position(pcd, [a, b, c], min_points=min_points)
            cores = len(pcds)
        
        else:
            pcds = [pcd]
            while len(pcds) < cores:
                lengths = np.array([len(p.points) for p in pcds])
                i = np.where(lengths == max(lengths))[0][0]
                pcds += list(separate_pointcloud_in_two(pcds.pop(i)))
            
        if debug:
            print(f'Starting parallel with {cores} cores.')
        
        # Create processes and queues
        manager = multiprocessing.Manager()
        data = manager.dict()
        processes = [multiprocessing.Process(
            target=_get_labels_region_growing,
            args=(pcds[i], None, k, k_retest, cos_thr, min_points, debug > 2, i, data)) for i in range(cores)]
        
        # Start all processes
        # for process in [processes[1], processes[2], processes[3]]:
        for process in processes:
            process.start()
        if debug:
            print(f'All {cores} processes created, initializing...')
            
        # Wait until all processes are finished
        for process in processes:
            process.join()
            
        assert len(data) == cores
            
        if debug:
            print(f'All {cores} processes finished!')
        
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
        print(f'\n{len(pcds_segmented)} point clouds found')
        print(f'Algorithm took {m} minutes and {s} seconds')
        print(f'{num_ungroupped} ungroupped points')
        
    return pcds_segmented
