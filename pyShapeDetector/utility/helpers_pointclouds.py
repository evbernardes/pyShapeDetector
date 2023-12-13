#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions that act on pointclouds.

Created on Tue Nov 14 16:40:48 2023

@author: ebernardes
"""
import numpy as np
import h5py
from pathlib import Path
# import copy
import open3d as o3d
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector

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
            
            
def segment_with_region_growing(pcd, residuals=None, k=20, 
                                threshold_angle=np.radians(10), debug=False):
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
    threshold_angle : float, optional
        Maximum angle (in radians) for neighboring points to be considered in
        the same region. Default = 0.17453 (10 degrees).
    debug : boolean, optional
        If True, print information .
        
    Returns
    -------
    list
        Segmented pointclouds
    """
    
    cos_thr = np.cos(threshold_angle)
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
        # idx = list(idx)
        # idx.remove(usedseeds)
        idx = list(set(idx) - usedseeds)
        print(f'-- adding {len(idx)} points')
        seedlist += idx
        
    pcds_segmented = []
    for label in set(labels):
        idx = np.where(labels == label)[0]
        pcds_segmented.append(pcd.select_by_index(idx))
    
    if debug:
        print(f'\n {len(pcds_segmented)} point clouds found')
    return pcds_segmented
