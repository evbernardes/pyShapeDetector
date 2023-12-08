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
