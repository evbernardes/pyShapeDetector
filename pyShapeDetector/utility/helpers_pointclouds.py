#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:40:48 2023

@author: ebernardes
"""
import numpy as np
import h5py
from pathlib import Path
import copy
import open3d as o3d
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector

def read_point_cloud(filepath):
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

def paint_random(pcds):
    if not isinstance(pcds, list):
        pcds.paint_uniform_color(np.random.random(3))
    else:
        for pcd in pcds:
            pcd.paint_uniform_color(np.random.random(3))

