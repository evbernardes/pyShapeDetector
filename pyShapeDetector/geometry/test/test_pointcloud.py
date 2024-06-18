#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:33:55 2024

@author: ebernardes
"""
import numpy as np

from pyShapeDetector.primitives import Sphere
from pyShapeDetector.geometry import PointCloud


def test_distribute():
    spheres = [Sphere.random() for i in range(10)]

    for sphere in spheres:
        sphere.translate(np.random.random(3) * 10)
        sphere.set_inliers(sphere.sample_PointCloud_uniformly(100))

    pcds = [sphere.inliers for sphere in spheres]
    N_before = sum([len(pcd.points) for pcd in pcds])
    bbox = PointCloud.fuse_pointclouds(pcds).get_axis_aligned_bounding_box()
    bbox.color = (0, 0, 1)

    pcd_sampled = bbox.sample_PointCloud_uniformly(10000)
    pcd_sampled.distribute_to_closest(pcds)
    N_after = sum([len(pcd.points) for pcd in pcds])

    assert N_after == N_before + len(pcd_sampled.points)
