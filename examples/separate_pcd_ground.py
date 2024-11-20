#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:35:16 2023

@author: ebernardes
"""
import numpy as np
from pathlib import Path

from pyShapeDetector.primitives import PlaneBounded#, Sphere, Plane, Cylinder
from pyShapeDetector import utility as util
from pyShapeDetector import methods
    
def separate_pcd_ground(pcd_full, threshold_angle_degrees=20, downsample=50, 
                        draw=True, timer=None, method=methods.BDSAC):
    
    if timer is None:
        timer = util.Timer()

    # Detect plane with downsampled pointcloud
    plane_detector = method()
    plane_detector.options.downsample = downsample
    plane_detector.options.threshold_angle_degrees = threshold_angle_degrees
    plane_detector.add(PlaneBounded)

    print(f"{timer} - Detecting ground...")
    ground, inliers, ground_metrics = plane_detector.fit(pcd_full, set_inliers=True)
    print(len(inliers))
    print(f"{timer} - Done, {len(inliers)} inliers found!")

    # Remove ground from full pcd
    pcd_no_ground_full = pcd_full.select_by_index(inliers, invert=True)
    ground.set_bounds(pcd_full.points, flatten=True, convex=True) # Use inliers and outliers for boudaries, to assure nothing is "floating"

    camera_options = {
        'lookat': ground.inlier_mean.tolist(), 
        'up': ground.parallel_vectors[0].tolist(),
        'front': (ground.normal).tolist(), 
        'zoom': 0.8}

    c = np.cross(camera_options['up'], camera_options['front'])
    d = pcd_full.points.dot(c) / np.linalg.norm(c)
    camera_options['dist'] = (max(d) - min(d)) * 1.2

    if draw:
        util.draw_two_columns(pcd_no_ground_full, [pcd_no_ground_full, ground], **camera_options)

    return pcd_no_ground_full, ground, camera_options