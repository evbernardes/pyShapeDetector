#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:30:28 2023

@author: ebernardes
"""

import numpy as np

from pyShapeDetector.primitives import Cylinder
from pyShapeDetector.utility import draw_geometries
from pyShapeDetector.methods import RANSAC_Classic  # , MSAC, BDSAC, LDSAC

DEG = np.pi / 180

# Parameters and input
method = RANSAC_Classic

num_points = 10000
num_samples = int(num_points / 100)
noise_coef = 0.8

shape = Cylinder.from_center_half_vector_radius((0, 0, 0), (0, 0, 3), 2)
pcd = shape.sample_PointCloud_uniformly(num_points)

bb = shape.mesh.get_axis_aligned_bounding_box()
scale = np.linalg.norm(bb.max_bound - bb.min_bound)

pcd.estimate_normals()
normals = np.asarray(pcd.normals)

# Adding noise to half of cylinder
noise = 2 * (np.random.random(np.shape(pcd.points)) - 0.5) * noise_coef * scale
pcd.points += (pcd.points[:, 0] > 0)[:, np.newaxis] * noise
eps = pcd.average_nearest_dist(k=15)

# direct fit
shape_direct = Cylinder.fit(pcd.points, pcd.normals)
shape_direct.color = (0.8, 0.8, 0.8)

if shape_direct is None:
    print("no direct fit")

# RANSAC fit
detector = method()
detector.options.connected_components_eps = None
detector.options.inliers_min = 100
detector.options.max_sample_distance = 2 * eps
detector.add(Cylinder)

shape_ransac = detector.fit(pcd, debug=False)[0]
shape_ransac.color = (1, 0, 0)

draw_geometries(
    [pcd, shape_ransac, shape_direct], window_name="RANSAC in red, direct fit in gray"
)
