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

# Creating random cylinder
shape_original = Cylinder.random()
shape_original.color = (0, 1, 0)

# Sampling points on surface
pcd = shape_original.sample_PointCloud_uniformly(1000)
pcd.estimate_normals()

# Creating instance of RANSAC fitter
detector = RANSAC_Classic()
detector.options.inliers_min = 200
detector.options.num_iterations = 15
detector.options.threshold_angle = 30 * np.pi / 180
detector.options.inliers_min = 200
detector.add(Cylinder)

# Using classic RANSAC to get Cylinder model from sampled points
shape, inliers, metrics = detector.fit(pcd, debug=True)
pcd_shape = pcd.select_by_index(inliers)
pcd_rest = pcd.select_by_index(inliers, invert=True)

# Plotting result and printing errors
draw_geometries([pcd, shape])
error_distance = np.linalg.norm(shape.center - shape_original.center)
error_length = np.abs(shape.length - shape_original.length)
cos_angle = np.abs(shape.axis.dot(shape_original.axis))
error_angle = np.arccos(np.clip(cos_angle, 0, 1)) * 180 / np.pi

print(f"distance error = {error_distance}")
print(f"length error = {error_length}")
print(f"angle error = {error_angle} degrees")
