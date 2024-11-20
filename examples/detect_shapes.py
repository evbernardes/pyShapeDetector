#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:30:28 2023

@author: ebernardes
"""

from pathlib import Path
import numpy as np
import open3d as o3d

from pyShapeDetector.geometry import PointCloud
from pyShapeDetector.primitives import Sphere, Plane, PlaneBounded, Cylinder
from pyShapeDetector.utility import (
    MultiDetector,
    PrimitiveLimits,
    draw_geometries,
    get_painted,
)
from pyShapeDetector import methods

filedir = Path("./data")

# Choose RANSAC-based method:
# method = methods.RANSAC_Classic
# method = methods.RANSAC_Weighted
# method = methods.MSAC
method = methods.BDSAC
# method = methods.LDSAC

# Choose one of the possible example datasets:
# filename = "disjoint_planes.pcd"
# filename = '3planes_3spheres_3cylinders.pcd'
filename = "cylinders_box_sphere.noise_0.0.pcd"
# filename = '1planes_1spheres.pcd'
# filename = '1cylinders.pcd'
# filename = '1spheres.pcd'
# filename = 'big.pcd'

# Add noise:
noise_max = 0

pcd_full = PointCloud.read_point_cloud(filedir / filename)

# draw_geometries([pcd_full])
noise = noise_max * np.random.random(np.shape(pcd_full.points))
pcd_full.points += noise
pcd_full.estimate_normals()
draw_geometries(pcd_full, window_name="Input pointcloud with noise")

# Separate full pointcloud into clusters, if possible
eps = pcd_full.average_nearest_dist(k=15)
# pcds_segmented = pcd_full.segment_dbscan(2 * eps, min_points=10)
pcds_segmented = [pcd_full]
pcds_segmented = get_painted(pcds_segmented, color="random")
draw_geometries(pcds_segmented, window_name="Segmented pointclouds, one color for each")

# Create RANSAC-based fitter instance and set parameters
detector = method()
detector.options.inliers_min = 100
detector.options.threshold_distance = 1 * eps
detector.options.threshold_angle_degree = 10
detector.options.threshold_refit_ratio = 1
detector.options.num_iterations = 20
detector.options.probability = 0.9999999

# When using a big number of samples it can be useful to limit their distance
detector.options.max_point_distance = 3 * eps
detector.options.num_samples = 10

# Assures only connected inliers
detector.options.connected_components_eps = 10 * eps

# Adding possible primitives while limiting the radii
detector.add(Sphere, PrimitiveLimits(("radius", [None, 3])))
detector.add(PlaneBounded)
detector.add(Cylinder, PrimitiveLimits(("radius", [None, 6])))

shape_detector = MultiDetector(
    detector,
    pcds_segmented,
    debug=True,
    points_min=100,
    shapes_per_cluster=30,
    compare_metric="fitness",
    metric_min=0.1,
)

# Plot detected shapes and outlier points
draw_geometries(shape_detector.shapes + shape_detector.pcds_rest)
