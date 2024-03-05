#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:30:28 2023

@author: ebernardes
"""

import copy
import time
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from open3d.utility import Vector3dVector

from pyShapeDetector.primitives import Sphere, Plane, PlaneBounded, Cylinder
from pyShapeDetector import utility as util
from pyShapeDetector.methods import BDSAC#, RANSAC_Classic, MSAC, LDSAC

DEG = 0.017453292519943295

#%% Parameters and input
# method = RANSAC_Classic
# method = RANSAC_Weighted
# method = MSAC
method = BDSAC
# method = LDSAC

filedir = Path('./data')

# filename = 'disjoint_planes'
# filename = '3planes_3spheres_3cylinders'
filename = 'cylinders_box_sphere.noise_0.0'
# filename = '1planes_1spheres'
# filename = '1cylinders'
# filename = '1spheres'
# filename = 'big'



noise_max = 0
fullpath = (filedir / filename)

pcd_full = o3d.io.read_point_cloud(str(fullpath) + '.pcd')
# draw_geometries([pcd_full])
noise = noise_max * np.random.random(np.shape(pcd_full.points))
pcd_full.points = Vector3dVector(np.asarray(pcd_full.points) + noise)
# draw_geometries([pcd_full])

#%% Separate full pointcloud into clusters
eps = util.average_nearest_dist(pcd_full.points, k=15)
pcds_segmented = util.segment_dbscan(pcd_full, 2*eps, min_points=10, colors=True)
# o3d.visualization.draw_geometries(pcds_segmented)
# pcds_segmented = [pcd_full]
#%%

detector = method()
detector.options.inliers_min = 1000
# detector.options.threshold_distance = 0.2 + 1 * noise_max
detector.options.threshold_distance = 10 * eps
detector.options.threshold_angle=25 * DEG
detector.options.connected_components_density=None
detector.options.threshold_refit_ratio = 3
detector.options.num_iterations = 100
# detector.options.probability = 0.9999999
detector.options.probability = 1

detector.add(Sphere, util.PrimitiveLimits(('radius', [None, 3])))
detector.add(PlaneBounded)

shape_detector = util.MultiDetector(
    detector, pcds_segmented, debug=True,
    # normals_reestimate=True,
    points_min=100, shapes_per_cluster=20,
    compare_metric='fitness', metric_min=0.5)
    # compare_metric='weight', metric_min=5385206)             

#%% Plot detected meshes
shapes = shape_detector.shapes
# paint_meshes_by_type(meshes, shapes)

pcds_rest = shape_detector.pcds_rest

# lookat=[0, 0, 1]
# up=[0, 0, 1]
# front=[1, 0, 0]
# zoom=1
# zoom = None

bbox = pcd_full.get_axis_aligned_bounding_box()
delta = bbox.max_bound - bbox.min_bound

util.draw_two_columns(pcds_segmented+shapes, shapes+[pcds_rest], 1.3*delta[1])

