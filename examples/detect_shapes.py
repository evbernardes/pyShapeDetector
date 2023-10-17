#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:30:28 2023

@author: ebernardes
"""

import copy
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from open3d.visualization import draw_geometries
from open3d.utility import Vector3dVector

# from helpers import color_blue, color_gray, color_red, color_yellow
from helpers import draw_two_colomns
from pyShapeDetector.primitives import Sphere, Plane, Cylinder
from pyShapeDetector.utility import MultiDetector
from pyShapeDetector.methods import RANSAC_Classic, RANSAC_Weighted, MSAC, \
    BDSAC, LDSAC
    
methods = [RANSAC_Classic, 
           RANSAC_Weighted,
           MSAC,
           BDSAC, 
           LDSAC]

DEG = 0.017453292519943295

#%% Parameters and input
method = methods[4] # select which method, 3 means "BDSAC"
filedir = Path('./data')

filename = '3planes_3spheres_3cylinders'
# filename = '1cylinders'
# filename = '1spheres'
# filename = 'big'

noise_max = 1
inliers_min = 1000
num_iterations = 100
threshold_distance = 0.2 + noise_max

pcd_full = o3d.io.read_point_cloud(str((filedir / filename).with_suffix('.pcd')))
# draw_geometries([pcd_full])

noise = noise_max * np.random.random(np.shape(pcd_full.points))
pcd_full.points = Vector3dVector(pcd_full.points + noise)
# draw_geometries([pcd_full])

#%% Separate full pointcloud into clusters
labels = pcd_full.cluster_dbscan(eps=1.0, min_points=20)#, print_progress=True))

labels = np.array(labels)
max_label = labels.max()
pcd_segmented = copy.copy(pcd_full)
print(f"\nPoint cloud has {len(set(labels))} clusters!\n")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd_segmented.colors = Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd_segmented])

pcds_segmented = []
for label in set(labels):
    idx = np.where(labels == label)[0]
    pcds_segmented.append(pcd_full.select_by_index(idx))

#%% Create detectors for each shape

sphere_detector = method(Sphere, num_iterations=num_iterations,
                         threshold_angle=30 * DEG,
                         threshold_distance=threshold_distance,
                         max_point_distance=1,
                         model_max=Sphere.limit_radius(5),
                         inliers_min=inliers_min)

plane_detector = method(Plane, num_iterations=num_iterations,
                        threshold_angle=20 * DEG,
                        threshold_distance=threshold_distance,
                        # max_point_distance=0.5,
                        inliers_min=inliers_min)

cylinder_detector = method(Cylinder, num_iterations=num_iterations,
                           threshold_angle=20 * DEG,
                           threshold_distance=threshold_distance,
                           model_max=Cylinder.limit_radius(15),
                           # max_point_distance=0.5,
                           inliers_min=inliers_min)

detectors = [sphere_detector, plane_detector, cylinder_detector]

#%% Assemble detectors and detect shapes
print(f'Using {method._type} method')
shape_detector = MultiDetector(detectors, pcds_segmented, points_min=500, num_iterations=20)
shape_detector.run(debug=True)

#%% Plot detected meshes
meshes = shape_detector.meshes_detected
pcds_rest = shape_detector.pcds_rest

lookat=[0, 0, 1]
up=[0, 0, 1]
front=[1, 0, 0]
zoom=1
# zoom = None

bbox = pcd_full.get_axis_aligned_bounding_box()
delta = bbox.max_bound - bbox.min_bound

draw_two_colomns([pcd_full]+meshes, meshes+pcds_rest, 1.3*delta[1],
                 lookat, up, front, zoom)

