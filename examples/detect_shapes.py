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
from open3d.visualization import draw_geometries
from open3d.utility import Vector3dVector

# from helpers import color_blue, color_gray, color_red, color_yellow
from helpers import draw_two_colomns, paint_meshes_by_type
from pyShapeDetector.primitives import Sphere, Plane, PlaneBounded, Cylinder
from pyShapeDetector.utility import MultiDetector, PrimitiveLimits
from pyShapeDetector.methods import RANSAC_Classic, RANSAC_Weighted, MSAC, \
    BDSAC, LDSAC

DEG = 0.017453292519943295

#%% Parameters and input
# method = RANSAC_Classic
# method = RANSAC_Weighted
# method = MSAC
# method = BDSAC
method = LDSAC

filedir = Path('./data')

# filename = 'disjoint_planes'
filename = '3planes_3spheres_3cylinders'
# filename = '1planes_1spheres'
# filename = '1cylinders'
# filename = '1spheres'
# filename = 'big'

noise_max = 1
inliers_min = 1000
num_iterations = 30
threshold_distance = 0.2 + 2 * noise_max
threshold_angle=35 * DEG
fullpath = (filedir / filename).with_suffix('.pcd')

pcd_full = o3d.io.read_point_cloud(str(fullpath))
# draw_geometries([pcd_full])
noise = noise_max * np.random.random(np.shape(pcd_full.points))
pcd_full.points = Vector3dVector(np.asarray(pcd_full.points) + noise)
# draw_geometries([pcd_full])

#%% Separate full pointcloud into clusters
idx = random.sample(range(len(pcd_full.points)), 1)[0]
dist = np.linalg.norm(pcd_full.points - pcd_full.points[idx], axis=1)
args = np.argsort(dist)[1:1 + 20]
eps = np.mean(dist[args]) * 2
# eps = 0 

#%%
labels = pcd_full.cluster_dbscan(eps=eps, min_points=10)#, print_progress=True))

labels = np.array(labels)
max_label = labels.max()
pcd_segmented = copy.copy(pcd_full)
print(f"\nPoint cloud has {len(set(labels))} clusters!\n")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd_segmented.colors = Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd_segmented])

pcds_segmented = []
for label in set(labels):
    idx = np.where(labels == label)[0]
    pcds_segmented.append(pcd_full.select_by_index(idx))
# pcds_segmented = [pcd_full]
#%%
limits = [
    PrimitiveLimits(('radius', 'max', 3)),
    PrimitiveLimits(('radius', 'max', 3)),
    None]

detector = method([Sphere, Cylinder, PlaneBounded], 
# detector = method([PlaneBounded], 
                  num_iterations=num_iterations,
                  threshold_angle=threshold_angle,
                  threshold_distance=threshold_distance,
                  # max_point_distance=0.5,
                   limits=limits,
                  inliers_min=inliers_min,
                  connected_components_density=None)

shape_detector = MultiDetector(detector, pcds_segmented, debug=True,
                               points_min=100, num_iterations=20,
                               compare_metric='fitness', metric_min=0.5)             

#%% Plot detected meshes
meshes = shape_detector.meshes
shapes = [shape.canonical for shape in shape_detector.shapes]
paint_meshes_by_type(meshes, shapes)

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

