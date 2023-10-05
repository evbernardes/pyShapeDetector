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

# from helpers import color_blue, color_gray, color_red, color_yellow
from pyShapeDetector.primitives import Sphere, Plane
from pyShapeDetector.methods import RANSAC_Classic, RANSAC_Weighted, MSAC, BDSAC
methods = [RANSAC_Classic, 
           RANSAC_Weighted,
           MSAC,
           BDSAC]

#%% Parameters and input
method = methods[2]
filedir = Path('./data')
filename = '2spheres_3planes'
pcd_full = o3d.io.read_point_cloud(str((filedir / filename).with_suffix('.pcd')))
# draw(pcd_full)

# Detection of clusters
labels = pcd_full.cluster_dbscan(eps=0.9, min_points=50)#, print_progress=True))

labels = np.array(labels)
max_label = labels.max()
pcd_segmented = copy.copy(pcd_full)
print(f"\nPoint cloud has {len(set(labels))} clusters!\n")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd_segmented.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd_segmented])

pcds_segmented = []
for label in set(labels):
    idx = np.where(labels == label)[0]
    pcds_segmented.append(pcd_full.select_by_index(idx))

#%%
inliers_min = 300

sphere_detector = method(Sphere, 
                         threshold_distance=0.1, 
                         ransac_n=4, 
                         num_iterations=50, probability=0.9, 
                         model_max=[None, None, None, 10],
                         threshold_angle=5,
                         inliers_min=inliers_min)

plane_detector = method(Plane, 
                        threshold_distance=0.1,
                        ransac_n=3, 
                        num_iterations=50, probability=0.9, 
                        threshold_angle=1,
                        max_point_distance=1,
                        inliers_min=inliers_min)

detectors = [sphere_detector, plane_detector]

shapes_detected = []
meshes_detected = []
pcds = []
print(f'Testing with {method._type}\n')
start = time.time()
# pcd_rest = copy.copy(pcd_full)
for idx in range(len(pcds_segmented)):
    print(f'Testing cluster {idx+1}...')
    pcd_ = copy.copy(pcds_segmented[idx])
    
    iteration = 0
    while(len(pcd_.points) > 500 and iteration < 20):
        print(f'iteration {iteration}')
        # pcd_.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = pcd_.normals
        
        output_shapes = []
        output_fitness = []
        output_inliers = []
        output_info = []
        for detector in detectors:
            shape, inliers, info = detector.fit(
                pcd_.points, debug=True, filter_model=False, normals=normals)
            
            output_shapes.append(shape)
            output_inliers.append(inliers)
            output_info.append(info)
            output_fitness.append(info['fitness'])
            
        if np.all(np.array(output_shapes) == None):
            print('No shapes found anymore, breaking...')
            break
        
        max_fitness = max(output_fitness)
        if max_fitness < 0.1:
            print('Fitness to small, breaking...')
            break
        
        idx = np.where(np.array(output_fitness) == max_fitness)[0][0]
        shape = output_shapes[idx]
        inliers = output_inliers[idx]
        print(f'- {shape._name} found!')

        pcd_primitive = pcd_.select_by_index(inliers)
        pcd_ = pcd_.select_by_index(inliers, invert=True)
            
        mesh = shape.get_mesh(pcd_primitive.points)
        # mesh = shape.get_square_mesh(pcd_primitive)
        mesh.paint_uniform_color(np.random.random(3))
        meshes_detected.append(mesh)
        
        window_name = f'{shape._name}:{shape.model}, {len(inliers)} inliers'
        
        shapes_detected.append({
            'shape': shape,
            'pcd_primitive': pcd_primitive,
            'mesh': mesh})
        iteration += 1
            
    pcds.append(pcd_)

print(f'{method._type} finished after {time.time() - start:.5f}s')
#%%
# draw_geometries(pcds_segmented,
#                 lookat=[0, 0, 1],
#                 up=[0, 0, 1],
#                 front=[1, 0, 0],
#                 zoom=1)

draw_geometries([pcd_full]+meshes_detected,
                lookat=[0, 0, 1],
                up=[0, 0, 1],
                front=[1, 0, 0],
                zoom=1)
    
draw_geometries(meshes_detected+pcds,
                lookat=[0, 0, 1],
                up=[0, 0, 1],
                front=[1, 0, 0],
                zoom=1)

