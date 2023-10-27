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
from pyShapeDetector.primitives import Sphere, Plane, Cylinder
from pyShapeDetector.utility import PrimitiveLimits, RANSAC_Options
from pyShapeDetector.methods import RANSAC_Classic, RANSAC_Weighted, MSAC, BDSAC, LDSAC

methods = [RANSAC_Classic, 
           RANSAC_Weighted,
           MSAC,
           BDSAC,
           LDSAC]



#%% Parameters and input
method = methods[0]
filedir = Path('./data')
filename = '1cylinders'
pcd = o3d.io.read_point_cloud(str((filedir / filename).with_suffix('.pcd')))
pcd.estimate_normals()
normals = pcd.normals

#%%

opt = RANSAC_Options()
opt.inliers_min = 200
opt.num_iterations = 15
opt.threshold_angle = 30 * np.pi / 180
opt.inliers_min = 200

detector = method(opt)
detector.add(Cylinder)
detector.add(Sphere)

shape, inliers, metrics = detector.fit(pcd.points, debug=True, normals=normals)
pcd_shape = pcd.select_by_index(inliers)
pcd_rest = pcd.select_by_index(inliers, invert=True)
mesh = shape.get_mesh(pcd_shape.points)
mesh.paint_uniform_color([1, 0, 0])

draw_geometries([pcd]+[mesh])
draw_geometries([pcd_rest]+[mesh])





















