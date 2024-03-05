#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:30:28 2023

@author: ebernardes
"""
from pathlib import Path
import numpy as np

from pyShapeDetector.primitives import Cylinder
from pyShapeDetector.utility import read_point_cloud, draw_geometries
from pyShapeDetector.methods import RANSAC_Classic #, MSAC, BDSAC, LDSAC

#%% Parameters and input
filedir = Path('./data')
pcd = read_point_cloud(filedir / '1cylinders.pcd')
pcd.estimate_normals()
normals = pcd.normals

#%%
detector = RANSAC_Classic()
detector.options.inliers_min = 200
detector.options.num_iterations = 15
detector.options.threshold_angle = 30 * np.pi / 180
detector.options.inliers_min = 200

detector.add(Cylinder)

shape, inliers, metrics = detector.fit(pcd.points, debug=True, normals=normals)
pcd_shape = pcd.select_by_index(inliers)
pcd_rest = pcd.select_by_index(inliers, invert=True)
draw_geometries([pcd, shape])





















