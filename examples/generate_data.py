#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:52:20 2023

@author: ebernardes
"""

from pathlib import Path
import open3d as o3d
from open3d.geometry import PointCloud
from open3d.visualization import draw_geometries

from helpers import color_gray
from helpers import get_random_spheres, get_random_planes
            
#%% Parameters and input
filedir = Path('./data')

num_points = 2000
translate_lim = [-15, 15]

num_planes = 3
size_lim = [5, 10]
num_spheres = 2
radius_lim = [1, 7]

# noise = 0.1

filename = filedir / f'{num_spheres}spheres_{num_planes}planes.pcd'

#%% Creation of example PCDs

pcd_full = PointCloud()

pcd_planes, model_planes = get_random_planes(
    num_planes, translate_lim, size_lim, num_points=num_points)

pcd_spheres, model_spheres = get_random_spheres(
    num_spheres, translate_lim, radius_lim, num_points=num_points)

for pcd in pcd_planes+pcd_spheres:
    pcd_full += pcd

pcd_full.paint_uniform_color(color_gray)
pcd_full.estimate_normals()
# pcd_full.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=30))
draw_geometries([pcd_full])

#%%
val = input(f'Save as {filename}? (y)es, (N)o, (o)ther name. ').lower()
if val == 'y':
    o3d.io.write_point_cloud(str(filename), pcd_full)
elif val == 'o':
    filename = filedir / input('Enter filename:\n') / '.pcd'
else:
    pass

