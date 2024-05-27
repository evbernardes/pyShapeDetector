#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:02:34 2023

Collection of useful methods.

PointCloud-related 
------------------
new_PointCloud
write_point_cloud
read_point_cloud
paint_random
average_nearest_dist
segment_with_region_growing
segment_dbscan
segment_by_position
fuse_pointclouds
separate_pointcloud_in_two
find_closest_points_indices
find_closest_points
alphashape_2d
polygonize_alpha_shape

Math-related
-----------------
get_rotation_from_axis
    
Visualization-related
---------------------
paint_random
draw_geometries
draw_two_columns

Input/Output-related
--------------------
check_existance
save_elements
save_ask
ask_and_save

@author: ebernardes
"""

from .multidetector import MultiDetector
from .primitivelimits import PrimitiveLimits
from .detector_options import DetectorOptions

from .helpers_math import (
    get_rotation_from_axis
    )

from .helpers_visualization import (
    paint_random,
    draw_geometries,
    draw_two_columns
    )

from .helpers_io import (
    check_existance,
    save_elements,
    save_ask,
    ask_and_save,
    )
