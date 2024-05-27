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

Primitive-related
-----------------
get_rotation_from_axis
group_similar_shapes
fuse_shape_groups
cut_planes_with_cylinders 
get_meshes
fuse_similar_shapes
find_plane_intersections
glue_planes_with_intersections
glue_nearby_planes
    
Visualization-related
---------------------
draw_geometries
draw_two_columns

@author: ebernardes
"""

from .multidetector import MultiDetector
from .primitivelimits import PrimitiveLimits
from .detector_options import DetectorOptions

from .helpers_primitives import (
    get_rotation_from_axis,
    group_similar_shapes,
    fuse_shape_groups, 
    cut_planes_with_cylinders,
    fuse_similar_shapes,
    find_plane_intersections,
    glue_planes_with_intersections,
    glue_nearby_planes
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
