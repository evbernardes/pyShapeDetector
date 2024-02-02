#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:02:34 2023

@author: ebernardes
"""

from .multidetector import MultiDetector
from .primitivelimits import PrimitiveLimits
from .detector_options import DetectorOptions

from .helpers_pointclouds import (
    read_point_cloud, 
    paint_random, 
    segment_dbscan, 
    average_nearest_dist,
    segment_with_region_growing, 
    segment_dbscan, 
    segment_by_position, 
    fuse_pointclouds, 
    separate_pointcloud_in_two, 
    find_closest_points,
    alphashape_2d,
    polygonize_alpha_shape
    )

from .helpers_primitives import (
    get_rotation_from_axis, 
    group_similar_shapes, 
    fuse_shape_groups, 
    cut_planes_with_cylinders, 
    get_meshes, 
    fuse_similar_shapes, 
    glue_nearby_planes
    )

from .helpers_meshes import (
    get_triangle_perimeters,
    new_TriangleMesh,
    clean_crop, 
    paint_by_type, 
    fuse_meshes
    )
