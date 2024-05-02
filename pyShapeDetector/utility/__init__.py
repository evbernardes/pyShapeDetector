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

Mesh-related
------------
get_triangle_lines
get_triangle_LineSet
get_triangle_boundary_indexes
get_loop_indexes_from_boundary_indexes
simplify_loop_with_angle
get_triangle_points
get_triangle_sides
get_triangle_perimeters
get_triangle_surface_areas
get_triangle_circumradius
get_rectangular_grid
select_grid_points
new_TriangleMesh
clean_crop 
paint_by_type
fuse_vertices_triangles
fuse_meshes
remove_big_triangles
planes_ressample_and_triangulate
planes_ressample_and_triangulate_gui
triangulate_earclipping
    
Visualization-related
---------------------
draw_geometries
draw_two_columns

@author: ebernardes
"""

from .multidetector import MultiDetector
from .primitivelimits import PrimitiveLimits
from .detector_options import DetectorOptions

from .helpers_pointclouds import (
    new_PointCloud,
    write_point_cloud, 
    read_point_cloud, 
    paint_random,
    average_nearest_dist,
    segment_with_region_growing, 
    segment_dbscan, 
    segment_by_position, 
    fuse_pointclouds, 
    separate_pointcloud_in_two, 
    find_closest_points_indices,
    find_closest_points,
    alphashape_2d,
    polygonize_alpha_shape
    )

from .helpers_primitives import (
    get_rotation_from_axis, 
    # _get_partitions,
    # _get_partitions_legacy,
    group_similar_shapes,
    fuse_shape_groups, 
    cut_planes_with_cylinders, 
    get_meshes, 
    fuse_similar_shapes,
    find_plane_intersections,
    glue_planes_with_intersections,
    glue_nearby_planes
    )

from .helpers_meshes import (
    get_triangle_lines,
    get_triangle_LineSet,
    get_triangle_boundary_indexes,
    get_loop_indexes_from_boundary_indexes,
    simplify_loop_with_angle,
    get_triangle_points,
    get_triangle_sides,
    get_triangle_perimeters,
    get_triangle_surface_areas,
    get_triangle_circumradius,
    get_rectangular_grid,
    select_grid_points,
    new_TriangleMesh,
    clean_crop, 
    paint_by_type,
    fuse_vertices_triangles,
    fuse_meshes,
    remove_big_triangles,
    planes_ressample_and_triangulate,
    planes_ressample_and_triangulate_gui,
    triangulate_earclipping
    )

from .helpers_visualization import (
    draw_geometries,
    draw_two_columns
    )
