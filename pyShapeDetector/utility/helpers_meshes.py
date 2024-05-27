#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions that act on meshes.

Created on Wed Dec  6 14:59:38 2023

@author: ebernardes
"""
import warnings
import copy
import open3d as o3d
import numpy as np
from itertools import product
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from open3d.geometry import TriangleMesh
from open3d.utility import Vector3dVector, Vector3iVector
from open3d.visualization import draw_geometries_with_key_callbacks
from .helpers_primitives import fuse_shape_groups

# def _get_vertices_triangles(mesh_or_vertices, triangles=None):
#     """ Helper function to deal with inputs. """
#     if isinstance(mesh_or_vertices, TriangleMesh):
#         if triangles is not None:
#             raise ValueError("Input is either a single mesh, or an array of "
#                              "vertices and one array of triangles.")
#         vertices = np.asarray(mesh_or_vertices.vertices)
#         triangles = np.asarray(mesh_or_vertices.triangles)
#     else:
#         if triangles is None:
#             raise ValueError("If vertices are given as first input instead of "
#                              "a mesh, then triangles must be given.")
#         vertices = np.asarray(mesh_or_vertices)
#         triangles = np.asarray(triangles)
#     return vertices, triangles

# def remove_big_triangles(vertices, triangles, radius_ratio):
#     vertices = np.asarray(vertices)
#     triangles = np.asarray(triangles)
    
#     from pyShapeDetector.utility import get_triangle_circumradius
    
#     circumradiuses = get_triangle_circumradius(vertices, triangles)
#     mean_circumradiuses = np.mean(circumradiuses)
#     return triangles[circumradiuses < radius_ratio * mean_circumradiuses]

def planes_ressample_and_triangulate(planes, density, radius_ratio=None, double_triangles=False):
    from pyShapeDetector.primitives import PlaneBounded
    
    if isinstance(planes, PlaneBounded):
        planes = [planes]
    else:
        for plane in planes:
            if not isinstance(plane, PlaneBounded):
                raise ValueError("Input must one or more instances of "
                                 "PlaneBounded.")
                
                
    vertices = []
    for p in planes:
        N = int(np.ceil(density * len(p.inlier_points)))
    #     print(N)
        vertices.append(p.sample_points_uniformly(N))
    vertices = np.vstack(vertices)
    
    # number_of_points = int(self.surface_area * density)
    # vertices = [p.sample_points_uniformly(int(len(p.inlier_points) * density)) for p in planes]
    # vertices = [p.sample_points_density(density).points for p in planes]
    # vertices = np.vstack(vertices)
    
    # from pyShapeDetector.utility import fuse_shape_groups
    plane_fused = PlaneBounded.fuse(planes, ignore_extra_data=True, force_concave=False)

    projections = plane_fused.get_projections(vertices)
    triangles = Delaunay(projections).simplices
    
    if radius_ratio is not None:
        circumradiuses = TriangleMesh(vertices, triangles).get_triangle_circumradius()
        triangles = triangles[circumradiuses < radius_ratio * np.mean(circumradiuses)]
        
    if double_triangles:
        triangles = np.vstack([triangles, triangles[:, ::-1]])

    return vertices, triangles

def planes_ressample_and_triangulate_gui(planes, translation_ratio = 0.055,
                                         double_triangles=False):
    
    meshes = [plane.get_mesh() for plane in planes]

    plane_fused = fuse_shape_groups(
        [planes], detector=None, line_intersection_eps=1e-3)[0]
    
    density = 0
    all_points = plane_fused.inlier_points_flattened
    all_projections = plane_fused.get_projections(all_points)
    all_triangles = Delaunay(all_projections).simplices
    
    # all_points = np.vstack(
    #     [p.sample_points_uniformly(density * len(p.inlier_points)) for p in planes])
    # all_projections = planes[0].get_projections(all_points)
    # all_triangles = Delaunay(all_projections).simplices
    # vertices = all_points
    
    # perimeters = util.get_triangle_perimeters(all_points, triangles)
    circumradius = TriangleMesh(all_points, all_triangles).get_triangle_circumradius()
    # mean_circumradius = np.mean(circumradius)
    
    global data
    # viss = 0
    
    window_name = """ 
        (W): increase cutout limit. 
        (S): decrease cutout limit. 
        (A): add point density.
        (D): reduce point density.
        """
    
    data = {
        'original_vertices': all_points,
        'original_triangles': all_triangles,
        'density': density,
        'mean_circumradius': np.mean(circumradius),
        'circumradius': circumradius,
        'vertices': all_points,
        'triangles': all_triangles,
        'radius_ratio': 1,
        'color': (1, 0, 0),
        'translation': translation_ratio * plane_fused.normal,
        'plotted': False}
    
    triangles = data['triangles'][
        data['circumradius'] < data['radius_ratio'] * data['mean_circumradius']]
    data['current'] = TriangleMesh(all_points, triangles)
    data['current'].paint_uniform_color(data['color'])
    data['current'].translate(data['translation'])
    data['lines'] = TriangleMesh(data['vertices']+data['translation'], data['triangles']).get_triangle_LineSet()
    
    def update_geometries(vis):
        global data
        print(f"density: {data['density']}, radius_ratio: {data['radius_ratio']}")
        
        triangles = data['triangles'][
            data['circumradius'] < data['radius_ratio'] * data['mean_circumradius']]
        
        data['old'] = data['current']
        data['current'] = TriangleMesh(data['vertices'], triangles)
        data['current'].paint_uniform_color(data['color'])
        data['current'].translate(data['translation'])
        
        vis.add_geometry(data['current'])
        vis.remove_geometry(data['old'])
        
        return False
    
    def ratio_increase(vis):
        global data
        data['radius_ratio'] += 0.1
        # data['radius_ratio'] *= 2
        update_geometries(vis)
    
    def ratio_decrease(vis):
        global data
        data['radius_ratio'] -= 0.1
        # data['radius_ratio'] /= 2
        data['radius_ratio'] = max(data['radius_ratio'], 0)
        update_geometries(vis)
        
    def update_density(vis):
        global data
        
        if data['density'] == 0:
            data['vertices'] = data['original_vertices']
            data['triangles'] = data['original_triangles']
        else:
            data['vertices'], data['triangles'] = planes_ressample_and_triangulate(
                planes, data['density'])
            
        data['circumradius'] = TriangleMesh(
            data['vertices'], data['triangles']).get_triangle_circumradius
        data['mean_circumradius'] = np.mean(circumradius)
        
        vis.remove_geometry(data['lines'])
        data['lines'] = TriangleMesh(data['vertices']+data['translation'], data['triangles']).get_triangle_LineSet()
        vis.add_geometry(data['lines'])
        
    def density_increase(vis):
        global data
        data['density'] += 0.1
        update_density(vis)
        update_geometries(vis)
    
    def density_decrease(vis):
        global data
        data['density'] -= 0.1
        data['density'] = max(data['density'], 0.0)
        update_density(vis)
        update_geometries(vis)
    
    key_to_callback = {}
    key_to_callback[ord("W")] = ratio_increase
    key_to_callback[ord("S")] = ratio_decrease
    key_to_callback[ord("A")] = density_decrease
    key_to_callback[ord("D")] = density_increase
    # key_to_callback[ord("D")] = remove_last
    # key_to_callback[ord("R")] = load_render_option
    # key_to_callback[ord(",")] = capture_depth
    # key_to_callback[ord(".")] = capture_image
    
    draw_geometries_with_key_callbacks(
        meshes+[data['current'], data['lines']], 
        # [data['current'], data['lines']], 
        key_to_callback,
        window_name = window_name,
        # mesh_show_wireframe=True
        )
    
    triangles = data['triangles'][
        data['circumradius'] < data['radius_ratio'] * data['mean_circumradius']]
    
    if double_triangles:
        triangles = np.vstack([triangles, triangles[:, ::-1]])
    
    out_data = {
        'density': data['density'],
        'radius_ratio': data['radius_ratio'],
        'vertices': data['vertices'],
        'triangles': triangles}
    
    return out_data


