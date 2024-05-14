#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:59:02 2024

@author: ebernardes
"""
import numpy as np
import copy
from open3d import visualization
from .helpers_pointclouds import new_PointCloud
# from pyShapeDetector.primitives import Primitive, Line

def draw_geometries(elements, **camera_options):
    from pyShapeDetector.primitives import Primitive, Line, Plane, PlaneBounded
    
    # elements = np.asarray(elements).flatten()
        
    try:
        draw_inliers = camera_options.pop('draw_inliers')
    except KeyError:
        draw_inliers = False
    
    try:
        draw_boundary_lines = camera_options.pop('draw_boundary_lines')
    except KeyError:
        draw_boundary_lines = False
    
    try:
        draw_planes = camera_options.pop('draw_planes')
    except KeyError:
        draw_planes = True
        
    pcds = []
    geometries = []
    lines = []
    boundary_lines = []
    hole_boundary_lines = []
    for element in elements:
        if isinstance(element, Line):
            lines.append(element)
        elif isinstance(element, Plane):
            if draw_planes:
                geometries.append(element.mesh)
        elif isinstance(element, Primitive):
            geometries.append(element.mesh)
        elif isinstance(element, np.ndarray):
            if len(element.shape) != 2 or element.shape[1] != 3:
                raise ValueError("3D arrays are interpreted as PointClouds, "
                                 "but they need to have a shape of (N, 3), got "
                                 f"{element.shape}.")
            geometries.append(new_PointCloud(element))
        else:
            geometries.append(element)
            
        if draw_inliers and isinstance(element, Primitive):
            pcds.append(element.inlier_PointCloud)
            
        if draw_boundary_lines and isinstance(element, PlaneBounded):
            boundary_LineSet = element.bound_LineSet
            boundary_LineSet.paint_uniform_color((1, 0, 0))
            boundary_lines.append(boundary_LineSet)
            
        if draw_boundary_lines and isinstance(element, Plane):
            for hole in element.holes:
                hole_boundary_LineSet = hole.bound_LineSet
                hole_boundary_LineSet.paint_uniform_color((0, 0, 1))
                hole_boundary_lines.append(hole_boundary_LineSet)
                
    
    if len(lines) > 0:
        geometries.append(Line.get_LineSet_from_list(lines))
        
    if draw_inliers:
        geometries += pcds
    
    if 'mesh_show_back_face' not in camera_options:
        camera_options['mesh_show_back_face'] = True
        
    if draw_boundary_lines:
        geometries += boundary_lines + hole_boundary_lines
        
    visualization.draw_geometries(geometries, **camera_options)
    
def draw_two_columns(objs_left, objs_right, dist=5, **camera_options):
                         
    lookat = camera_options.pop('lookat', None)
    up = camera_options.pop('up', None)
    front = camera_options.pop('front', None)
    zoom = camera_options.pop('zoom', None)
    # draw_inliers = camera_options.pop('draw_inliers', False)
    
    has_options = not any(v is None for v in [lookat, up, front, zoom])
    
    if not isinstance(objs_left, list):
        objs_left = [objs_left]
    
    if not isinstance(objs_right, list):
        objs_right = [objs_right]
    
    # precalculating meshes just in case
    for elem in objs_left+objs_right:
        try:
            elem.mesh
        except AttributeError:
            pass
        
    objs_left = copy.deepcopy(objs_left)
    objs_right = copy.deepcopy(objs_right)
    
    if has_options:
        translate = 0.5 * dist * np.cross(up, front)
    else:
        translate = np.array([0, 0.5 * dist, 0])
        
    for i in range(len(objs_left)):
        objs_left[i].translate(-translate)
    for i in range(len(objs_right)):
        objs_right[i].translate(translate)
        
    if not has_options:
        draw_geometries(objs_right + objs_left, **camera_options)
    else:
        draw_geometries(objs_right + objs_left,
                        lookat=lookat,
                        up=up,
                        front=front,
                        zoom=zoom,
                        **camera_options
                        )