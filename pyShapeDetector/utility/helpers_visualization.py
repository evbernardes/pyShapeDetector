#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:59:02 2024

@author: ebernardes
"""
import numpy as np
import copy
from open3d import visualization
from pyShapeDetector.geometry import PointCloud
# from pyShapeDetector.primitives import Primitive, Line

def paint_random(elements, paint_inliers=False):
    """ Paint each pointcloud/mesh with a different random color.
    
    Parameters
    ----------
    elements : list of geomery elements
        Elements to be painted
    """
    
    from pyShapeDetector.primitives import Primitive
    
    if not isinstance(elements, list):
        elements = [elements]

    for element in elements:
        color = np.random.random(3)
        if isinstance(element, Primitive):
            element._color = color
            if paint_inliers:
                element._inlier_colors[:] = color
        else:
            element.paint_uniform_color(color)

def _treat_up_normal(camera_options):
    
    normal = camera_options.pop('normal', None)
    up = camera_options.get('up', None)
    
    if normal is not None and  up is not None:
            raise ValueError("Cannot enter both 'up' and 'normal'")
    
    elif normal is not None:
        x = np.cross(np.random.random(3), normal)
        x /= np.linalg.norm(x)
        # camera_options['up'] = np.cross(normal, x)
        camera_options['up'] = x
    
    elif 'up' in camera_options:
        camera_options['up'] = up

def draw_geometries(elements, **camera_options):
    
    from pyShapeDetector.primitives import Primitive, Line, Plane, PlaneBounded

    # _treat_up_normal(camera_options)
    _ = camera_options.pop('dist', None)
        
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
    
    if not isinstance(elements, (list, tuple)):
        elements = [elements]
    
    for element in elements:
        if hasattr(element, 'as_open3d'):
            geometries.append(element.as_open3d)
        elif isinstance(element, Line):
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
            pcd = PointCloud.from_points_normals_colors(element)
            geometries.append(pcd.as_open3d)
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
                  
    # _treat_up_normal(camera_options)       
    lookat = camera_options.pop('lookat', None)
    up = camera_options.pop('up', None)
    front = camera_options.pop('front', None)
    zoom = camera_options.pop('zoom', None)
    
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