#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:59:02 2024

@author: ebernardes
"""
import numpy as np
import copy
from open3d import visualization
# from pyShapeDetector.primitives import Primitive, Line

def draw_geometries(elements, print_points=False, **camera_options):
    from pyShapeDetector.primitives import Primitive, Line
    
    # print_points = args.get('print_points', False)
    
    elements = np.asarray(elements).flatten()
    pcds = []
    geometries = []
    lines = []
    for element in elements:
        if isinstance(element, Line):
            lines.append(element)
        elif isinstance(element, Primitive):
            geometries.append(element.mesh)
        else:
            geometries.append(element)
            
        if print_points and isinstance(element, Primitive):
            pcds.append(element.inlier_PointCloud)
            
    geometries.append(Line.get_LineSet_from_list(lines))
    if print_points:
        geometries += pcds
    
    if 'mesh_show_back_face' not in camera_options:
        camera_options['mesh_show_back_face'] = True
        
    visualization.draw_geometries(geometries, **camera_options)
    
def draw_two_columns(objs_left, objs_right, dist=5, **camera_options):
                     # lookat=None, up=None, front=None, zoom=None):
                         
    lookat = camera_options.get('lookat')
    up = camera_options.get('up')
    front = camera_options.get('front')
    zoom = camera_options.get('zoom')
    print_points = camera_options.get('print_points', False)
    
    has_options = not any(v is None for v in [lookat, up, front, zoom])
    
    # lookat = camera_options.get('lookat', [0, 0, 1])
    # up = camera_options.get('get', [0, 0, 1])
    # front = camera_options.get('front', [1, 0, 0])
    # zoom = camera_options.get('zoom', 1)
    
    if not isinstance(objs_left, list):
        objs_left = [objs_left]
    objs_left = copy.deepcopy(objs_left)
    if not isinstance(objs_right, list):
        objs_right = [objs_right]
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
        draw_geometries(objs_right + objs_left, print_points=print_points)
    else:
        draw_geometries(objs_right + objs_left, print_points=print_points,
                        lookat=lookat,
                        up=up,
                        front=front,
                        zoom=zoom
                        )