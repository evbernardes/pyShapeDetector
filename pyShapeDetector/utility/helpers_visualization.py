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

def draw_geometries(elements, **args):
    from pyShapeDetector.primitives import Primitive, Line
    elements = np.asarray(elements).flatten()
    geometries = []
    lines = []
    for i, element in enumerate(elements):
        if isinstance(element, Line):
            # geometries.append(element.as_LineSet)
            lines.append(element)
        elif isinstance(element, Primitive):
            geometries.append(element.mesh)
        else:
            geometries.append(element)
            
    geometries.append(Line.get_LineSet_from_list(lines))
    visualization.draw_geometries(geometries, **args)
    
def draw_two_columns(objs_left, objs_right, dist=5, **camera_options):
                     # lookat=None, up=None, front=None, zoom=None):
                         
    lookat = camera_options.get('lookat')
    up = camera_options.get('get')
    front = camera_options.get('front')
    zoom = camera_options.get('zoom')
    
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
    
    if up and front:
        translate = 0.5 * dist * np.cross(up, front)
    else:
        translate = np.array([0, 0.5 * dist, 0])
        
    for i in range(len(objs_left)):
        objs_left[i].translate(-translate)
    for i in range(len(objs_right)):
        objs_right[i].translate(translate)
        
    if zoom is None:
        draw_geometries(objs_right + objs_left)
    else:
        draw_geometries(objs_right + objs_left,
                        lookat=lookat,
                        up=up,
                        front=front,
                        zoom=zoom
                        )