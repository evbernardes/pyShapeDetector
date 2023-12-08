#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:48:31 2023

@author: ebernardes
"""
import numpy as np
from itertools import combinations
# from pyShapeDetector.primitives import Plane, Cylinder

def group_similar_shapes(shapes, rtol=1e-02, atol=1e-02):
    
    # repeated_idx = [True]*len(shapes)
    num_shapes = len(shapes)
    partitions = list(range(num_shapes))
    
    for i, j in combinations(partitions, 2):
        if partitions[j] != j:
            continue
        if shapes[i].is_similar_to(shapes[j], rtol=rtol, atol=atol):
            partitions[j] = i
    
    sublists = []
    for i in set(partitions):
        sublist = [shapes[j] for j in partitions if j == i]
        sublists.append(sublist)
        
    return sublists

def fuse_shape_groups(shapes_lists, detector=None):
    # num_partitions = len(shapes_lists):
    new_list = []
    for sublist in shapes_lists:
        fitness = [s.metrics['fitness'] for s in sublist]
        model = np.vstack([s.model for s in sublist])
        model = np.average(model, axis=0, weights=fitness)
        shape = type(sublist[0])(model)
        
        points = np.vstack([s.inlier_points for s in sublist])
        normals = np.vstack([s.inlier_normals for s in sublist])
        shape.add_inliers(points, normals)
        
        if detector is not None:
            num_points = sum([s.metrics['num_points'] for s in sublist])
            num_inliers = len(points)
            distances, angles = shape.get_residuals(points, normals)
            shape.metrics = detector.get_metrics(
                num_points, num_inliers, distances, angles)
            
        new_list.append(shape)
        
    return new_list

def cut_planes_with_cylinders(shapes, radius_min, total_cut=False, eps=0):
    planes = [s for s in shapes if s.name == 'plane' or s.name == 'bounded plane']
    cylinders = [s for s in shapes if s.name == 'cylinder']
    cylinders = [s for s in cylinders if s.radius < radius_min + eps]

    for c in cylinders:
        for p in planes:
            if c.cuts(p, total_cut=total_cut, eps=eps):
                p.add_holes(c.project_to_plane(p))
                
