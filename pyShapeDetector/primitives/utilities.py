#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:48:31 2023

@author: ebernardes
"""
import numpy as np
from itertools import combinations, compress

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

def fuse_shapes(shapes_lists, detector=None):
    # num_partitions = len(shapes_lists):
    new_list = []
    for sublist in shapes_lists:
        model = np.vstack([s.model for s in sublist]).mean(axis=0)
        shape = type(sublist[0])(model)
        
        points = np.vstack([s.inlier_points for s in sublist])
        normals = np.vstack([s.inlier_normals for s in sublist])
        shape.add_inliers(points, normals)
        
        if detector is not None:
            distances, angles = shape.get_residuals(points, normals)
            num_points = num_inliers = len(points)
            shape.metrics = detector.get_metrics(
                num_points, num_inliers, distances, angles)
            
        new_list.append(shape)
        
    return new_list