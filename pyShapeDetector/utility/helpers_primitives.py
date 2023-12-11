#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions that act on primitives.

Created on Tue Dec  5 15:48:31 2023

@author: ebernardes
"""
import numpy as np
from itertools import combinations, product
# from pyShapeDetector.primitives import Plane, Cylinder

def group_similar_shapes(shapes, rtol=1e-02, atol=1e-02):
    """ Detect shapes with similar model and group.
    
    See: fuse_shape_groups
    
    Parameters
    ----------
    shapes : list of shapes
        List containing all shapes.    
    rtol : float, optional
        The relative tolerance parameter. Default: 1e-02.
    atol : float, optional
        The absolute tolerance parameter. Default: 1e-02.
        
    Returns
    -------
    list of lists
        Grouped shapes.
    
    """
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
    """ Find weigthed average of shapes, where the weight is the fitness
    metric.
    
    If a detector is given, use it to compute the metrics of the resulting
    average shapes.
    
    See: group_similar_shapes
    
    Parameters
    ----------
    shapes_lists : list of list shapes
        Grouped shapes.
    detector : instance of some Detector, optional
        Used to recompute metrics.
        
    Returns
    -------
    list
        Average shapes.    
    """
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
    """ Isolates planes and cylinders. For every plane and cylinder 
    combination, check if cylinder cuts plane and, if it does, add a hole.
    
    Parameters
    ----------
    shapes : list of shapes
        List containing all shapes.
    radius_min : float
        Only isolates cylinders with radius below this threshold.
    total_cut : boolean, optional
        When True, only accepts cuts when either the top of the bottom 
        completely cuts the plane. Default: False.
    eps : float, optional
        Adds some backlash to top and bottom of cylinder. Default: 0.
    
    """
    planes = [s for s in shapes if s.name == 'plane' or s.name == 'bounded plane']
    cylinders = [s for s in shapes if s.name == 'cylinder' and s.radius < radius_min + eps]

    for c, p in product(cylinders, planes):
        if c.cuts(p, total_cut=total_cut, eps=eps):
            p.add_holes(c.project_to_plane(p))
                
def get_meshes(shapes, crop_types=['sphere', 'cone']):
    """ Returns meshes from shapes.
    
    Parameters
    ----------
    crop_types : list of strings, optional
        Define type of primitives that should be cropped according to their
        inlier points. Default: ['sphere', 'cone']
    
    Returns
    -------
    list
        List of meshes.
    """
    meshes = []
    for shape in shapes:
        if shape.name in crop_types:
            mesh = shape.get_cropped_mesh()
        else:
            mesh = shape.get_mesh()
        mesh.paint_uniform_color(np.random.random(3))
        # mesh.paint_uniform_color(np.random.random(3))
        meshes.append(mesh)

    return meshes
                
