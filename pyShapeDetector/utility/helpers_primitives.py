#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions that act on primitives.

Created on Tue Dec  5 15:48:31 2023

@author: ebernardes
"""
import numpy as np
from itertools import combinations, product
from scipy.spatial.transform import Rotation
# from pyShapeDetector.primitives import Plane, PlaneBounded

def get_rotation_from_axis(axis_origin, axis):
    """ Rotation matrix that transforms `axis_origin` in `axis`.
    
    Parameters
    ----------
    axis_origin : 3 x 1 array
        Initial axis.
    axis : 3 x 1 array
        Goal axis.
    
    Returns
    -------
    rotation
        3x3 rotation matrix
    """
    axis = np.array(axis) / np.linalg.norm(axis)
    axis_origin = np.array(axis_origin) / np.linalg.norm(axis_origin)
    if abs(axis.dot(axis_origin) + 1) > 1E-6:
        # axis_origin = -axis_origin
        halfway_axis = (axis_origin + axis)[..., np.newaxis]
        halfway_axis /= np.linalg.norm(halfway_axis)
        return 2 * halfway_axis * halfway_axis.T - np.eye(3)
    else:
        orthogonal_axis = np.cross(np.random.random(3), axis)
        orthogonal_axis /= np.linalg.norm(orthogonal_axis)
        return Rotation.from_quat(list(orthogonal_axis)+[0]).as_matrix()

def group_similar_shapes(shapes, rtol=1e-02, atol=1e-02, bbox_intersection=None):
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
    
    def _check_bboxes(shape1, shape2):
        
        bb1 = [
            np.min(shape1.inlier_points, axis=0) - bbox_intersection,
            np.max(shape1.inlier_points, axis=0) + bbox_intersection]
        bb2 = [
            np.min(shape2.inlier_points, axis=0) - bbox_intersection,
            np.max(shape2.inlier_points, axis=0) + bbox_intersection]
        
        # bb1 = shape1.get_axis_aligned_bounding_box()
        # bb2 = shape2.get_axis_aligned_bounding_box()
        # test_order = (bb2.max_bound + atol) - (bb1.min_bound - atol) >= 0
        test_order = bb2[1] - bb1[0] >= 0
        if test_order.all():
            pass
        elif (~test_order).all():
            bb1, bb2 = bb2, bb1
        else:
            return False
        
        # test_intersect = (bb1.max_bound + atol) - (bb2.min_bound - atol) >= 0
        test_intersect = bb1[1] - bb2[0] >= 0
        return test_intersect.all()
    
    num_shapes = len(shapes)
    partitions = np.array(range(num_shapes))
    
    # import time
    # time1 = 0
    # time2 = 0
    
    for i, j in combinations(partitions, 2):
        if partitions[j] != j:
            continue
        
        # start = time.time()
        # test1 = _check_bboxes(shapes[i], shapes[j])
        # time1 += time.time() - start
    
        # start = time.time()
        # test2 = shapes[i].is_similar_to(shapes[j], rtol=rtol, atol=atol)
        # time2 += time.time() - start
        test1 = True if bbox_intersection is None else _check_bboxes(shapes[i], shapes[j])
        if test1 and shapes[i].is_similar_to(shapes[j], rtol=rtol, atol=atol):
        # if test1 and test2:
            partitions[j] = i
    
    sublists = []
    # partitions = np.array(partitions)
    for partition in set(partitions):
        idx = np.where(partitions == partition)
        sublist = [shapes[j] for j in np.where(partitions == partition)[0]]
        sublists.append(sublist)
    # print(f'time1 = {time1}, time2 = {time2}')
        
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
        
        primitive = type(sublist[0])
        if sublist[0].name == 'bounded plane':
            bounds = np.vstack([s.bounds for s in sublist])
            projection = np.vstack([s.projection for s in sublist])
            shape = primitive(model, bounds=None, rmse_max=None)
            shape.bounds = bounds
            shape.projection = projection
            # shape = shape.get_bounded_plane(bounds)
        else:
            shape = primitive(model)
        
        points = np.vstack([s.inlier_points for s in sublist])
        normals = np.vstack([s.inlier_normals for s in sublist])
        colors = np.vstack([s.inlier_colors for s in sublist])
        shape.add_inliers(points, normals, colors)
        
        if detector is not None:
            num_points = sum([s.metrics['num_points'] for s in sublist])
            num_inliers = len(points)
            distances, angles = shape.get_residuals(points, normals)
            shape.metrics = detector.get_metrics(
                num_points, num_inliers, distances, angles)
            
        new_list.append(shape)
        
    return new_list


def fuse_similar_shapes(shapes, detector=None, 
                        rtol=1e-02, atol=1e-02, bbox_intersection=None):
    """ Detect shapes with similar model and fuse them.
    
    If a detector is given, use it to compute the metrics of the resulting
    average shapes.
    
    See: group_shape_groups, fuse_shape_groups
    
    Parameters
    ----------
    shapes : list of shapes
        List containing all shapes.  
    detector : instance of some Detector, optional
        Used to recompute metrics.  
    rtol : float, optional
        The relative tolerance parameter. Default: 1e-02.
    atol : float, optional
        The absolute tolerance parameter. Default: 1e-02.
    
    Returns
    -------
    list
        Average shapes.
    """
    shapes_groupped = group_similar_shapes(shapes, rtol=rtol, atol=atol, 
                                           bbox_intersection=bbox_intersection)
    return fuse_shape_groups(shapes_groupped, detector)

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
                
