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
# from pyShapeDetector.primitives import Line

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

def group_similar_shapes(shapes, rtol=1e-02, atol=1e-02, 
                         bbox_intersection=None, inlier_max_distance=None):
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
    bbox_intersection : float, optional
        Max distance between inlier bounding boxes. If None, ignore this test.
        Default: None.
    inlier_max_distance : float, optional
        Max distance between points in shapes. If None, ignore this test.
        Default: None.
        
    Returns
    -------
    list of lists
        Grouped shapes.
    
    """
    
    num_shapes = len(shapes)
    partitions = np.array(range(num_shapes))
    
    # import time
    # time1 = 0
    # time2 = 0
    
    for i, j in combinations(partitions, 2):
        if partitions[j] != j:
            continue
            
        test = shapes[i].is_similar_to(shapes[j], rtol=rtol, atol=atol)
        test = test and shapes[i].check_bbox_intersection(shapes[j], bbox_intersection)
        test = test and shapes[i].check_inlier_distance(shapes[j], inlier_max_distance)
            
        if test:
        # if test1 and test2:
            partitions[j] = i
    
    sublists = []
    # partitions = np.array(partitions)
    for partition in set(partitions):
        # idx = np.where(partitions == partition)
        sublist = [shapes[j] for j in np.where(partitions == partition)[0]]
        sublists.append(sublist)
    # print(f'time1 = {time1}, time2 = {time2}')
        
    return sublists

def fuse_shape_groups(shapes_lists, detector=None, line_intersection_eps=1e-3):
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
    # from pyShapeDetector.primitives import PlaneBounded
        
    new_list = []
    for sublist in shapes_lists:
        try:
            fitness = [s.metrics['fitness'] for s in sublist]
        except:
            fitness = [1] * len(sublist)
        
        model = np.vstack([s.model for s in sublist])
        model = np.average(model, axis=0, weights=fitness)
        
        primitive = type(sublist[0])
        if sublist[0].name == 'bounded plane':
        # if isinstance(primitive, PlaneBounded):
            bounds = np.vstack([s.bounds for s in sublist])
            shape = primitive(model, bounds=bounds, rmse_max=None)
            
            intersections = []
            for plane1, plane2 in combinations(sublist, 2):
                points = plane1.intersection_bounds(plane2, True, eps=line_intersection_eps)
                if len(points) > 0:
                    intersections.append(points)
            
            # temporary hack, saving intersections for mesh generation
            if len(intersections) > 0:
                shape._fusion_intersections = np.vstack(intersections)
            
        else:
            shape = primitive(model)
        
        points = np.vstack([s.inlier_points for s in sublist])
        normals = np.vstack([s.inlier_normals for s in sublist])
        colors = np.vstack([s.inlier_colors for s in sublist])
        if points.shape[1] == 0:
            points = None
        if normals.shape[1] == 0 or len(normals) < len(points):
            normals = None
        if colors.shape[1] == 0 or len(colors) < len(points):
            colors = None
        shape.add_inliers(points, normals, colors)
        
        if detector is not None:
            num_points = sum([s.metrics['num_points'] for s in sublist])
            num_inliers = len(points)
            distances, angles = shape.get_residuals(points, normals)
            shape.metrics = detector.get_metrics(
                num_points, num_inliers, distances, angles)
            
        new_list.append(shape)
        
    return new_list


def fuse_similar_shapes(shapes, detector=None,  rtol=1e-02, atol=1e-02, 
                        bbox_intersection=None, inlier_max_distance=None,
                        line_intersection_eps=1e-3):
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
    bbox_intersection : float, optional
        Max distance between inlier bounding boxes. If None, ignore this test.
        Default: None.
    inlier_max_distance : float, optional
        Max distance between points in shapes. If None, ignore this test.
        Default: None.
    
    Returns
    -------
    list
        Average shapes.
    """
    shapes_groupped = group_similar_shapes(shapes, rtol=rtol, atol=atol, 
                                           bbox_intersection=bbox_intersection,
                                           inlier_max_distance=inlier_max_distance)
    return fuse_shape_groups(shapes_groupped, detector, line_intersection_eps=line_intersection_eps)

def glue_nearby_planes(shapes, bbox_intersection=None, inlier_max_distance=None,
                       length_max=None, distance_max=None, ignore=None, intersect_parallel=False,
                       eps_angle=np.deg2rad(0.9), eps_distance=1e-2):
    """ For every possible pair of neighboring bounded planes, calculate their
    intersection and then glue them to this intersection.
    
    Also returns list of all intersection lines.
    
    See: group_shape_groups, fuse_shape_groups
    
    Parameters
    ----------
    shapes : list of shapes
        List containing all shapes.  
    bbox_intersection : float, optional
        Max distance between planes.
    inlier_max_distance : float, optional
        Max distance between points in shapes. If None, ignore this test.
        Default: None.
    length_max : float, optional
        If a value is given, limits lenghts of intersection lines. 
        Default: None.
    distance_max : float, optional
        If a value is given, limits the distance of intersections. 
        Default: None.
    ignore : list of booleans, optional
        If a list of booleans is given, ignore every ith plane in shapes if
        the ith value of 'ignore' is True.
    intersect_parallel : boolean, optional
        If True, try to intersect parallel planes too. Default: False.
    eps_angle : float, optional
        Minimal angle (in radians) between normals necessary for detecting
        whether planes are parallel. Default: 0.0017453292519943296
    eps_distance : float, optional
        When planes are parallel, eps_distance is used to detect if the 
        planes are close enough to each other in the dimension aligned
        with their axes. Default: 1e-2.
    
    Returns
    -------
    list
        List of intersections.
    """
    
    if bbox_intersection is None and inlier_max_distance is None:
        raise ValueError("bbox_intersection and inlier_max_distance cannot "
                         "both be equal to None.")
    
    from pyShapeDetector.primitives import Line
    
    if length_max is not None and length_max <= 0:
        raise ValueError(f"'length_max' must be a positive float, got {length_max}." )    
    
    if ignore is None:
        ignore = [False] * len(shapes)
    if len(ignore) != len(shapes):
        raise ValueError("'ignore' must be a list of booleans the size of 'shapes'.")
    
    lines = []
    a = {}

    for i, j in combinations(range(len(shapes)), 2):
        
        if shapes[i].name != 'bounded plane' or shapes[j].name != 'bounded plane':
            continue
        
        if ignore[i] or ignore[j]:
            continue
        
        if not shapes[i].check_bbox_intersection(shapes[j], bbox_intersection):
            continue
        
        if not shapes[i].check_inlier_distance(shapes[j], inlier_max_distance):
            continue
        
        line = Line.from_plane_intersection(shapes[i], shapes[j], intersect_parallel=intersect_parallel,
                                            eps_angle=eps_angle, eps_distance=eps_distance)
        if line is None:
            line = None
            # continue
        
        if length_max is not None and line.length > length_max:
            line = None
            # continue
        
        if distance_max is not None:
            if min(line.get_distances(shapes[i].bounds)) > distance_max:
                line = None
                # continue
            if min(line.get_distances(shapes[j].bounds)) > distance_max:
                line = None
                # continue
        
        if line is not None:
            lines.append(line)
        a[i, j] = line
        
    for i, j in combinations(range(len(shapes)), 2):
        line = a[i, j]
        if line is None:
            continue

        for shape in [shapes[i], shapes[j]]:
            line_ = line.get_line_fitted_to_projections(shape.bounds)
            shape.add_bound_points([line_.beginning, line_.ending])
            new_points = [line.beginning, line.ending]
            # # shapes[i]._set_bounds(np.vstack([shapes[i].bounds] + new_points))
            # # shapes[j]._set_bounds(np.vstack([shapes[j].bounds] + new_points))
            # shapes[i].add_bound_points(new_points)
            # shapes[j].add_bound_points(new_points)
            
    return lines

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
                
def get_meshes(shapes, crop_types=['sphere', 'cone'], paint_random=False):
    """ Returns meshes from shapes.
    
    Parameters
    ----------
    crop_types : list of strings, optional
        Define type of primitives that should be cropped according to their
        inlier points. Default: ['sphere', 'cone']
    paint_random: boolean, optional
        When positive, paint each mesh with a different random color.
    
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
        if paint_random:
            mesh.paint_uniform_color(np.random.random(3))
        # mesh.paint_uniform_color(np.random.random(3))
        meshes.append(mesh)

    return meshes
                
