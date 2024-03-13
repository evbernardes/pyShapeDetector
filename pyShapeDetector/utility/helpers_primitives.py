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
# from open3d.geometry import LineSet, TriangleMesh, PointCloud
# from pyShapeDetector.primitives import Plane, PlaneBounded, Line
# from pyShapeDetector.primitives import Primitive, Line

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

# def _group_similar_shapes_legacy(shapes, rtol=1e-02, atol=1e-02, 
#                           bbox_intersection=None, inlier_max_distance=None):
#     """ Legacy implementation, see group_similar_shapes """    
#     def _test(i, j):
#         if not shapes[i].is_similar_to(shapes[j], rtol=rtol, atol=atol):
#             return False
#         if not shapes[i].check_bbox_intersection(shapes[j], bbox_intersection):
#             return False
#         if not shapes[i].check_inlier_distance(shapes[j], inlier_max_distance):
#             return False
#         return True
    
#     num_shapes = len(shapes)
#     partitions = np.array(range(num_shapes))
    
#     for i, j in combinations(partitions, 2):
#         if partitions[j] != j:
#             continue
            
#         if _test(i, j):
#             partitions[j] = i
            
#         # if not shapes[i].is_similar_to(shapes[j], rtol=rtol, atol=atol):
#         #     continue
#         # if not shapes[i].check_bbox_intersection(shapes[j], bbox_intersection):
#         #     continue
#         # if not shapes[i].check_inlier_distance(shapes[j], inlier_max_distance):
#         #     continue

#         # partitions[j] = i
    
#     sublists = []
#     for partition in set(partitions):
#         sublist = [shapes[j] for j in np.where(partitions == partition)[0]]
#         sublists.append(sublist)
        
#     return sublists

def _get_partitions_legacy(num_shapes, pairs):
    new_indices = np.array(range(num_shapes))
    for pair, result in pairs.items():
        i, j = pair
        
        if new_indices[j] != j:
            continue
            
        if result:
            new_indices[j] = i
        
    partitions = []
    for index in set(new_indices):
        partition = set([i for i in np.where(new_indices == index)[0]])
        partitions.append(partition)            
    return partitions

def _get_partitions(num_shapes, pairs):
    # Step 2: graph-based partitions from pairs
    partitions = []
    added_indices = set()
    for pair, result in pairs.items():
        
        if pair[0] in added_indices and pair[1] in added_indices:
            if not result:
                continue
            
            i0 = np.where([pair[0] in p for p in partitions])[0][0]
            partition = partitions.pop(i0)
            
            # both added in same partitions
            if pair[1] in partition:
                partitions.append(partition)
         
            # fuse partitions
            else:
               i1 = np.where([pair[1] in p for p in partitions])[0][0]
               partitions[i1] |= partition
        
        # Check if pair passes the test
        elif result:
            if len(partitions) == 0:
                partitions.append(set(pair))
                added_indices.add(pair[0])
                added_indices.add(pair[1])
            else:
                for partition in partitions:
                    if pair[1] in partition:
                        partition.add(pair[0])
                        added_indices.add(pair[0])
                        break
                    if pair[0] in partition:
                        partition.add(pair[1])
                        added_indices.add(pair[1])
                        # added_to_partition = True
                        break

        else:
            # Add single-element partitions for elements that failed the test
            if pair[0] not in added_indices and pair[1] not in added_indices:
                partitions.append({pair[0]})
                partitions.append({pair[1]})
                added_indices.add(pair[0])
                added_indices.add(pair[1])
                
    for i in range(num_shapes):
        if i not in added_indices:
            partitions.append({i})
                
    if (num_test := sum(len(p) for p in partitions)) != num_shapes:
        print(f"This shouldn't have happened, implementatino error: {num_test} != {num_shapes}")
        assert False
        
    return partitions

def group_similar_shapes(shapes, rtol=1e-02, atol=1e-02, 
                          bbox_intersection=None, inlier_max_distance=None,
                          legacy=False):
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
    legacy : bool, optional
        Uses legacy implementation of `group_similar_shapes`. Default: False
    return_partitions : 
        
    Returns
    -------
    list of lists
        Grouped shapes
    list of sets
        Index partitions defining the shape groups.
    """
     
    def _test(i, j):
        if not shapes[i].is_similar_to(shapes[j], rtol=rtol, atol=atol):
            return False
        if not shapes[i].check_bbox_intersection(shapes[j], bbox_intersection):
            return False
        if not shapes[i].check_inlier_distance(shapes[j], inlier_max_distance):
            return False
        return True
    
    # Step 1: check all pairs
    shape_pairs = combinations(range(len(shapes)), 2)
    pairs = {pair: _test(*pair) for pair in shape_pairs}
    
    # Step 2: partitions from pairs
    if legacy:
        partitions = _get_partitions_legacy(len(shapes), pairs)
    else:
        # graph-based 
        partitions = _get_partitions(len(shapes), pairs)
                
    # Step 3: get sublists of shapes from partitions
    shape_groups = [[shapes[i] for i in partition] for partition in partitions]
    
    # shape_groups = []
    # for partition in partitions:
    #     group = [shapes[i] for i in partition]
    #     shape_groups.append(group)
        
    return shape_groups, partitions

def fuse_shape_groups(shapes_lists, **fuse_options):
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
    ignore_extra_data : boolean, optional
        If True, ignore everything and only fuse model. Default: False.
    line_intersection_eps : float, optional
        Distance for detection of intersection between planes. Default: 0.001.
        
    Extra parameters for PlaneBounded
    ---------------------------------
    force_concave : boolean, optional.
        If True, the fused plane will be concave regardless of inputs.
        Default: True.
    ressample_density : float, optional
        Default: 1.5
    ressample_radius_ratio : float, optional
        Default: 1.2
        
    Returns
    -------
    list
        Average shapes.    
    """
    fused_shapes = []
    for sublist in shapes_lists:
        primitive = type(sublist[0])
        fused_shape = primitive.fuse(
            sublist, **fuse_options)
        
        num_points = sum(len(s.inlier_points) for s in sublist)
        if num_points != len(fused_shape.inlier_points):
            pass
        
        fused_shapes.append(fused_shape)
        
    return fused_shapes

def fuse_similar_shapes(shapes, rtol=1e-02, atol=1e-02, 
                        bbox_intersection=None, inlier_max_distance=None,
                        legacy=False, **fuse_options):
    """ Detect shapes with similar model and fuse them.
    
    If a detector is given, use it to compute the metrics of the resulting
    average shapes.
    
    For extra fuse options, see: fuse_shape_groups
    
    See: group_shape_groups
    
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
    list
        Average shapes.
    """
    shapes_groupped, _ = group_similar_shapes(
        shapes, rtol, atol, bbox_intersection, inlier_max_distance, legacy)
    
    return fuse_shape_groups(shapes_groupped, **fuse_options)

def find_plane_intersections(
        shapes, bbox_intersection=None, inlier_max_distance=None,
        length_max=None, distance_max=None, ignore=None, 
        intersect_parallel=False, eps_angle=np.deg2rad(5.0), 
        eps_distance=1e-2):
    """ For every possible pair of neighboring bounded planes, calculate their
    intersection and return dictionary of all intersection lines.
    
    See: group_shape_groups, fuse_shape_groups, glue_nearby_planes
    
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
        whether planes are parallel. Default: 0.08726646259971647 (5 degrees).
    eps_distance : float, optional
        When planes are parallel, eps_distance is used to detect if the 
        planes are close enough to each other in the dimension aligned
        with their axes. Default: 1e-2.
    
    Returns
    -------
    dict
        Dictionary of intersections.
    """
    # if bbox_intersection is None and inlier_max_distance is None:
    #     raise ValueError("bbox_intersection and inlier_max_distance cannot "
    #                      "both be equal to None.")
    
    from pyShapeDetector.primitives import Plane, PlaneBounded, Line
    
    if length_max is not None and length_max <= 0:
        raise ValueError(f"'length_max' must be a positive float, got {length_max}." )    
    
    if ignore is None:
        ignore = [False] * len(shapes)
        
    if len(ignore) != len(shapes):
        raise ValueError("'ignore' must be a list of booleans the size of 'shapes'.")
    
    lines = []
    intersections = {}
    num_shapes = len(shapes)

    for i, j in combinations(range(num_shapes), 2):
        
        # if shapes[i].name != 'bounded plane' or shapes[j].name != 'bounded plane':
        #     continue
    
        if not isinstance(shapes[i], Plane) or not isinstance(shapes[j], Plane):
            continue
        
        if not shapes[i].is_convex or not shapes[j].is_convex:
            continue
        
        if ignore[i] or ignore[j]:
            continue
        
        both_bounded = isinstance(shapes[i], PlaneBounded) and isinstance(shapes[j], PlaneBounded)
        
        if both_bounded and not shapes[i].check_bbox_intersection(shapes[j], bbox_intersection):
            continue
        
        if both_bounded and not shapes[i].check_inlier_distance(shapes[j], inlier_max_distance):
            continue
        
        line = Line.from_plane_intersection(
            shapes[i], shapes[j], intersect_parallel=intersect_parallel,
            eps_angle=eps_angle, eps_distance=eps_distance)
        
        if line is None:
            continue
        
        if length_max is not None and line.length > length_max:
            continue
        
        elif distance_max is not None:
            # TODO: bounds_or_vertices should be changed for bounds if we are 
            # sure that it will never be implemented for non convex planes
            points = shapes[i].bounds_or_vertices

            if min(line.get_distances(points)) > distance_max:
                continue
                # continue
            if min(line.get_distances(points)) > distance_max:
                continue
                # continue
        
        lines.append(line)            
        intersections[i, j] = line
        
    return intersections

def glue_planes_with_intersections(shapes, intersections):
    """ Glue shapes using intersections in a dict.
    
    Also returns dictionary of all intersection lines.
    
    See: group_shape_groups, fuse_shape_groups, find_plane_intersections, glue_nearby_planes
    
    Parameters
    ----------
    intersections : dict
        Dictionary with keys of type `(i, j)` and values of type Primitive.Line.
    
    Returns
    -------
    dict
        Dictionary of intersections.
    """
    from pyShapeDetector.primitives import PlaneBounded
    
    new_intersections = []
        
    for (i, j), line in intersections.items():
        
        if not isinstance(shapes[i], PlaneBounded) or not isinstance(shapes[j], PlaneBounded):
            # del intersections[i, j]
            continue
        
        new_intersections[i, j] = line
        
        for shape in [shapes[i], shapes[j]]:
            line_ = line.get_line_fitted_to_projections(shape.bounds)
            # TODO: add vertices too?
            shape.add_bound_points([line_.beginning, line_.ending])
            shape.add_inliers([line_.beginning, line_.ending])
            # new_points = [line.beginning, line.ending]
            
    return new_intersections

def glue_nearby_planes(shapes, **options):
    """ For every possible pair of neighboring bounded planes, calculate their
    intersection and then glue them to this intersection.
    
    Also returns dictionary of all intersection lines.
    
    See: group_shape_groups, fuse_shape_groups, find_plane_intersections
    
    Parameters
    ----------
    To see every possible parameter, see: find_plane_intersections
    
    Returns
    -------
    dict
        Dictionary of intersections.
    """
    intersections = find_plane_intersections(shapes, **options)
    return glue_planes_with_intersections(shapes, intersections)

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
        else:
            mesh.paint_uniform_color(shape.color)

        meshes.append(mesh)

    return meshes
                
