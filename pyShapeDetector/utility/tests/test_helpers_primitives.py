#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  11 13:52:35 2024

@author: ebernardes
"""

import pytest
import numpy as np

from pyShapeDetector.primitives import Plane, PlaneBounded
from pyShapeDetector.utility import (
    group_similar_shapes, find_plane_intersections)

def test_planes_group_bbox_intersection():
    # num_samples = 50
    centroid = np.array([0, 0, 0])
    normal = np.array([0, 0, 1])
    direction = np.array([1, 0, 0])
    side = 2
    dist = 0.1
    eps = 1e-3
    p = Plane.from_normal_point(normal, centroid).get_square_plane(side)
    pleft = p.copy()
    pleft.translate(- direction * (side + dist))
    pright = p.copy()
    pright.translate(+ direction * (side + dist))

    # slightly below limit
    assert 1 == len(group_similar_shapes([pleft, p], bbox_intersection=dist+eps))
    assert 1 == len(group_similar_shapes([pright, p], bbox_intersection=dist+eps))
    
    # slightly above limit
    assert 2 == len(group_similar_shapes([pleft, p], bbox_intersection=dist-eps))
    assert 2 == len(group_similar_shapes([pleft, p], bbox_intersection=dist-eps))
    
    # all planes
    assert 1 == len(group_similar_shapes([pleft, p, pright], bbox_intersection=dist+eps))
    assert 3 == len(group_similar_shapes([pleft, p, pright], bbox_intersection=dist-eps))
    
def test_planes_group_inlier_max_distance():
    num_samples = 1000
    centroid = np.array([0, 0, 0])
    normal = np.array([0, 0, 1])
    direction = np.array([1, 0, 0])
    side = 2
    dist = 0.1
    eps = 5e-2
    p = Plane.from_normal_point(normal, centroid).get_square_plane(side)
    pleft = p.copy()
    pleft.translate(- direction * (side + dist))
    pright = p.copy()
    pright.translate(+ direction * (side + dist))
    
    with pytest.raises(RuntimeError, match='must have inlier points'): 
        group_similar_shapes([pleft, p], inlier_max_distance=eps)
        
    # adding points...
    p.set_inliers(p.sample_points_uniformly(num_samples))
    pleft.set_inliers(pleft.sample_points_uniformly(num_samples))
    pright.set_inliers(pright.sample_points_uniformly(num_samples))
    
    # ... and now it should work:
    # slightly below limit
    assert 1 == len(group_similar_shapes([pleft, p], inlier_max_distance=dist+eps))
    assert 1 == len(group_similar_shapes([pright, p], inlier_max_distance=dist+eps))
    
    # slightly above limit
    assert 2 == len(group_similar_shapes([pleft, p], inlier_max_distance=dist-eps))
    assert 2 == len(group_similar_shapes([pleft, p], inlier_max_distance=dist-eps))
    
    # all planes
    assert 1 == len(group_similar_shapes([pleft, p, pright], inlier_max_distance=dist+eps))
    assert 3 == len(group_similar_shapes([pleft, p, pright], inlier_max_distance=dist-eps))
    
    
def test_box_intersections():
    length = 5
    
    planes = PlaneBounded.create_box((0, 0, 0), length)
    intersections = find_plane_intersections(planes, 1e-3)
    assert len(intersections) == 12
    
    planes[0].translate(planes[0].normal * 10)
    intersections = find_plane_intersections(planes, 1e-3)
    assert len(intersections) == 12 - 4
    intersections = find_plane_intersections(planes, 10 + 1e-3)
    assert len(intersections) == 12
    
    
# if __name__ == "__main__":
    # test_box_intersections()
#     test_group_inlier_max_distance()