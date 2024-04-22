#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  11 13:52:35 2024

@author: ebernardes
"""

import pytest
import numpy as np
from scipy.spatial import Delaunay

from pyShapeDetector.primitives import Plane, PlaneBounded
from pyShapeDetector.utility import (
    get_triangle_boundary_indexes)

def test_get_triangle_boundary_indexes():
    plane = Plane.random()

    square = plane.get_square_plane(1)
    points_inner = plane.get_square_plane(0.8).sample_points_uniformly(10)

    vertices = np.vstack([square.bounds, points_inner])
    proj = plane.get_projections(vertices)
    triangles = Delaunay(proj).simplices

    boundary_indexes = get_triangle_boundary_indexes(vertices, triangles)

    assert len(boundary_indexes) == 4

    idx = list(set(np.array(boundary_indexes).flatten()))
    points_in_boundary = vertices[idx]

    for p in points_in_boundary:
        assert p in square.bounds





