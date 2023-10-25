#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:51:40 2023

@author: ebernardes
"""

import pytest

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from pyShapeDetector.primitives import Plane, Sphere, Cylinder
primitives = [Plane, Sphere, Cylinder]

def get_shape_and_pcd(primitive, num_points, canonical=False):
    model = np.random.rand(primitive._model_args_n)
    shape = primitive(model)
    mesh = shape.get_mesh()
    pcd = mesh.sample_points_uniformly(num_points)
    pcd.estimate_normals()
    if canonical:
        shape = shape.canonical
    
    return shape, pcd


def test_primitive_init():
    for primitive in primitives:
        model = np.random.rand(primitive._model_args_n)
        shape = primitive(model)

        with pytest.raises(ValueError, match='elements, got'): 
            model = np.random.rand(primitive._model_args_n+1)
            shape = primitive(model)

        with pytest.raises(ValueError, match='elements, got'): 
            model = np.random.rand(primitive._model_args_n-1)
            shape = primitive(model)


def test_fit():
    for primitive in primitives:
        for i in range(10):
            shape, pcd = get_shape_and_pcd(primitive, 10000, canonical=True)
            shape_fit = primitive.fit(pcd.points, normals=pcd.normals).canonical
            assert_allclose(shape.model, shape_fit.model, rtol=1e-2, atol=1e-2)


def test_distances():
    for primitive in primitives:
        for i in range(10):
            shape, pcd = get_shape_and_pcd(primitive, 100, canonical=False)
            distances = shape.get_distances(pcd.points)
            # rmse = np.sqrt(sum(distances * distances)) / len(pcd.points)
            assert_allclose(distances, 0, atol=1e-3)

        

        
        
        