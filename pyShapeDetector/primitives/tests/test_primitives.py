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

def get_shape_and_pcd(primitive, num_points):
    model = np.random.rand(primitive._model_args_n)
    shape = primitive(model)
    mesh = shape.get_mesh()
    pcd = mesh.sample_points_uniformly(num_points)
    pcd.estimate_normals()
    
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


def test_fit_plane():
    for i in range(2):
        shape, pcd = get_shape_and_pcd(Plane, 10000)
        shape_fit = Plane.fit(pcd.points, normals=pcd.normals)
        assert_allclose(shape.normal, shape_fit.normal, atol=1e-2)
    
    
def test_fit_sphere():
    for i in range(2):
        shape, pcd = get_shape_and_pcd(Sphere, 10000)
        shape_fit = Sphere.fit(pcd.points, normals=pcd.normals)
        assert_allclose(shape.center, shape_fit.center, atol=1e-2)
        assert_allclose(shape.radius, shape_fit.radius, atol=1e-2)
    

def test_fit_cylinder():
    for i in range(2):
        shape, pcd = get_shape_and_pcd(Cylinder, 10000)
        shape_fit = Cylinder.fit(pcd.points, normals=pcd.normals)
        assert_allclose(shape.center, shape_fit.center, atol=1e-2)
        assert_allclose(shape.radius, shape_fit.radius, atol=1e-2)
        assert_allclose(shape.axis * np.sign(shape.axis[-1]), 
                        shape_fit.axis * np.sign(shape_fit.axis[-1]), 
                        atol=1e-2)
        assert_allclose(shape.height, shape_fit.height, atol=1e-3)

def test_distances():
    for primitive in [Sphere]:
        shape, pcd = get_shape_and_pcd(primitive, 100)
        distances = shape.get_distances(pcd.points)
        # relative tolerance does not make sense here
        assert_allclose(distances, np.zeros(len(distances)), atol=1e-2)
        
        
        
        