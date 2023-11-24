#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:51:40 2023

@author: ebernardes
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector

from pyShapeDetector.primitives import Plane, Sphere, Cylinder, Cone
from pyShapeDetector.primitives import PlaneBounded
primitives_simple = [Plane, Sphere, Cylinder, Cone]

def rmse(x):
    """ Helper for root mean square error. """
    return np.sqrt(sum(x * x)) / len(x)

def get_shape_and_pcd(primitive, num_points, canonical=False):
    
    if primitive._name == 'bounded plane':
        shape = Plane(np.random.rand(4)).get_square_plane(np.random.rand())
    else:
        model = np.random.rand(primitive._model_args_n)
        shape = primitive(model)
    mesh = shape.get_mesh()
    pcd = mesh.sample_points_uniformly(num_points)
    pcd.estimate_normals()
    if canonical:
        shape = shape.canonical
    return shape, pcd


def test_primitive_init():
    for primitive in primitives_simple:
        model = np.random.rand(primitive._model_args_n)
        if primitive.name == 'bounded plane':
            with pytest.warns(UserWarning, match='returning square plane'):
                primitive(model)
        else:
            primitive(model)
            
        with pytest.raises(ValueError, match='elements, got'): 
            model = np.random.rand(primitive._model_args_n+1)
            primitive(model)

        with pytest.raises(ValueError, match='elements, got'): 
            model = np.random.rand(primitive._model_args_n-1)
            primitive(model)
            
    
def test_plane_surface_area_and_volume():
    model = np.random.rand(4)
    plane = Plane(model)
    assert plane.volume == 0
    with pytest.warns(UserWarning, match='surface area is undefined'):
        assert np.isnan(plane.surface_area)
    
    length = np.random.rand()
    plane_bounded = plane.get_square_plane(length)
    assert plane_bounded.volume == 0
    assert_allclose(plane_bounded.surface_area, length ** 2)


def test_fit():
    for primitive in primitives_simple:
        # if primitive == Cone:
            # pass
        for i in range(5):
            shape, pcd = get_shape_and_pcd(primitive, 10000, canonical=True)
            assert shape is not None
            shape_fit = primitive.fit(pcd.points, normals=pcd.normals).canonical
            assert_allclose(shape.model, shape_fit.model, rtol=1e-2, atol=1e-2)


def test_distances():
    for primitive in primitives_simple:
        for i in range(5):
            shape, pcd = get_shape_and_pcd(primitive, 100, canonical=False)
            assert shape is not None
            distances = shape.get_distances(pcd.points)
            assert_allclose(rmse(distances), 0, atol=1e-3)


def test_distances_flatten():
    for primitive in primitives_simple:
        for i in range(5):
            shape, _ = get_shape_and_pcd(primitive, 100, canonical=False)
            points = np.random.rand(100, 3)
            points_flattened = shape.flatten_points(points)
            distances = shape.get_distances(points_flattened)
            assert_allclose(distances, 0, atol=1e-10)


def test_normals_flatten_non_problematic():
    for primitive in [Plane]:
        for i in range(5):
            shape, _ = get_shape_and_pcd(primitive, 100, canonical=False)
            points = np.random.rand(1000, 3)
            points_flattened = shape.flatten_points(points)
            # distances = shape.get_distances(points_flattened)
            pcd_flattened = PointCloud(Vector3dVector(points_flattened))
            pcd_flattened.estimate_normals()
            normals = pcd_flattened.normals
            angles = shape.get_angles(points_flattened, normals)
            assert_allclose(angles, 0, atol=1e-5)      
    

def test_normals_flatten_problematic():
    # TODO: Estimation of normals in Cylinder and Sphere give big angles 
    for primitive in [Sphere, Cylinder]:
        for i in range(5):
            shape, _ = get_shape_and_pcd(primitive, 100, canonical=False)
            points = np.random.rand(1000, 3)
            points_flattened = shape.flatten_points(points)
            # distances = shape.get_distances(points_flattened)
            pcd_flattened = PointCloud(Vector3dVector(points_flattened))
            pcd_flattened.estimate_normals()
            normals = pcd_flattened.normals
            angles = shape.get_angles(points_flattened, normals)
            assert_allclose(rmse(angles), 0, atol=1e-1)
