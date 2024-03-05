#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:51:40 2023

@author: ebernardes
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import warnings

from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector

from pyShapeDetector.primitives import Plane, Sphere, Cylinder, Cone
from pyShapeDetector.primitives import PlaneBounded
# primitives_simple = [Plane, Sphere, Cylinder, Cone]
# primitives_simple = [Plane, Sphere, Cylinder]

def rmse(x):
    """ Helper for root mean square error. """
    return np.sqrt(sum(x * x)) / len(x)

def get_shape_and_pcd(primitive, num_points, canonical=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def test_plane_projections():
    plane = Plane.random()
    points = plane.flatten_points(np.random.random((100, 3)))
    projections = plane.get_projections(points)
    points_recovered = plane.get_points_from_projections(projections)
    assert_allclose(points_recovered, points)


def test_primitive_init():
    for primitive in [Plane, Sphere, Cylinder]:
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
    
    vertices = np.array(plane_bounded.mesh.vertices)
    triangles = np.array(plane_bounded.mesh.triangles)

    # get only half of the doubled triangles
    # triangles = triangles[:int(len(triangles) / 2)]

    plane_bounded.set_vertices_triangles(vertices, triangles)
    assert_allclose(plane_bounded.surface_area, length ** 2)


def test_fit():
    # testing Cylinder separately
    for i in range(5):
        shape, pcd = get_shape_and_pcd(Cylinder, 1000, canonical=True)
        shape_fit = Cylinder.fit(pcd.points, normals=pcd.normals).canonical
        # assert_allclose(shape.center, shape_fit.center, rtol=1e-2, atol=1e-2)
        assert_allclose(shape.vector, shape_fit.vector, rtol=1e-2, atol=1e-2)
        assert_allclose(shape.radius, shape_fit.radius, rtol=1e-2, atol=1e-2)

    # # testing Cone separately
    # for i in range(5):
    #     shape, pcd = get_shape_and_pcd(Cone, 10000, canonical=True)
    #     shape_fit = Cone.fit(pcd.points, normals=pcd.normals).canonical
    #     assert_allclose(shape.appex, shape_fit.appex, rtol=1e-1, atol=1e-1)
    #     assert_allclose(shape.half_angle, shape_fit.half_angle, rtol=1e-1, atol=1e-1)
    #     # assert_allclose(shape.vector, shape_fit.vector, rtol=1e-2, atol=1e-2)
    #     # assert_allclose(shape.radius, shape_fit.radius, rtol=1e-2, atol=1e-2)

    for primitive in [Plane, Sphere]:
    # for primitive in [Sphere]:
        for i in range(5):
            shape, pcd = get_shape_and_pcd(primitive, 100, canonical=True)
            shape_fit = primitive.fit(pcd.points, normals=pcd.normals).canonical
            assert_allclose(shape.model, shape_fit.model, rtol=1e-2, atol=1e-2)


def test_distances():
    for primitive in [Plane, Sphere, Cylinder, Cone]:
        for i in range(5):
            shape, pcd = get_shape_and_pcd(primitive, 100, canonical=False)
            distances = shape.get_distances(pcd.points)
            # assert_allclose(rmse(distances), 0, atol=1e-2)
            assert_allclose(distances, 0, atol=1e-1)


def test_distances_flatten():
    # for primitive in [Plane, Sphere, Cylinder, Cone]:
    for primitive in [Plane, Sphere, Cylinder]:
        for i in range(5):
            shape, _ = get_shape_and_pcd(primitive, 100, canonical=False)
            points = np.random.rand(100, 3)
            points_flattened = shape.flatten_points(points)
            distances = shape.get_distances(points_flattened)
            assert_allclose(distances, 0, atol=1e-6)

            points_reflattened = shape.flatten_points(points_flattened)
            assert_allclose(points_reflattened, points_flattened, atol=1e-6)


def test_normals_flatten_plane():
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
    

def test_normals_flatten_others():
    # TODO: Estimation of normals in Cylinder and Sphere give big angles 
    for primitive in [Sphere, Cylinder, Cone]:
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
