#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:51:40 2023

@author: ebernardes
"""

import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose
import warnings
from itertools import combinations

from scipy.spatial.transform import Rotation

from open3d.geometry import PointCloud, AxisAlignedBoundingBox
from open3d.utility import Vector3dVector

from pyShapeDetector.primitives import (
    Plane, PlaneBounded, Sphere, Cylinder, Cone, Line)

all_primitives_regular = [Plane, Sphere, Cylinder, Cone]
all_primitives = all_primitives_regular + [PlaneBounded, Line]
all_primitives_regular_bounded = [PlaneBounded, Sphere, Cylinder, Cone]
all_primitives_bounded = [PlaneBounded, Sphere, Cylinder, Cone] + [Line]

def rmse(x):
    """ Helper for root mean square error. """
    return np.sqrt(sum(x * x)) / len(x)

def get_shape(primitive, num_points, canonical=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        shape = primitive.random()
        pcd = shape.sample_PointCloud_uniformly(num_points)
        shape.set_inliers(pcd)

        if canonical:
            shape = shape.canonical
            
        return shape
    

def test_init_primitives_regular():
    for primitive in [Plane, Sphere, Cylinder]:
        
        model = np.random.rand(primitive._model_args_n)
        primitive(model)
            
        with pytest.raises(ValueError, match='elements, got'): 
            model = np.random.rand(primitive._model_args_n+1)
            primitive(model)

        with pytest.raises(ValueError, match='elements, got'): 
            model = np.random.rand(primitive._model_args_n-1)
            primitive(model)


def test_init_plane_bounded():
    model = np.random.rand(PlaneBounded._model_args_n)
    bounds = np.random.random((50, 3))

    with pytest.warns(UserWarning, match='returning square plane'):
        PlaneBounded(model)

    PlaneBounded(model, bounds)


def test_canonical():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for primitive in all_primitives:
            shape = get_shape(PlaneBounded, 10)
            shape2 = shape.canonical
            assert shape == shape2


def test_plane_projections():
    plane = Plane.random()
    points = plane.flatten_points(np.random.random((100, 3)))
    projections = plane.get_projections(points)
    points_recovered = plane.get_points_from_projections(projections)
    assert_allclose(points_recovered, points)


def test_equal():
    primitives = all_primitives
    primitives.remove(Plane)
    primitives.remove(PlaneBounded)
    for _ in range(10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for primitive in primitives:
                shape1 = primitive.random(decimals=3)
                shape2 = primitive.random(decimals=3)
                assert np.all(shape1.model == shape1.copy().model)
                assert shape1 == shape1.copy()
                assert not np.all(shape1.model == shape2.model)
                assert shape1 != shape2


def test_issue_3_plane_init():
    for _ in range(100):
        plane = Plane.random(decimals=4)
        assert plane == plane.copy()


def test_copy_regular():
    for primitive in all_primitives_regular:
        shape = get_shape(primitive, 100)

        shape_copy = shape.copy()
        assert shape == shape_copy
        assert id(shape) != id(shape_copy)
        assert id(shape.inlier_points) != id(shape_copy.inlier_points)
        shape_copy = copy.copy(shape)
        assert shape == shape_copy
        assert id(shape) != id(shape_copy)
        assert id(shape.inlier_points) != id(shape_copy.inlier_points)


def test_copy_planebounded():

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shape = get_shape(PlaneBounded, 100)
        hole = PlaneBounded.create_circle(
            center=shape.centroid, normal=shape.normal, radius=0.2
        )
        shape.add_holes(hole)

        for shape_copy in [shape.copy(), copy.copy(shape)]:
            assert shape == shape_copy
            assert id(shape) != id(shape_copy)
            assert id(shape.inlier_points) != id(shape_copy.inlier_points)
            assert shape.holes[0] == shape_copy.holes[0]
            assert id(shape.holes[0]) != id(shape_copy.holes[0])
            assert np.all(shape.holes[0].bounds == shape_copy.holes[0].bounds)
            assert id(shape.holes[0].bounds) != id(shape_copy.holes[0].bounds)


def test_deepcopy():
    for primitive1, primitive2 in combinations(all_primitives, 2):
        shapes = [get_shape(primitive1, 20), get_shape(primitive2, 20)]
        shapes_copy = copy.deepcopy(shapes)

        assert shapes[0] == shapes_copy[0]
        assert shapes[1] == shapes_copy[1]
        assert_allclose(shapes[0].inlier_points, shapes_copy[0].inlier_points)
        assert_allclose(shapes[0].inlier_points, shapes_copy[0].inlier_points)
        assert id(shapes[0]) != id(shapes_copy[1])
        assert id(shapes[1]) != id(shapes_copy[1])
        assert id(shapes[0].inlier_points) != id(shapes_copy[0].inlier_points)
        assert id(shapes[0].inlier_points) != id(shapes_copy[0].inlier_points)


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
        shape = get_shape(Cylinder, 1000, canonical=True)
        shape_fit = Cylinder.fit(
            shape.inlier_points, normals=shape.inlier_normals).canonical
        # test axis instead of vector for direct fit
        assert_allclose(shape.axis, shape_fit.axis, rtol=1e-2, atol=1e-2)
        assert_allclose(shape.radius, shape_fit.radius, rtol=1e-2, atol=1e-2)

    # # testing Cone separately
    # for i in range(5):
    #     shape = get_shape(Cone, 10000, canonical=True)
        # pcd = shape.inlier_PointCloud
    #     shape_fit = Cone.fit(pcd.points, normals=pcd.normals).canonical
    #     assert_allclose(shape.appex, shape_fit.appex, rtol=1e-1, atol=1e-1)
    #     assert_allclose(shape.half_angle, shape_fit.half_angle, rtol=1e-1, atol=1e-1)
    #     # assert_allclose(shape.vector, shape_fit.vector, rtol=1e-2, atol=1e-2)
    #     # assert_allclose(shape.radius, shape_fit.radius, rtol=1e-2, atol=1e-2)

    for primitive in [Plane, Sphere]:
    # for primitive in [Sphere]:
        for i in range(5):
            shape = get_shape(primitive, 100, canonical=True)
            shape_fit = primitive.fit(
                shape.inlier_points, normals=shape.inlier_normals).canonical
            assert_allclose(shape.model, shape_fit.model, rtol=1e-2, atol=1e-2)


def test_distances():
    for primitive in all_primitives_regular:
        for i in range(5):
            shape = get_shape(primitive, 100, canonical=False)
            distances = shape.get_distances(shape.inlier_points)
            # assert_allclose(rmse(distances), 0, atol=1e-2)
            assert_allclose(distances, 0, atol=1e-1)


def test_distances_flatten():
    for primitive in all_primitives_regular:
        for i in range(5):
            shape = get_shape(primitive, 100, canonical=False)
            points = np.random.rand(100, 3)
            points_flattened = shape.flatten_points(points)
            distances = shape.get_distances(points_flattened)
            assert_allclose(distances, 0, atol=1e-6)

            points_reflattened = shape.flatten_points(points_flattened)
            assert_allclose(points_reflattened, points_flattened, atol=1e-6)


def test_issue_2_cone_distances():

    appex = np.array([0.18595758, 0.80201102, 0.85902078])
    top = np.array([0.56415032, 1.40444288, 1.14442953])
    radius = 0.769831334575714
    shape = Cone.from_appex_top_radius(appex, top, radius)

    point = shape.top[np.newaxis]

    point_flat = shape.flatten_points(point)
    point_reflat = shape.flatten_points(point_flat)
    points = np.vstack([point_flat, point_reflat])
    distances = shape.get_distances(points)
    assert_allclose(distances, 0, atol=1e-10, rtol=1e-10)


def test_normals_flatten_plane():
    for i in range(5):
        shape = get_shape(Plane, 100, canonical=False)
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
            shape = get_shape(primitive, 100, canonical=False)
            points = np.random.rand(1000, 3)
            points_flattened = shape.flatten_points(points)
            # distances = shape.get_distances(points_flattened)
            pcd_flattened = PointCloud(Vector3dVector(points_flattened))
            pcd_flattened.estimate_normals()
            normals = pcd_flattened.normals
            angles = shape.get_angles(points_flattened, normals)
            assert_allclose(rmse(angles), 0, atol=1e-1)


def test_translate_and_rotate():
    inlier_points = np.random.random((100, 3))
    inlier_normals = np.random.random((100, 3))
    inlier_normals /= np.linalg.norm(inlier_normals, axis=1)[:, None]

    translation = np.random.random(3) * 10
    position = np.random.random(3)
    position_translated = position + translation
    rotation = Rotation.from_quat(np.random.random(4))
    vector = np.random.random(3)
    vector /= np.linalg.norm(vector)
    vector_rotated = rotation.apply(vector)

    # testing both Plane and PlaneBounded
    plane = Plane.from_normal_point(vector, position)
    shapes = [plane, plane.get_square_plane()]
    for p in shapes:
        p.set_inliers(inlier_points, inlier_normals)
        p.translate(translation)
        assert_allclose(p.get_distances(position_translated), 0, atol=1e-10)
        p.rotate(rotation)
        assert_allclose(p.normal, vector_rotated)

    sphere = Sphere.from_center_radius(position, np.random.random())
    sphere.set_inliers(inlier_points, inlier_normals)
    sphere.translate(translation)
    assert_allclose(sphere.center, position_translated)
    sphere.rotate(rotation)
    # shapes.append(sphere)

    cylinder = Cylinder.from_base_vector_radius(
        position, vector, np.random.random())
    cylinder.set_inliers(inlier_points, inlier_normals)
    cylinder.translate(translation)
    assert_allclose(cylinder.base, position_translated)
    cylinder.rotate(rotation)
    assert_allclose(cylinder.vector, vector_rotated)
    # shapes.append(cylinder)

    cone = Cone.from_appex_vector_radius(
        position, vector, np.random.random())
    cone.set_inliers(inlier_points, inlier_normals)
    cone.translate(translation)
    assert_allclose(cone.appex, position_translated)
    cone.rotate(rotation)
    assert_allclose(cone.vector, vector_rotated)
    # shapes.append(cone)

    line = Line.from_point_vector(position, vector)
    line.set_inliers(inlier_points, inlier_normals)
    line.translate(translation)
    assert_allclose(line.beginning, position_translated)
    line.rotate(rotation)
    assert_allclose(line.vector, vector_rotated)

    shapes += [sphere, cylinder, cone, line]

    # testing all transformed inliers
    inlier_points = rotation.apply(inlier_points + translation)
    inlier_normals = rotation.apply(inlier_normals)

    for s in shapes:
        assert_allclose(s.inlier_points, inlier_points)
        assert_allclose(s.inlier_normals, inlier_normals)
        
        
# def test_bounding_box_bounds():
#     with pytest.warns(UserWarning, match='infinite axis aligned bounding'):
#         for primitive in all_primitives:
#             shape = get_shape(primitive, 10)
#             bbox = AxisAlignedBoundingBox(*shape.bbox_bounds)
#             bbox2 = shape.get_axis_aligned_bounding_box()
#             print(bbox)
#             print(bbox2)
#             assert np.all(bbox.min_bound == bbox.min_bound)
#             assert np.all(bbox.min_bound == bbox.max_bound)
#             # if not shape.get_axis_aligned_bounding_box() == bbox:
#                 # print(shape)
#                 # assert False


def test_axis_aligned_bounding_box_no_planes():
    num_samples = 1000
    for primitive in all_primitives_regular_bounded:
        shape = get_shape(Sphere, num_samples)
        pcd = shape.inlier_PointCloud.crop(shape.bbox)
        assert len(pcd.points) == num_samples
        
        
def test_axis_aligned_bounding_box_planes():
    plane_x = Plane([1, 0, 0, np.random.random()])
    plane_y = Plane([0, 1, 0, np.random.random()])
    plane_z = Plane([0, 0, 1, np.random.random()])
    
    for plane in [plane_x, plane_y, plane_z]:
        with pytest.warns(UserWarning, match='infinite axis aligned bounding'):
            assert sum(np.isinf(plane_y.bbox.min_bound)) == 2
            assert sum(np.isinf(plane_y.bbox.max_bound)) == 2
      
    plane_all = Plane.random()
    with pytest.warns(UserWarning, match='infinite axis aligned bounding'):
        assert sum(np.isinf(plane_all.bbox.min_bound)) == 3
        assert sum(np.isinf(plane_all.bbox.max_bound)) == 3
    
    # cannot be used because Open3D does not crop with inf in bounding box
    #
    # num_samples = 1000
    # with pytest.warns(UserWarning, match='infinite axis aligned bounding'):
    #     for plane in [plane_x, plane_y, plane_z, plane_all]:
    #         plane = plane.get_square_plane(np.random.random())
            
    #         plane.set_inliers(plane.sample_points_uniformly(num_samples))
    #         pcd = plane.inlier_PointCloud.crop(plane.bbox)
    #         assert len(pcd.points) == num_samples

# if __name__ == "__main__":
    # test_distances()
    # test_equal()
    # test_bounding_box_bounds()
    # test_axis_aligned_bounding_box_no_planes()
    # test_axis_aligned_bounding_box_planes()t
        
