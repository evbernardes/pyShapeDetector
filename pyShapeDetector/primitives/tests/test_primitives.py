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

from pyShapeDetector.primitives import (
    Primitive, Plane, PlaneBounded, PlaneTriangulated, Sphere, Cylinder, Cone, Line)

from pyShapeDetector.geometry import PointCloud

all_primitives_regular = [Plane, Sphere, Cylinder, Cone]
all_primitives = all_primitives_regular + [PlaneBounded, PlaneTriangulated, Line]
all_primitives_regular_bounded = [PlaneBounded, PlaneTriangulated, Sphere, Cylinder, Cone]
all_primitives_bounded = [PlaneBounded, PlaneTriangulated, Sphere, Cylinder, Cone] + [Line]

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
    

def test_plane_transformations():
    points = np.random.random((20, 3))

    plane = Plane.random()
    plane_bounded = plane.get_square_plane(1)
    plane_triangulated = PlaneTriangulated.from_plane_with_mesh(plane_bounded)
    assert not plane.has_inliers
    assert not plane_bounded.has_inliers
    assert not plane_triangulated.has_inliers

    plane_bounded.set_inliers(points)
    plane_triangulated = PlaneTriangulated.from_plane_with_mesh(plane_bounded)
    assert not plane.has_inliers
    assert plane_bounded.has_inliers
    assert plane_triangulated.has_inliers

    plane_unbounded = plane_bounded.get_unbounded_plane()
    assert not plane.has_inliers
    assert plane_unbounded.has_inliers
    assert plane_bounded.has_inliers

    plane_bounded_two = plane_unbounded.get_bounded_plane(plane_bounded.bounds)
    assert plane_bounded_two.has_inliers


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


def test_plane_circle():
    center = (0, 0, 0)
    normal = (0, 0, 1)
    radius = 1

    plane = Plane.create_circle(center, normal, radius)
    assert type(plane) is Plane

    plane = PlaneBounded.create_circle(center, normal, radius)
    assert type(plane) is PlaneBounded

    plane = PlaneTriangulated.create_circle(center, normal, radius)
    assert type(plane) is PlaneTriangulated


def test_plane_ellipse():
    center = (0, 0, 0)
    vx = (1, 0, 0)
    vy = (0, 2, 0)

    plane1 = Plane.create_ellipse(center, vx, vy)
    assert type(plane1) is Plane
    print(type(plane1))

    plane2 = PlaneBounded.create_ellipse(center, vx, vy)
    assert type(plane2) is PlaneBounded
    print(type(plane2))

    plane3 = PlaneTriangulated.create_ellipse(center, vx, vy)
    assert type(plane3) is PlaneTriangulated
    print(type(plane3))


def test_plane_box():
    center = (0, 0, 0)
    dimensions = (1, 2, 3)

    box = Plane.create_box(center, dimensions)
    for p in box:
        assert type(p) is Plane

    box = PlaneBounded.create_box(center, dimensions)
    for p in box:
        assert type(p) is PlaneBounded

    box = PlaneTriangulated.create_box(center, dimensions)
    for p in box:
        assert type(p) is PlaneTriangulated


def test_canonical():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for primitive in all_primitives:
            for _ in range(10):
                shape = primitive.random(decimals=4)
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
        plane = Plane.random(decimals=6)
        assert plane == plane.copy()
        # assert assert_allclose(model, model_copy, rtol=eps, atol=eps)


def test_copy_regular():
    for primitive in all_primitives_regular:
        shape = get_shape(primitive, 100)

        shape_copy = shape.copy()
        assert shape == shape_copy
        assert id(shape) != id(shape_copy)
        assert id(shape.inliers) != id(shape_copy.inliers)
        # assert id(shape.inliers.points) != id(shape_copy.inliers.points)
        
        shape_copy = copy.copy(shape)
        assert shape == shape_copy
        assert id(shape) != id(shape_copy)
        assert id(shape.inliers) != id(shape_copy.inliers)
        # assert id(shape.inliers.points) != id(shape_copy.inliers.points)


def test_copy_planebounded():

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shape = get_shape(PlaneBounded, 100)
        hole = PlaneBounded.create_circle(
            center=shape.centroid, normal=shape.normal, radius=0.1
        )
        shape.add_holes(hole)

        assert shape.is_convex
        assert shape.copy().is_convex

        for shape_copy in [shape.copy(), copy.copy(shape)]:
            assert shape == shape_copy
            assert id(shape) != id(shape_copy)
            assert id(shape.inliers.points) != id(shape_copy.inliers.points)
            assert shape.holes == shape_copy.holes
            if len(shape.holes) > 0:
                assert id(shape.holes[0]) != id(shape_copy.holes[0])
                assert np.all(shape.holes[0].bounds == shape_copy.holes[0].bounds)
                assert id(shape.holes[0].bounds) != id(shape_copy.holes[0].bounds)


def test_deepcopy():
    for primitive1, primitive2 in combinations(all_primitives, 2):
        shapes = [get_shape(primitive1, 20), get_shape(primitive2, 20)]
        shapes_copy = copy.deepcopy(shapes)

        assert shapes[0] == shapes_copy[0]
        assert shapes[1] == shapes_copy[1]
        assert_allclose(shapes[0].inliers.points, shapes_copy[0].inliers.points)
        assert_allclose(shapes[0].inliers.points, shapes_copy[0].inliers.points)
        assert id(shapes[0]) != id(shapes_copy[1])
        assert id(shapes[1]) != id(shapes_copy[1])
        assert id(shapes[0].inliers.points) != id(shapes_copy[0].inliers.points)
        assert id(shapes[0].inliers.points) != id(shapes_copy[0].inliers.points)


def test_get_rectangular_vectors_from_points():
    for i in range(5):
        plane = Plane.random()
        plane.set_inliers(np.random.random([100, 3]), flatten=True)

        (V1, V2), center = plane.get_rectangular_vectors_from_points(return_center=True)
        
        # making vectors slightly bigger for boundary points
        V1 *= 1 + 1e-7
        V2 *= 1 + 1e-7
        plane_rect = plane.get_rectangular_plane((V1, V2), center)

        assert np.all(plane_rect.contains_projections(plane.inliers.points))


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
    
    # vertices = np.array(plane_bounded.mesh.vertices)
    # triangles = np.array(plane_bounded.mesh.triangles)

    # get only half of the doubled triangles
    # triangles = triangles[:int(len(triangles) / 2)]

    plane_triangulated = PlaneTriangulated.from_plane_with_mesh(plane_bounded)
    assert_allclose(plane_triangulated.surface_area, length ** 2)


def test_planebounded_contains_projections():
    for n in range(3, 7):
        length = np.random.random() * 10
        plane = Plane.random()
        square_small = plane.get_polygon_plane(n, length)
        square_big = plane.get_polygon_plane(n, length * 2)
        square_small.color = (0, 0, 0)
        points_small = square_small.sample_points_uniformly(10000)
        points_big = square_big.sample_points_uniformly(10000)

        small_inside_small = square_small.contains_projections(points_small)
        small_inside_big = square_big.contains_projections(points_small)
        big_inside_small = square_small.contains_projections(points_big)
        big_inside_big = square_big.contains_projections(points_big)

        assert np.all(small_inside_small)
        assert np.all(small_inside_big)
        assert not np.all(big_inside_small)
        assert np.any(big_inside_small)
        assert np.all(big_inside_big)

        # "donut" is the points of the big square that are not in the small square
        donut = points_big[~big_inside_small]
        donut_inside_small = square_small.contains_projections(donut)
        donut_inside_big = square_big.contains_projections(donut)
        assert not np.any(donut_inside_small)
        assert np.all(donut_inside_big)

        # "inside both" are the points of the big square that should also be in the 
        # small square
        inside_both = points_big[big_inside_small]
        inside_both_inside_small = square_small.contains_projections(inside_both)
        inside_both_inside_big = square_big.contains_projections(inside_both)
        assert np.all(inside_both_inside_small)
        assert np.all(inside_both_inside_big)


def test_planebounded_convex_tringulation():
    plane = Plane([0, 0, 1, 0]).get_square_plane(10)
    area_no_holes = plane.surface_area

    hole1 = plane.get_square_plane(2)
    hole2 = hole1.copy()
    hole1.translate([-2, 0, 0])
    hole2.translate([2, 0, 0])

    plane.add_holes([hole1])
    assert_allclose(plane.surface_area, area_no_holes - hole1.surface_area)

    plane.add_holes([hole2])
    assert_allclose(plane.surface_area, area_no_holes - hole1.surface_area - hole2.surface_area)

    assert_allclose(plane.surface_area, plane.mesh.get_surface_area())


def test_fit():
    # testing Cylinder separately
    for i in range(5):
        shape = get_shape(Cylinder, 1000, canonical=True)
        shape_fit = Cylinder.fit(
            shape.inliers.points, normals=shape.inliers.normals).canonical
        # test axis instead of vector for direct fit
        assert_allclose(shape.axis, shape_fit.axis, rtol=1e-2, atol=1e-2)
        assert_allclose(shape.radius, shape_fit.radius, rtol=1e-2, atol=1e-2)

    # # testing Cone separately
    # for i in range(5):
    #     shape = get_shape(Cone, 10000, canonical=True)
        # pcd = shape.inliers
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
                shape.inliers.points, normals=shape.inliers.normals).canonical
            assert_allclose(shape.model, shape_fit.model, rtol=1e-2, atol=1e-2)


def test_distances():
    for primitive in all_primitives_regular:
        for i in range(5):
            shape = get_shape(primitive, 100, canonical=False)
            distances = shape.get_distances(shape.inliers.points)
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
        pcd_flattened = PointCloud(points_flattened)
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
            pcd_flattened = PointCloud(points_flattened)
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
    shapes = [plane, plane.get_square_plane(1)]
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
        assert_allclose(s.inliers.points, inlier_points)
        assert_allclose(s.inliers.normals, inlier_normals)
        
        
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
        pcd = shape.inliers.crop(shape.bbox)
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
    assert 1 == len(Primitive.group_similar_shapes([pleft, p], bbox_intersection=dist+eps))
    assert 1 == len(Primitive.group_similar_shapes([pright, p], bbox_intersection=dist+eps))
    
    # slightly above limit
    assert 2 == len(Primitive.group_similar_shapes([pleft, p], bbox_intersection=dist-eps))
    assert 2 == len(Primitive.group_similar_shapes([pleft, p], bbox_intersection=dist-eps))
    
    # all planes
    assert 1 == len(Primitive.group_similar_shapes([pleft, p, pright], bbox_intersection=dist+eps))
    assert 3 == len(Primitive.group_similar_shapes([pleft, p, pright], bbox_intersection=dist-eps))
    
    
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
        Primitive.group_similar_shapes([pleft, p], inlier_max_distance=eps)
        
    # adding points...
    p.set_inliers(p.sample_points_uniformly(num_samples))
    pleft.set_inliers(pleft.sample_points_uniformly(num_samples))
    pright.set_inliers(pright.sample_points_uniformly(num_samples))
    
    # ... and now it should work:
    # slightly below limit
    assert 1 == len(Primitive.group_similar_shapes([pleft, p], inlier_max_distance=dist+eps))
    assert 1 == len(Primitive.group_similar_shapes([pright, p], inlier_max_distance=dist+eps))
    
    # slightly above limit
    assert 2 == len(Primitive.group_similar_shapes([pleft, p], inlier_max_distance=dist-eps))
    assert 2 == len(Primitive.group_similar_shapes([pleft, p], inlier_max_distance=dist-eps))
    
    # all planes
    assert 1 == len(Primitive.group_similar_shapes([pleft, p, pright], inlier_max_distance=dist+eps))
    assert 3 == len(Primitive.group_similar_shapes([pleft, p, pright], inlier_max_distance=dist-eps))
    
    
def test_box_intersections():
    length = 5
    
    planes = PlaneBounded.create_box((0, 0, 0), length)
    intersections = Plane.get_plane_intersections(planes, 1e-3)
    assert len(intersections) == 12
    
    planes[0].translate(planes[0].normal * 10)
    intersections = Plane.get_plane_intersections(planes, 1e-3)
    assert len(intersections) == 12 - 4
    intersections = Plane.get_plane_intersections(planes, 10 + 1e-3)
    assert len(intersections) == 12
    
    # cannot be used because Open3D does not crop with inf in bounding box
    #
    # num_samples = 1000
    # with pytest.warns(UserWarning, match='infinite axis aligned bounding'):
    #     for plane in [plane_x, plane_y, plane_z, plane_all]:
    #         plane = plane.get_square_plane(np.random.random())
            
    #         plane.set_inliers(plane.sample_points_uniformly(num_samples))
    #         pcd = plane.inlier_PointCloud.crop(plane.bbox)
    #         assert len(pcd.points) == num_samples


def test_plane_bounded_degenerated_line():
    model = [0.61302647, 0.10281596, 0.78334375, -10.56448992]

    projections = np.array([
        [0.06210584, 1.73375236],
        [0.07249562, 1.68208517],
        [0.07249562, 1.68208517],
        [0.06210584, 1.73375236]])
    
    plane = Plane(model)
    bounds = plane.get_points_from_projections(projections)

    with pytest.warns(UserWarning, match='Convex hull failed'):
        plane.get_bounded_plane(bounds)

    

# if __name__ == "__main__":
    # test_plane_transformations()
    # test_distances()
    # test_equal()
    # test_bounding_box_bounds()
    # test_axis_aligned_bounding_box_no_planes()
    # test_axis_aligned_bounding_box_planes()t
        
