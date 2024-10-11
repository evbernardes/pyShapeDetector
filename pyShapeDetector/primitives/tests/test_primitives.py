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
import tempfile
from itertools import combinations, pairwise

from scipy.spatial.transform import Rotation

from pyShapeDetector.primitives import (
    Primitive,
    Plane,
    PlaneBounded,
    PlaneTriangulated,
    PlaneRectangular,
    Sphere,
    Cylinder,
    Cone,
    Line,
)

from pyShapeDetector.geometry import PointCloud

all_primitives_regular = [Plane, Sphere, Cylinder, Cone]
all_primitives = all_primitives_regular + [
    PlaneBounded,
    PlaneTriangulated,
    PlaneRectangular,
    Line,
]

all_primitives_regular_bounded = [
    PlaneBounded,
    PlaneTriangulated,
    PlaneRectangular,
    Sphere,
    Cylinder,
    Cone,
]
all_primitives_bounded = all_primitives_regular_bounded + [Line]


def normalized(x):
    return x / np.linalg.norm(x)


def rmse(x):
    """Helper for root mean square error."""
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


def test_plane_creation_methods():
    # axis=None flattens the array before sorting
    sort = lambda x: np.sort(x, axis=None)
    # sort = lambda x: np.sort(x, axis=0)

    vx = np.random.random(3)
    vy = np.random.random(3)
    normal = np.cross(vx, vy)
    vy = np.cross(normal, vx)
    point = np.random.random(3)

    for primitive in [Plane, PlaneBounded, PlaneTriangulated, PlaneRectangular]:
        shape1 = primitive.from_vectors_center(np.array([vx, vy]), point)
        shape2 = primitive.from_normal_point(shape1.normal, shape1.centroid)
        shape3 = primitive.from_normal_dist(shape1.normal, shape1.dist)

        assert shape1.is_similar_to(shape2)
        assert shape2.is_similar_to(shape3)
        assert shape3.is_similar_to(shape1)

        if hasattr(shape1, "vertices"):
            # assert_allclose(np.sort(shape1.vertices), np.sort(shape2.vertices))
            assert_allclose(sort(shape2.vertices), sort(shape3.vertices))

    dimensions = abs(np.random.random(3) * 10)
    boxes = []
    for primitive in [PlaneBounded, PlaneTriangulated, PlaneRectangular]:
        boxes.append(primitive.create_box(center=point, dimensions=dimensions))

    for box1, box2 in combinations(boxes, 2):
        for plane1, plane2 in zip(box1, box2):
            assert np.all(sort(plane1.mesh.triangles) == sort(plane2.mesh.triangles))
            assert_allclose(sort(plane1.mesh.vertices), sort(plane2.mesh.vertices))


def test_rectangular_plane_from_vectors():
    vx = np.random.random(3)
    vz = np.random.random(3)
    vy = np.cross(vz, vx)
    point = np.random.random(3)

    plane = PlaneBounded.from_vectors_center([vx, vy], point)
    assert (plane.surface_area - np.linalg.norm(vx) * np.linalg.norm(vy)) < 1e-4


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

    plane_bounded_two = plane_unbounded.get_bounded_plane(plane_bounded.vertices)
    assert plane_bounded_two.has_inliers


def test_init_primitives_regular():
    for primitive in [Plane, Sphere, Cylinder]:
        model = np.random.rand(primitive._model_args_n)
        primitive(model)

        with pytest.raises(ValueError, match="elements, got"):
            model = np.random.rand(primitive._model_args_n + 1)
            primitive(model)

        with pytest.raises(ValueError, match="elements, got"):
            model = np.random.rand(primitive._model_args_n - 1)
            primitive(model)


def test_init_plane_bounded():
    model = np.random.rand(PlaneBounded._model_args_n)
    vertices = np.random.random((50, 3))

    with pytest.warns(UserWarning, match="returning square plane"):
        PlaneBounded(model)

    PlaneBounded(model, vertices)


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

    box_unbounded = Plane.create_box(center, dimensions)
    for p in box_unbounded:
        assert isinstance(p, Plane)

    box_bounded = PlaneBounded.create_box(center, dimensions)
    for p in box_bounded:
        assert isinstance(p, PlaneBounded)

    box_triangulated = PlaneTriangulated.create_box(center, dimensions)
    for p in box_triangulated:
        assert isinstance(p, PlaneTriangulated)

    box_rectangular = PlaneRectangular.create_box(center, dimensions)
    for p in box_rectangular:
        assert isinstance(p, PlaneRectangular)
        assert_allclose(p.mesh.vertices.mean(axis=0), p.center)


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
                assert np.all(shape.holes[0].vertices == shape_copy.holes[0].vertices)
                assert id(shape.holes[0].vertices) != id(shape_copy.holes[0].vertices)


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

        assert abs(plane.get_distances(center)) < 1e-8

        # making vectors slightly bigger for boundary points
        V1 *= 1 + 1e-7
        V2 *= 1 + 1e-7
        plane_rect = plane.get_rectangular_plane((V1, V2), center)

        assert np.all(plane_rect.contains_projections(plane.inliers.points))
        assert_allclose(
            plane.canonical.model, plane_rect.canonical.model, rtol=1e-01, atol=1e-01
        )


def test_plane_surface_area_and_volume():
    model = np.random.rand(4)
    plane = Plane(model)
    assert plane.volume == 0
    with pytest.warns(UserWarning, match="surface area is undefined"):
        assert np.isnan(plane.surface_area)

    length = np.random.rand()
    plane_rectangular = plane.get_square_plane(length)
    assert isinstance(plane_rectangular, PlaneRectangular)
    assert plane_rectangular.volume == 0
    assert_allclose(plane_rectangular.surface_area, length**2)

    # vertices = np.array(plane_bounded.mesh.vertices)
    # triangles = np.array(plane_bounded.mesh.triangles)

    # get only half of the doubled triangles
    # triangles = triangles[:int(len(triangles) / 2)]

    plane_triangulated = PlaneTriangulated.from_plane_with_mesh(plane_rectangular)
    assert_allclose(plane_triangulated.surface_area, length**2)


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
    plane = PlaneBounded(plane)
    area_no_holes = plane.surface_area

    hole1 = plane.get_square_plane(2)
    hole2 = hole1.copy()
    hole1.translate([-2, 0, 0])
    hole2.translate([2, 0, 0])

    plane.add_holes([hole1])
    assert_allclose(plane.surface_area, area_no_holes - hole1.surface_area)

    plane.add_holes([hole2])
    assert_allclose(
        plane.surface_area, area_no_holes - hole1.surface_area - hole2.surface_area
    )

    assert_allclose(plane.surface_area, plane.mesh.get_surface_area())


def test_fit():
    # testing Cylinder separately
    for i in range(5):
        shape = get_shape(Cylinder, 1000, canonical=True)
        shape_fit = Cylinder.fit(
            shape.inliers.points, normals=shape.inliers.normals
        ).canonical
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
                shape.inliers.points, normals=shape.inliers.normals
            ).canonical
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

    cylinder = Cylinder.from_base_vector_radius(position, vector, np.random.random())
    cylinder.set_inliers(inlier_points, inlier_normals)
    cylinder.translate(translation)
    assert_allclose(cylinder.base, position_translated)
    cylinder.rotate(rotation)
    assert_allclose(cylinder.vector, vector_rotated)
    # shapes.append(cylinder)

    cone = Cone.from_appex_vector_radius(position, vector, np.random.random())
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
        pcd = shape.inliers.crop(shape.aabb)
        assert len(pcd.points) == num_samples


def test_axis_aligned_bounding_box_planes():
    plane_x = Plane([1, 0, 0, np.random.random()])
    plane_y = Plane([0, 1, 0, np.random.random()])
    plane_z = Plane([0, 0, 1, np.random.random()])

    for plane in [plane_x, plane_y, plane_z]:
        with pytest.warns(UserWarning, match="infinite bounding boxes"):
            assert sum(np.isinf(plane_y.aabb.min_bound)) == 2
            assert sum(np.isinf(plane_y.aabb.max_bound)) == 2

    plane_all = Plane.random()
    with pytest.warns(UserWarning, match="infinite bounding boxes"):
        assert sum(np.isinf(plane_all.aabb.min_bound)) == 3
        assert sum(np.isinf(plane_all.aabb.max_bound)) == 3


def test_planes_group_aabb_intersection():
    # num_samples = 50
    centroid = np.array([0, 0, 0])
    normal = np.array([0, 0, 1])
    direction = np.array([1, 0, 0])
    side = 2
    dist = 0.1
    eps = 1e-3
    p = Plane.from_normal_point(normal, centroid).get_square_plane(side)
    pleft = p.copy()
    pleft.translate(-direction * (side + dist))
    pright = p.copy()
    pright.translate(+direction * (side + dist))

    # slightly below limit
    assert 1 == len(
        Primitive.group_similar_shapes([pleft, p], aabb_intersection=dist + eps)
    )
    assert 1 == len(
        Primitive.group_similar_shapes([pright, p], aabb_intersection=dist + eps)
    )

    # slightly above limit
    assert 2 == len(
        Primitive.group_similar_shapes([pleft, p], aabb_intersection=dist - eps)
    )
    assert 2 == len(
        Primitive.group_similar_shapes([pleft, p], aabb_intersection=dist - eps)
    )

    # all planes
    assert 1 == len(
        Primitive.group_similar_shapes([pleft, p, pright], aabb_intersection=dist + eps)
    )
    assert 3 == len(
        Primitive.group_similar_shapes([pleft, p, pright], aabb_intersection=dist - eps)
    )


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
    pleft.translate(-direction * (side + dist))
    pright = p.copy()
    pright.translate(+direction * (side + dist))

    with pytest.warns(UserWarning, match="No inliers found"):
        Primitive.group_similar_shapes([pleft, p], inlier_max_distance=eps)

    # adding points...
    p.set_inliers(p.sample_points_uniformly(num_samples))
    pleft.set_inliers(pleft.sample_points_uniformly(num_samples))
    pright.set_inliers(pright.sample_points_uniformly(num_samples))

    # ... and now it should work:
    # slightly below limit
    assert 1 == len(
        Primitive.group_similar_shapes([pleft, p], inlier_max_distance=dist + eps)
    )
    assert 1 == len(
        Primitive.group_similar_shapes([pright, p], inlier_max_distance=dist + eps)
    )

    # slightly above limit
    assert 2 == len(
        Primitive.group_similar_shapes([pleft, p], inlier_max_distance=dist - eps)
    )
    assert 2 == len(
        Primitive.group_similar_shapes([pleft, p], inlier_max_distance=dist - eps)
    )

    # all planes
    assert 1 == len(
        Primitive.group_similar_shapes(
            [pleft, p, pright], inlier_max_distance=dist + eps
        )
    )
    assert 3 == len(
        Primitive.group_similar_shapes(
            [pleft, p, pright], inlier_max_distance=dist - eps
        )
    )


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
    #         pcd = plane.inlier_PointCloud.crop(plane.aabb)
    #         assert len(pcd.points) == num_samples


def test_plane_bounded_degenerated_line():
    model = [0.61302647, 0.10281596, 0.78334375, -10.56448992]

    projections = np.array(
        [
            [0.06210584, 1.73375236],
            [0.07249562, 1.68208517],
            [0.07249562, 1.68208517],
            [0.06210584, 1.73375236],
        ]
    )

    plane = Plane(model)
    vertices = plane.get_points_from_projections(projections)

    with pytest.warns(UserWarning, match="Convex hull failed"):
        plane.get_bounded_plane(vertices)


def test_save_load():
    def test(shape, temp_dir, extension, save_inliers):
        path = temp_dir + "shape" + extension

        shape.save(path, save_inliers=save_inliers)
        primitive = type(shape)
        shape_loaded = primitive.load(path)
        assert_allclose(shape.model, shape_loaded.model)
        assert_allclose(shape.color, shape_loaded.color)

        if save_inliers:
            assert_allclose(
                shape.inliers.points, shape_loaded.inliers.points, atol=1e-10
            )
            assert_allclose(
                shape.inliers.normals, shape_loaded.inliers.normals, atol=1e-10
            )
            assert_allclose(
                shape.inliers.colors, shape_loaded.inliers.colors, atol=1e-10
            )

        if primitive is PlaneBounded:
            assert_allclose(shape.vertices, shape_loaded.vertices)

            if len(shape.holes) > 0:
                for hole in shape.holes:
                    test(hole, temp_dir, extension, save_inliers=False)

        if primitive is PlaneTriangulated:
            assert_allclose(shape.vertices, shape_loaded.vertices)
            assert np.all(shape.triangles == shape_loaded.triangles)

        # if primitive in [Plane, PlaneBounded, PlaneTriangulated]:
        #     shape_with_parallels = shape.copy()
        #     assert shape_with_parallels.parallel_vectors is None
        #     shape_with_parallels.set_parallel_vectors()
        #     assert shape_with_parallels.parallel_vectors.shape == (2, 3)

        #     shape_with_parallels.save(path, save_inliers=save_inliers)
        #     shape_with_parallels_loaded = primitive.load(path)
        #     assert shape_with_parallels_loaded.parallel_vectors.shape == (2, 3)

    with pytest.warns(UserWarning, match="returning square plane"):
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(10):
                for primitive in all_primitives:
                    shape = primitive.random()

                    if primitive is PlaneBounded:
                        sides = np.random.randint(3, 8)
                        radius = np.sqrt(shape.surface_area / np.pi) / 3
                        center = np.median(shape.vertices, axis=0)
                        hole = shape.get_polygon_plane(sides, radius, center)
                        shape.add_holes([hole])

                    assert not shape.has_inliers
                    with pytest.warns(UserWarning, match="consider saving"):
                        test(shape, temp_dir, ".json", True)
                    # with pytest.warns(UserWarning, match="consider saving"):
                    test(shape, temp_dir, ".json", False)
                    test(shape, temp_dir, ".tar", True)
                    test(shape, temp_dir, ".tar", False)

                    shape.set_inliers(shape.sample_PointCloud_uniformly(100))

                    assert shape.has_inliers
                    with pytest.warns(UserWarning, match="consider saving"):
                        test(shape, temp_dir, ".json", True)
                    test(shape, temp_dir, ".tar", True)


def test_planetriangulated_translate():
    plane = PlaneTriangulated.random()
    assert_allclose(plane.vertices, plane.mesh.vertices)

    plane.translate(np.random.random(3))
    assert_allclose(plane.vertices, plane.mesh.vertices)

    plane_bounded = PlaneBounded.random()
    # with pytest.warns(UserWarning, match="using plane's mesh"):
    plane = PlaneTriangulated(plane_bounded)

    assert_allclose(plane.vertices, plane.mesh.vertices)

    plane_bounded.translate(np.random.random(3))
    assert_allclose(plane.vertices, plane.mesh.vertices)


def test_fuse():
    n = 10
    for primitive in all_primitives:
        shapes = [primitive.random() for i in range(n)]
        colors = [np.random.random(3) for i in range(n)]
        # models = [shape.model for shape in shapes]

        for color, shape in zip(colors, shapes):
            shape.color = color

        shape_fused = primitive.fuse(shapes)
        assert_allclose(shape_fused.color, np.mean(colors, axis=0))
        # assert_allclose(shape_fused.model, np.mean(models, axis=0))


def test_line_checks():
    points = np.random.random([50, 3]) * 10

    plane = PlaneBounded.fit(points)
    points = plane.flatten_points(points)
    plane.set_inliers(points)
    lines = plane.vertices_lines

    for line1, line2 in combinations(lines, 2):
        assert line1.check_coplanar(line2)
        assert not line1.check_colinear(line2)

    # out of plane
    lines[0].translate(plane.normal)

    for line2 in lines[1:]:
        assert not lines[0].check_coplanar(line2)

    # back to plane, displaced along planne
    lines[0].translate(-plane.normal)
    delta = np.cross(plane.normal, np.random.random(3))
    lines[0].translate(delta)
    for line2 in lines[1:]:
        assert lines[0].check_coplanar(line2)
    lines[0].translate(-delta)  # back to boundary

    for line in lines:
        point = line.beginning + np.random.random() * line.axis
        other_line = Line.from_point_vector(point, line.axis)
        assert line.check_colinear(other_line)
        other_line.translate(delta)
        assert not line.check_colinear(other_line)
        other_line.translate(-delta)

    lines.append(lines[0].copy())

    for line1, line2 in pairwise(lines):
        intersection = line1.point_from_intersection(line2)
        assert_allclose(intersection, line1.ending)
        line1.translate(plane.normal)
        assert line1.point_from_intersection(line2) is None


def test_glue_convex_planes_with_line():
    for i in range(20):
        vz = normalized(np.random.random(3))
        vx = normalized(np.cross(np.random.random(3), vz))
        vy = normalized(np.cross(vz, vx))

        delta = np.random.random()
        shrink = np.random.random()
        dims1 = np.random.random(2)

        # assuring dims2[0] is smaller than dims2[0]
        dims2 = np.array([dims1[0] * (shrink**2), np.random.random()])
        assert dims2[0] < dims1[0]

        vectors1 = dims1[np.newaxis].T * (vx, vy)
        vectors2 = dims2[np.newaxis].T * (vx, vz)

        plane1 = PlaneRectangular.from_vectors_center(vectors1)
        plane2 = PlaneRectangular.from_vectors_center(vectors2)
        plane1.translate(plane1.normal * dims2[1] / 2)
        plane2.translate(plane2.normal * dims1[1] / 2)

        np.testing.assert_allclose(vectors1, plane1.parallel_vectors)
        np.testing.assert_allclose(vectors2, plane2.parallel_vectors)
        np.testing.assert_allclose(dims1, plane1.dimensions)
        np.testing.assert_allclose(dims2, plane2.dimensions)

        plane1 = PlaneBounded(plane1)
        plane2 = PlaneBounded(plane2)

        area1 = plane1.surface_area
        area2 = plane2.surface_area

        np.testing.assert_allclose(area1, np.prod(dims1))
        np.testing.assert_allclose(area2, np.prod(dims2))

        plane1.translate(delta * plane1.normal)
        plane2.translate(delta * plane2.normal)

        line = PlaneBounded.get_plane_intersections([plane1, plane2])[0, 1]

        line1 = line.get_fitted_to_points(plane1.vertices)
        line2 = line.get_fitted_to_points(plane2.vertices)

        np.testing.assert_allclose(line.length, dims1[0])
        np.testing.assert_allclose(line1.length, dims1[0])
        np.testing.assert_allclose(line2.length, dims2[0])

        line_shrinked = line.get_extended(shrink)
        # np.testing.assert_allclose(line1.length * shrink, line_shrinked.length)

        assert line_shrinked.length < line1.length  # smaller, adds small trapezoid:
        plane1.add_line(line_shrinked)
        trapezoid1_small = delta * (line_shrinked.length + line1.length) / 2
        np.testing.assert_allclose(plane1.surface_area, area1 + trapezoid1_small)

        assert line_shrinked.length > line2.length  # bigger, surface is a trapezoid:
        plane2.add_line(line_shrinked)
        trapezoid2_big = (delta + dims2[1]) * (line_shrinked.length + line2.length) / 2
        np.testing.assert_allclose(plane2.surface_area, trapezoid2_big)


# if __name__ == "__main__":
# test_plane_transformations()
# test_distances()
# test_equal()
# test_bounding_box_bounds()
# test_axis_aligned_bounding_box_no_planes()
# test_axis_aligned_bounding_box_planes()t
