#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:01:29 2024

@author: ebernardes
"""

# from pyShapeDetector.utility import get_inputs, select_function_with_gui
from pyShapeDetector.geometry import PointCloud, TriangleMesh
from pyShapeDetector.primitives import (
    Primitive,
    PlaneBounded,
    Plane,
    Cylinder,
    Sphere,
    Cone,
)
from .helpers import (
    _apply_to,
    _extract_element_by_type,
    _get_pointcloud_sizes,
    _get_shape_areas,
)

MENU_NAME = "Simple functions"

extensions = []


@_apply_to(PointCloud)
def remove_small_pcds(pcds_input, min_points):
    return [pcd for pcd in pcds_input if len(pcd.points) >= min_points]


extensions.append(
    {
        "name": "PCDs by number of points",
        "function": remove_small_pcds,
        "menu": MENU_NAME + "/Remove small...",
        "parameters": {
            "min_points": {
                "type": int,
                "limits": (1, 100),
                "limit_setter": _get_pointcloud_sizes,
            }
        },
    }
)


@_apply_to(Primitive)
def remove_small_shapes_by_surface_area(shapes, min_area):
    return [s for s in shapes if s.surface_area >= min_area]


extensions.append(
    {
        "name": "Shapes per surface area",
        "function": remove_small_shapes_by_surface_area,
        "menu": MENU_NAME + "/Remove small...",
        "parameters": {
            "min_area": {
                "type": float,
                "limits": (1, 10000),
                "limit_setter": _get_shape_areas,
            }
        },
    }
)


@_apply_to(TriangleMesh)
def remove_small_meshes_by_surface_area(meshes, min_area):
    return [mesh for mesh in meshes if mesh.surface_area >= min_area]


extensions.append(
    {
        "name": "Meshes per surface area",
        "function": remove_small_meshes_by_surface_area,
        "menu": MENU_NAME + "/Remove small...",
        "parameters": {
            "min_area": {
                "type": float,
                "limits": (1, 10000),
                "limit_setter": _get_shape_areas,
            }
        },
    }
)


def fuse_elements(elements):
    primitives = [PlaneBounded, Cylinder, Sphere, Cone]

    shapes_per_type = {}
    rest = elements
    for primitive in primitives:
        input_shapes, rest = _extract_element_by_type(rest, primitive)
        if len(input_shapes) > 0:
            # print(f"Found {len(input_shapes)} instances of {primitive}.")
            shapes_per_type[primitive] = input_shapes
    pcds, rest = _extract_element_by_type(rest, PointCloud)

    shapes = []
    for primitive, input_shapes in shapes_per_type.items():
        try:
            # print(f"Fusing {primitive}:")
            shape = primitive.fuse(input_shapes)
            shapes.append(shape)
            # print(f"Found fused primitive: {shape}")
        except Exception as e:
            print(f"Could not fuse {len(input_shapes)} elements of type {primitive}:")
            print(e)
            shapes += input_shapes

    # print(f"Returning {len(shapes)} shapes.")

    pcd = PointCloud.fuse_pointclouds(pcds)
    if len(pcd.points) > 0:
        return shapes + [pcd] + rest
    return shapes + rest


extensions.append(
    {
        "function": fuse_elements,
        "select_outputs": True,
        "menu": MENU_NAME,
        "hotkey": 2,
    }
)


@_apply_to(Primitive)
def fuse_primitives_as_mesh(shapes_input):
    mesh = TriangleMesh()
    for shape in shapes_input:
        mesh += shape.mesh
    mesh.remove_duplicated_vertices()
    return [mesh]


extensions.append({"function": fuse_primitives_as_mesh, "menu": MENU_NAME})


@_apply_to(Primitive)
def extract_inliers(shapes_input):
    shapes_empty = []
    inliers = []
    for shape in shapes_input:
        if len(shape.inliers.points) > 0:
            inliers.append(shape.inliers)
        shape = shape.copy()
        shape._inliers = PointCloud()
        shapes_empty.append(shape)

    return shapes_empty + inliers


extensions.append({"function": extract_inliers, "menu": MENU_NAME})


@_apply_to(Primitive)
def revert_to_inliers(shapes_input):
    return [s.inliers for s in shapes_input if len(s.inliers.points) > 0]


extensions.append({"function": revert_to_inliers, "menu": MENU_NAME})


@_apply_to(Primitive)
def remove_inliers(shapes_input):
    shapes = []
    for s in shapes_input:
        new_shape = s.copy()
        new_shape._inliers = PointCloud()
        shapes.append(new_shape)

    return shapes


extensions.append({"function": remove_inliers, "menu": MENU_NAME})


def add_pcds_as_inliers(elements):
    shapes, other = _extract_element_by_type(elements, Primitive)
    pcds, other = _extract_element_by_type(other, PointCloud)

    shapes = [shape.copy() for shape in shapes]

    if len(shapes) != 1:
        raise ValueError(f"Only one primitive should be given, got {len(shapes)}.")

    shapes[0]._inliers += PointCloud.fuse_pointclouds(pcds)
    # shapes = [s.copy() for s in shapes]
    # new_inliers = {}
    # # pcds = [PointCloud(p) for p in pcds]
    # # new_inliers = [s.inliers for s in shapes]

    # for pcd in pcds:
    #     distances = [min(shape.get_distances(pcd.points)) for shape in shapes]
    #     idx = np.argmin(distances)

    #     if idx not in new_inliers:
    #         new_inliers[idx] = [shapes[idx].inliers]

    #     new_inliers[idx].append(pcd)

    # for i, pcds in new_inliers.items():
    #     shape = shapes[i]
    #     shape.set_inliers(PointCloud.fuse_pointclouds(pcds))
    #     shape.color = shape.inliers.colors.mean(axis=0)

    return shapes + other


extensions.append({"function": add_pcds_as_inliers, "menu": MENU_NAME})


@_apply_to(Primitive)
def convert_to_sampled_points(shapes_input, num_points):
    return [s.sample_PointCloud_uniformly(num_points) for s in shapes_input]


extensions.append(
    {
        "function": convert_to_sampled_points,
        "menu": MENU_NAME,
        "parameters": {
            "num_points": {"type": int, "default": 1000, "limits": (1, 10000)},
        },
    }
)


def get_distance(elements):
    if len(elements) != 2:
        raise ValueError("Expected two elements.")

    distance = None

    elem1, elem2 = elements
    if isinstance(elem1, PointCloud) and isinstance(elem2, PointCloud):
        distance = min(elem1.get_distances(elem2.points))

    if isinstance(elem1, Plane) and isinstance(elem2, Plane):
        distance = elem1.get_distances(elem2.centroid)

    if isinstance(elem1, Plane) and isinstance(elem2, PointCloud):
        elem1, elem2 = elem2, elem1

    if isinstance(elem1, PointCloud) and isinstance(elem2, Plane):
        distance = min(elem2.get_distances(elem1.points))

    if distance is None:
        print(f"Could not calculate distance between {elem1} and {elem2}")
    else:
        print(f"Distance: {distance}")

    return elements


extensions.append({"function": get_distance, "menu": MENU_NAME})
