#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:01:29 2024

@author: ebernardes
"""

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

MENU_NAME = "Misc"

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
def convert_to_sampled_points(shapes_input: list[Primitive], num_points: int):
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
