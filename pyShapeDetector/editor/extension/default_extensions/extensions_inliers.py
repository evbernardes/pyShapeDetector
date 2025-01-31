#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-31 10:39:43

@author: evbernardes
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

MENU_NAME = "Inliers"

extensions = []


@_apply_to(Primitive)
def extract_inliers_from_primitives(shapes_input: list[Primitive]):
    shapes_empty = []
    inliers = []
    for shape in shapes_input:
        if len(shape.inliers.points) > 0:
            inliers.append(shape.inliers)
        shape = shape.copy()
        shape._inliers = PointCloud()
        shapes_empty.append(shape)

    return shapes_empty + inliers


extensions.append({"function": extract_inliers_from_primitives, "menu": MENU_NAME})


@_apply_to(Primitive)
def revert_primitives_to_inliers(shapes_input: list[Primitive]):
    return [s.inliers for s in shapes_input if len(s.inliers.points) > 0]


extensions.append({"function": revert_primitives_to_inliers, "menu": MENU_NAME})


@_apply_to(Primitive)
def remove_inliers_from_primitives(shapes_input: list[Primitive]):
    shapes = []
    for s in shapes_input:
        new_shape = s.copy()
        new_shape._inliers = PointCloud()
        shapes.append(new_shape)

    return shapes


extensions.append({"function": remove_inliers_from_primitives, "menu": MENU_NAME})


def add_pointclouds_as_inliers(elements: list[PointCloud, Primitive]):
    shapes, other = _extract_element_by_type(elements, Primitive)
    pcds, other = _extract_element_by_type(other, PointCloud)

    shapes = [shape.copy() for shape in shapes]

    if len(shapes) != 1:
        raise ValueError(f"Only one primitive should be given, got {len(shapes)}.")

    shapes[0]._inliers += PointCloud.fuse_pointclouds(pcds)

    return shapes + other


extensions.append({"function": add_pointclouds_as_inliers, "menu": MENU_NAME})
