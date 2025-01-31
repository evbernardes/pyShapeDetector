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

MENU_NAME = "Edit triangle meshes"

extensions = []


@_apply_to(Primitive)
def convert_primitives_to_meshes(shapes_input: list[Primitive], fuse: bool):
    meshes = [shape.mesh for shape in shapes_input]

    if not fuse:
        return meshes

    fused_mesh = TriangleMesh()
    for mesh in meshes:
        fused_mesh += mesh
    fused_mesh.remove_duplicated_vertices()

    return [fused_mesh]


extensions.append(
    {
        "function": convert_primitives_to_meshes,
        "menu": MENU_NAME,
        "parameters": {"fuse": {"type": bool}},
    }
)
