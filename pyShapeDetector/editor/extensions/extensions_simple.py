#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:01:29 2024

@author: ebernardes
"""
import traceback
import warnings
import numpy as np
import copy
from scipy.spatial.transform import Rotation

# from pyShapeDetector.utility import get_inputs, select_function_with_gui
from pyShapeDetector.geometry import PointCloud, TriangleMesh, OrientedBoundingBox
from pyShapeDetector.primitives import Primitive, PlaneBounded, Plane
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
        "function": remove_small_pcds,
        "menu": MENU_NAME,
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
def remove_small_surfaces(shapes, min_area):
    return [s for s in shapes if s.surface_area >= min_area]


extensions.append(
    {
        "function": remove_small_surfaces,
        "menu": MENU_NAME,
        "parameters": {
            "min_area": {
                "type": float,
                "limits": (1, 10000),
                "limit_setter": _get_shape_areas,
            }
        },
    }
)


def _transform_with_rotation_matrix_and_translation(
    elements, rotation_center, rotation_matrix, vector
):
    transformed_elements = copy.deepcopy(elements)

    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -rotation_center

    # Step 2: Apply rotation
    rotation = np.eye(4)
    rotation[:3, :3] = rotation_matrix

    # Step 3: Translate back
    translate_back = np.eye(4)
    translate_back[:3, 3] = rotation_center

    # Step 4: Apply final translation
    translation = np.eye(4)
    translation[:3, 3] = vector

    # Combine transformations
    transformation_matrix = (
        translation @ translate_back @ rotation @ translate_to_origin
    )

    if isinstance(transformed_elements, list):
        for elem in transformed_elements:
            elem.transform(transformation_matrix)
    else:
        transformed_elements.transform(transformation_matrix)

    # print(transformation_matrix)

    return transformed_elements


def transform_with_angles(
    elements, vector, angles_ZYX_degrees, reverse_translation, reverse_rotation
):
    # transformed_elements = copy.deepcopy(elements)

    if not isinstance(elements, list):
        try:
            bbox = elements.get_oriented_bounding_box()
            vector = bbox.R @ vector
            rotation_center = bbox.center
            rotation = Rotation.identity()
            for angle, rot_axis in zip(angles_ZYX_degrees, bbox.R.T):
                rotation *= Rotation.from_rotvec(rot_axis * angle, degrees=True)
            rotation_matrix = rotation.as_matrix()

        except Exception:
            warnings.warn("Could not get Bounding Box")
            rotation_center = np.array([0.0, 0.0, 0.0])
            rotation_matrix = Rotation.from_euler(
                "zyx", angles_ZYX_degrees, degrees=True
            ).as_matrix()
    else:
        vector = vector
        rotation_matrix = Rotation.from_euler(
            "zyx", angles_ZYX_degrees, degrees=True
        ).as_matrix()
        rotation_center = np.array([0.0, 0.0, 0.0])
        try:
            for elem in elements:
                rotation_center += elem.get_oriented_bounding_box().center
            rotation_center /= len(elements)
        except Exception:
            warnings.warn("Could not get center for transform.")
            traceback.print_exc()

    if reverse_rotation:
        rotation = rotation.T

    if reverse_translation:
        vector = -vector

    return _transform_with_rotation_matrix_and_translation(
        elements, rotation_center, rotation_matrix, vector
    )


# extensions.append(
#     {
#         "name": "Transform Current",
#         "function": transform,
#         "inputs": "current",
#         "menu": "Edit",
#         "hotkey": "T",
#         "parameters": {
#             "vector": {"type": np.ndarray, "default": [0.0, 0.0, 0.0]},
#             "angles_ZYX_degrees": {"type": np.ndarray, "default": [0, 0, 0.0]},
#             "reverse_translation": {"type": bool},
#             "reverse_rotation": {"type": bool},
#         },
#     }
# )

extensions.append(
    {
        "name": "Transform Selected",
        "function": transform_with_angles,
        "menu": "Edit",
        "hotkey": "T",
        "lshift": "True",
        "select_outputs": True,
        "parameters": {
            "vector": {"type": np.ndarray, "default": [0.0, 0.0, 0.0]},
            "angles_ZYX_degrees": {"type": np.ndarray, "default": [0.0, 0.0, 0.0]},
            "reverse_translation": {"type": bool},
            "reverse_rotation": {"type": bool},
        },
    }
)


def _align_and_center_to_global_frame(elements):
    if isinstance(elements, list):
        bboxes = [element.get_oriented_bounding_box() for element in elements]
        points = np.vstack([bbox.get_box_points() for bbox in bboxes])
        bbox = OrientedBoundingBox.create_from_points(points)
    else:
        bbox = elements.get_oriented_bounding_box()

    return _transform_with_rotation_matrix_and_translation(
        elements, bbox.center, bbox.R.T, -bbox.center
    )


extensions.append(
    {
        "name": "Align and center elements to global frame",
        "inputs": "selected",
        "function": _align_and_center_to_global_frame,
        "menu": "Edit",
        "select_outputs": True,
    }
)


def fuse_elements(elements):
    shapes, other = _extract_element_by_type(elements, PlaneBounded)
    pcds, other = _extract_element_by_type(other, PointCloud)

    try:
        shapes = [PlaneBounded.fuse(shapes)]
        print(len(shapes))
    except Exception as e:
        shapes = shapes
        print(e)

    pcd = PointCloud.fuse_pointclouds(pcds)
    if len(pcd.points) > 0:
        return shapes + [pcd] + other
    return shapes + other


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
def revert_to_inliers(shapes_input):
    return [s.inliers for s in shapes_input]


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
