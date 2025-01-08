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
from pyShapeDetector.geometry import (
    AxisAlignedBoundingBox,
    OrientedBoundingBox,
)
from pyShapeDetector.primitives import Plane
from pyShapeDetector.utility import get_rotation_from_axis
from .helpers import (
    _apply_to,
    _extract_element_by_type,
    _get_pointcloud_sizes,
    _get_shape_areas,
)

MENU_NAME = "Edit/Transform..."

extensions = []


def _transform_with_rotation_matrix_and_translation(
    elements, rotation_center, rotation_matrix, translation_vector
):
    transformed_elements = copy.deepcopy(elements)

    rotation_center = np.asarray(rotation_center)
    translation_vector = np.asarray(translation_vector)

    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -rotation_center

    # Step 2: Apply rotation
    rotation = np.eye(4)
    rotation[:3, :3] = rotation_matrix

    # Step 3: Translate back
    translate_back = np.eye(4)
    translate_back[:3, 3] = rotation_center

    # Step 4: Apply final translation
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation_vector

    # Combine transformations
    transformation_matrix = (
        translation_matrix @ translate_back @ rotation @ translate_to_origin
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
#         "menu": MENU_NAME,
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
        "menu": MENU_NAME,
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


def rotate_aligning_vectors(elements, vector_in, vector_out, reverse_rotation):
    bbox = OrientedBoundingBox.from_multiple_elements(elements)
    vector_in /= np.linalg.norm(vector_in)
    vector_out /= np.linalg.norm(vector_out)

    R = get_rotation_from_axis(vector_in, vector_out)
    correction = Rotation.from_rotvec(180 * vector_in, degrees=True)
    R = R @ correction.as_matrix()

    if reverse_rotation:
        R = R.T

    return _transform_with_rotation_matrix_and_translation(
        elements, bbox.center, R, (0, 0, 0)
    )


extensions.append(
    {
        "name": "Rotate aligining vectors",
        "function": rotate_aligning_vectors,
        "menu": MENU_NAME,
        "select_outputs": True,
        "parameters": {
            "vector_in": {"type": np.ndarray, "default": [0.0, 0.0, 1.0]},
            "vector_out": {"type": np.ndarray, "default": [0.0, 0.0, 1.0]},
            "reverse_rotation": {"type": bool},
        },
    }
)


def _put_on_ground(elements, copy_elements):
    bbox = AxisAlignedBoundingBox.from_multiple_elements(elements)

    translation = -np.array([0, 0, bbox.min_bound[2]])

    if copy_elements:
        transformed_elements = copy.deepcopy(elements)
    else:
        transformed_elements = elements

    if isinstance(transformed_elements, list):
        for elem in transformed_elements:
            elem.translate(translation)
    else:
        transformed_elements.translate(translation)

    return transformed_elements


extensions.append(
    {
        "name": "Put on ground",
        "function": lambda elements: _put_on_ground(elements, copy_elements=True),
        "menu": MENU_NAME,
        "select_outputs": True,
    }
)


def _align_and_center_to_global_frame(elements):
    bbox = OrientedBoundingBox.from_multiple_elements(elements)

    # align element but place above XY plane
    translation = (0, 0, bbox.extent[2] / 2) - bbox.center

    return _transform_with_rotation_matrix_and_translation(
        elements, bbox.center, bbox.R.T, translation
    )


extensions.append(
    {
        "name": "Align and center selected elements to global frame",
        "inputs": "selected",
        "function": _align_and_center_to_global_frame,
        "menu": MENU_NAME,
        "select_outputs": True,
    }
)


def _align_with_current_plane_as_ground(elements, ground_plane):
    if not isinstance(ground_plane, Plane):
        raise TypeError(
            f"Current (target) element should be a Plane, got:\n'{ground_plane}'."
        )

    vector_in = ground_plane.normal
    vector_out = np.array([0.0, 0.0, 1.0])

    transformed_elements = rotate_aligning_vectors(
        elements, vector_in, vector_out, False
    )

    return _put_on_ground(transformed_elements, copy_elements=False)


extensions.append(
    {
        "name": "Align with current plane as ground",
        "inputs": "selected",
        "function": _align_with_current_plane_as_ground,
        "menu": MENU_NAME,
        "select_outputs": False,
        "parameters": {
            "ground_plane": {"type": "current"},
        },
    }
)
