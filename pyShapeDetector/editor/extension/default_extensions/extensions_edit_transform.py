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
from typing import TYPE_CHECKING
from scipy.spatial.transform import Rotation

from pyShapeDetector.geometry import (
    AxisAlignedBoundingBox,
    OrientedBoundingBox,
)
from pyShapeDetector.primitives import Plane
from pyShapeDetector.utility import get_rotation_from_axis

if TYPE_CHECKING:
    from pyShapeDetector.editor import Editor

MENU_NAME = "Edit/Transform..."

extensions = []


def _get_tranformation_matrix(rotation_center, rotation_matrix, translation_vector):
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

    return transformation_matrix


def _transform_elements_and_save_state(
    editor_instance: "Editor", indices, transformation_matrix
):
    if len(indices) == 0:
        return

    for index in indices:
        element = editor_instance.element_container.elements[index]
        element.transform(transformation_matrix)

    current_state = {
        "indices": copy.deepcopy(indices),
        "current_index": editor_instance.element_container.current_index,
        "operation": "transformation",
        "transformation_matrix": transformation_matrix,
    }

    editor_instance._save_state(current_state)


def transform_with_angles(
    editor_instance: "Editor",
    vector,
    angles_ZYX_degrees,
    reverse_translation,
    reverse_rotation,
    frame,
):
    indices = editor_instance.element_container.selected_indices
    if len(indices) == 0:
        return
    elements = [editor_instance.element_container[i] for i in indices]
    elements_raw = [element.raw for element in elements]

    try:
        bbox = OrientedBoundingBox.from_multiple_elements(elements_raw)
    except Exception:
        raise RuntimeError("Could not get Bounding Box")

    rotation_center = bbox.center

    rotation = Rotation.from_euler("zyx", angles_ZYX_degrees, degrees=True)

    if frame == "local":
        vector = bbox.R @ vector
        bbox_rotation = Rotation.from_matrix(np.array(bbox.R))
        rotation = bbox_rotation * rotation * bbox_rotation.inv()

    rotation_matrix = rotation.as_matrix()

    if reverse_rotation:
        rotation_matrix = rotation_matrix.T

    if reverse_translation:
        vector = -vector

    transformation_matrix = _get_tranformation_matrix(
        rotation_center, rotation_matrix, vector
    )

    _transform_elements_and_save_state(editor_instance, indices, transformation_matrix)


extensions.append(
    {
        "name": "Transform Selected",
        "function": transform_with_angles,
        "menu": MENU_NAME,
        "hotkey": "T",
        "lshift": "True",
        "inputs": "internal",
        "select_outputs": True,
        "parameters": {
            "vector": {"type": np.ndarray, "default": [0.0, 0.0, 0.0]},
            "angles_ZYX_degrees": {"type": np.ndarray, "default": [0.0, 0.0, 0.0]},
            "reverse_translation": {"type": bool},
            "reverse_rotation": {"type": bool},
            "frame": {"type": list, "options": ["local", "global"]},
        },
    }
)


def rotate_aligning_vectors(
    editor_instance: "Editor",
    vector_in,
    vector_out,
    reverse_rotation,
    translation,
    indices,
):
    if indices is None:
        indices = editor_instance.element_container.selected_indices

    if len(indices) == 0:
        return

    elements = [editor_instance.element_container[idx].raw for idx in indices]

    bbox = OrientedBoundingBox.from_multiple_elements(elements)
    vector_in /= np.linalg.norm(vector_in)
    vector_out /= np.linalg.norm(vector_out)

    R = get_rotation_from_axis(vector_in, vector_out)
    correction = Rotation.from_rotvec(180 * vector_in, degrees=True)
    R = R @ correction.as_matrix()

    if reverse_rotation:
        R = R.T

    transformation_matrix = _get_tranformation_matrix(bbox.center, R, translation)

    _transform_elements_and_save_state(editor_instance, indices, transformation_matrix)


extensions.append(
    {
        "name": "Rotate aligining vectors",
        "function": lambda editor_instance,
        vector_in,
        vector_out,
        reverse_rotation,
        translation: rotate_aligning_vectors(
            editor_instance, vector_in, vector_out, reverse_rotation, translation, None
        ),
        "menu": MENU_NAME,
        # "select_outputs": True,
        "inputs": "internal",
        "parameters": {
            "vector_in": {"type": np.ndarray, "default": [0.0, 0.0, 1.0]},
            "vector_out": {"type": np.ndarray, "default": [0.0, 0.0, 1.0]},
            "reverse_rotation": {"type": bool},
            "translation": {"type": np.ndarray, "default": [0.0, 0.0, 0.0]},
        },
    }
)


def _put_on_ground(editor_instance: "Editor"):
    indices = editor_instance.element_container.selected_indices
    if len(indices) == 0:
        return
    elements_raw = [editor_instance.element_container[i].raw for i in indices]

    bbox = AxisAlignedBoundingBox.from_multiple_elements(elements_raw)

    translation = -np.array([0, 0, bbox.min_bound[2]])

    transformation_matrix = _get_tranformation_matrix((0, 0, 0), np.eye(3), translation)

    _transform_elements_and_save_state(editor_instance, indices, transformation_matrix)


extensions.append(
    {
        "name": "Put on ground",
        "inputs": "internal",
        "function": _put_on_ground,
        "menu": MENU_NAME,
        "select_outputs": True,
    }
)


def _align_and_center_to_global_frame(editor_instance: "Editor"):
    indices = editor_instance.element_container.selected_indices
    if len(indices) == 0:
        return
    elements = [editor_instance.element_container[idx].raw for idx in indices]
    bbox = OrientedBoundingBox.from_multiple_elements(elements)

    # align element but place above XY plane
    translation = (0, 0, bbox.extent[2] / 2) - bbox.center

    transformation_matrix = _get_tranformation_matrix(
        bbox.center, bbox.R.T, translation
    )

    _transform_elements_and_save_state(editor_instance, indices, transformation_matrix)


extensions.append(
    {
        "name": "Align and center selected elements to global frame",
        "inputs": "internal",
        "function": _align_and_center_to_global_frame,
        "menu": MENU_NAME,
        "select_outputs": True,
    }
)


def _align_with_current_plane_as_ground(editor_instance: "Editor", ground_plane):
    if not isinstance(ground_plane, Plane):
        raise TypeError(
            f"Current (target) element should be a Plane, got:\n'{ground_plane}'."
        )

    if (N := len(editor_instance.element_container)) == 0:
        return

    vector_in = ground_plane.normal
    vector_out = np.array([0.0, 0.0, 1.0])

    # R = get_rotation_from_axis(vector_in, vector_out)
    # translation = -np.array([0, 0, ground_plane.get_signed_distances((0, 0, 0))])

    # print(f"R: {R}")
    # print(f"translation: {translation}")

    # translation = R @ translation
    # print(f"R x translation: {translation}")

    rotate_aligning_vectors(
        editor_instance,
        vector_in,
        vector_out,
        reverse_rotation=False,
        # translation=translation,
        translation=(0, 0, 0),
        indices=list(range(N)),
    )


extensions.append(
    {
        "name": "Align all elements with current plane as ground",
        "inputs": "internal",
        "function": _align_with_current_plane_as_ground,
        "menu": MENU_NAME,
        # "select_outputs": False,
        "parameters": {
            "ground_plane": {"type": "current"},
        },
    }
)
