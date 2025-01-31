#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-30 15:09:48

@author: evbernardes
"""
import pytest
from pathlib import Path
import tempfile
import warnings
import numpy as np
from pyShapeDetector.editor import Editor
from pyShapeDetector.editor.extension import Extension
from pyShapeDetector.primitives import Cylinder, Sphere
from pyShapeDetector.geometry import TriangleMesh, PointCloud

param_radius = {"radius": {"type": float, "default": 0}}


def _get_test_elements():
    elements = [
        Cylinder.random(),
        Sphere.random(),
        Cylinder.random().sample_PointCloud_uniformly(50),
        Sphere.random().sample_PointCloud_uniformly(50),
        Cylinder.random().mesh,
        Sphere.random().mesh,
    ]

    is_selected = np.random.choice([True, False], len(elements)).tolist()
    is_hidden = np.random.choice([True, False], len(elements)).tolist()

    return elements, is_selected, is_hidden


def test_save_scene_open_scene():
    old_instance = Editor(load_default_extensions=False)
    new_instance = Editor(load_default_extensions=False)

    elements, is_selected, is_hidden = _get_test_elements()

    old_instance.element_container.insert_multiple(
        elements, is_selected=is_selected, is_hidden=is_hidden
    )

    assert len(old_instance.element_container.elements) == len(elements)
    assert len(new_instance.element_container.elements) == 0

    for i, element in enumerate(old_instance.element_container.elements):
        assert element.is_selected == is_selected[i]
        assert element.is_hidden == is_hidden[i]

    assert old_instance._scene_file_path is None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        path = temp_dir / "invalid_suffix.scene"
        with pytest.warns(UserWarning, match="Extension '.scene' invalid."):
            old_instance._internal_functions._cb_save_scene(path=path)
        assert old_instance._scene_file_path is None
        assert not path.exists()

        path = temp_dir / "valid_no_suffix"
        old_instance._internal_functions._cb_save_scene(path=path)
        old_instance._wait_for_active_threads()
        assert old_instance._scene_file_path == path.with_suffix(".sdscene")
        assert path.with_suffix(".sdscene").exists()

        path = temp_dir / "valid_suffix.sdscene"
        old_instance._internal_functions._cb_save_scene(path=path)
        old_instance._wait_for_active_threads()
        assert old_instance._scene_file_path == path
        assert path.exists()

        new_instance._internal_functions._cb_open_scene(path)

    assert len(new_instance.element_container.elements) == len(elements)

    for i, element in enumerate(new_instance.element_container.elements):
        assert isinstance(element.raw, type(elements[i]))
        assert element.is_selected == is_selected[i]
        assert element.is_hidden == is_hidden[i]


# def test_import_export():
#     old_instance = Editor(load_default_extensions=False)
#     new_instance = Editor(load_default_extensions=False)

#     elements, _, _ = _get_test_elements()

#     old_instance.element_container.insert_multiple(elements)

#     with tempfile.TemporaryDirectory() as temp_dir:
#         temp_dir = Path(temp_dir)

#         for i in range(len(old_instance.element_container.elements)):
#             path = temp_dir / f"element_{i}"
#             old_instance.element_container.update_current_index(i)
#             old_instance._internal_functions._cb_export_current_element(path)
