#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-23 13:29:06

@author: evbernardes
"""
import warnings
from pathlib import Path
from typing import TYPE_CHECKING
from ..io import _write_one_element, RECOGNIZED_EXTENSION
# from ..element import ElementPrimitive, ElementPointCloud, ElementTriangleMesh

from pyShapeDetector.primitives import Primitive
from pyShapeDetector.geometry import PointCloud, TriangleMesh

if TYPE_CHECKING:
    from pyShapeDetector.editor import Editor

extensions = []
MENU_NAME = "File/Export"


def _export_multiple(
    editor_instance: "Editor",
    directory: Path,
    extension_primitive,
    extension_pointcloud,
    extension_trianglemesh,
):
    indices = editor_instance.element_container.selected_indices
    for idx in indices:
        element = editor_instance.element_container.elements[idx]

        if isinstance(element.raw, Primitive):
            suffix = extension_primitive
        elif isinstance(element.raw, PointCloud):
            suffix = extension_pointcloud
        elif isinstance(element.raw, TriangleMesh):
            suffix = extension_trianglemesh
        else:
            warnings.warn(
                f"Cannot export element {element}, writer for this type is undefined."
            )
            continue

        element_path = (directory / f"element_{idx}").with_suffix(suffix)
        _write_one_element(element, element_path)


def export_selected_to_directory(
    editor_instance: "Editor",
    directory: Path,
    extension_primitive,
    extension_pointcloud,
    extension_trianglemesh,
):
    if not directory.exists():
        directory.mkdir()

    _export_multiple(
        editor_instance,
        directory,
        extension_primitive,
        extension_pointcloud,
        extension_trianglemesh,
    )


extensions.append(
    {
        "name": "Export selected elements to directory",
        "function": export_selected_to_directory,
        "menu": MENU_NAME,
        "inputs": "internal",
        "parameters": {
            "directory": {"type": "path"},
            "extension_primitive": {
                "type": list,
                "options": [
                    suffix
                    for suffix in RECOGNIZED_EXTENSION["Primitive"]
                    if "." in suffix
                ],
                "default": RECOGNIZED_EXTENSION["Primitive"]["default"],
            },
            "extension_pointcloud": {
                "type": list,
                "options": [
                    suffix
                    for suffix in RECOGNIZED_EXTENSION["PointCloud"]
                    if "." in suffix
                ],
                "default": RECOGNIZED_EXTENSION["PointCloud"]["default"],
            },
            "extension_trianglemesh": {
                "type": list,
                "options": [
                    suffix
                    for suffix in RECOGNIZED_EXTENSION["TriangleMesh"]
                    if "." in suffix
                ],
                "default": RECOGNIZED_EXTENSION["TriangleMesh"]["default"],
            },
        },
    }
)
