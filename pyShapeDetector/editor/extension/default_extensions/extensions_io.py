#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-23 13:29:06

@author: evbernardes
"""
import warnings
import tempfile
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING
from ...io import _write_one_element, RECOGNIZED_EXTENSION

from pyShapeDetector.primitives import Primitive
from pyShapeDetector.geometry import PointCloud, TriangleMesh

if TYPE_CHECKING:
    from pyShapeDetector.editor import Editor
    from pyShapeDetector.editor.element import Element

extensions = []
MENU_NAME = "File/Export"

ALL_SUFFIXES_SEPARATED = {}

for type_name, file_extensions in RECOGNIZED_EXTENSION.items():
    if type_name == "all":
        continue

    for suffix, description in file_extensions.items():
        if suffix[0] != ".":
            continue
        ALL_SUFFIXES_SEPARATED[suffix] = description


def _export_multiple(
    editor_instance: "Editor",
    indices: list["Element"],
    directory: Path,
    extension_primitive,
    extension_pointcloud,
    extension_trianglemesh,
):
    filenames = []
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

        try:
            filename = Path(f"element_{idx}").with_suffix(suffix)
            _write_one_element(element, directory / filename)
            filenames.append(filename)
        except Exception:
            warnings.warn("Could not save element {element}.")
    return filenames


# def export_current(
#     editor_instance: "Editor",
#     path: Path,
# ):
#     directory = path.parent

#     if not directory.exists():
#         directory.mkdir()

#     element = editor_instance.element_container.current_element
#     idx = editor_instance.element_container.current_index

#     if path.stem == "":
#         path = path.with_stem(f"element_{idx}")

#     _write_one_element(element, path, raise_error=True)


# extensions.append(
#     {
#         "name": "Export current element",
#         "function": export_current,
#         "menu": MENU_NAME,
#         "inputs": "internal",
#         "hotkey": "E",
#         "lctrl": True,
#         "parameters": {
#             "path": {
#                 "type": "path",
#                 "path_type": "save",
#                 "suffixes": ALL_SUFFIXES_SEPARATED,
#             },
#         },
#     }
# )


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
        editor_instance.element_container.selected_indices,
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
            "directory": {"type": "path", "path_type": "open_dir"},
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


def export_selected_to_tar_file(
    editor_instance: "Editor",
    path: Path,
    extension_primitive,
    extension_pointcloud,
    extension_trianglemesh,
):
    if path.suffix == "":
        path = path.with_suffix(".tar")

    elif path.suffix != ".tar":
        raise ValueError(f"Extension '{path.suffix}' invalid.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        filenames = _export_multiple(
            editor_instance,
            editor_instance.element_container.selected_indices,
            temp_dir,
            extension_primitive,
            extension_pointcloud,
            extension_trianglemesh,
        )

        with tarfile.open(path, "w") as tar:
            for filename in filenames:
                tar.add(temp_dir / filename, arcname=filename)


extensions.append(
    {
        "name": "Export selected elements to tar file",
        "function": export_selected_to_tar_file,
        "menu": MENU_NAME,
        "inputs": "internal",
        "parameters": {
            "path": {
                "type": "path",
                "path_type": "save",
                "suffixes": {".tar": "Uncompressed tar files"},
            },
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
