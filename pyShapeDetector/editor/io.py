#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-13 11:12:42

@author: evbernardes
"""
import traceback
import warnings
import tempfile
import tarfile
import json
import numpy as np
from typing import Union
from pathlib import Path
from importlib.util import find_spec
from pyShapeDetector.geometry import PointCloud, TriangleMesh
from pyShapeDetector.primitives import Primitive
from pyShapeDetector.editor import Editor

if has_h5py := find_spec("h5py") is not None:
    import h5py

SCENE_FILE_EXTENSION = ".sdscene"

RECOGNIZED_EXTENSION = {
    "Primitive": {
        ".tar": "Primitive / inliers bundled descriptors (.tar)",
        ".json": "Primitive descriptors (.json)",
        "loader": Primitive.load,
        "writer": Primitive.save,
        "default": ".tar",
    },
    "PointCloud": {
        ".pcd": "Point Cloud Data files (.pcd)",
        ".ply": "Polygon files (.ply)",
        ".xyz": "ASCII point cloud files (.xyz)",
        ".xyzn": "ASCII point cloud with normals (.xyzn)",
        ".xyzrgb": "ASCII point cloud files with colors (.xyzrgb)",
        ".pts": "3D Points files (.pts)",
        "loader": PointCloud.read_point_cloud,
        "writer": PointCloud.write_point_cloud,
        "default": ".pcd",
    },
    "TriangleMesh": {
        ".stl": "Stereolithography files (.stl)",
        ".fbx": "Autodesk Filmbox files (.fbx)",
        ".obj": "Wavefront OBJ files (.obj)",
        ".off": "Object file format (.off)",
        ".gltf": "OpenGL transfer files (.gltf)",
        ".glb": "OpenGL binary transfer files (.glb)",
        "loader": TriangleMesh.read_triangle_mesh,
        "writer": TriangleMesh.write_triangle_mesh,
        "default": ".stl",
    },
}

if has_h5py:
    RECOGNIZED_EXTENSION["PointCloud"][
        ".h5"
    ] = "Point Cloud in Hierarchical Data Format (.h5)"

for type_name, extensions in RECOGNIZED_EXTENSION.items():
    extensions["all"] = " ".join([key for key in extensions.keys() if key[0] == "."])
    extensions["all_description"] = f"{type_name} files ({extensions['all']})"

RECOGNIZED_EXTENSION["all"] = " ".join(
    [extensions["all"] for extensions in RECOGNIZED_EXTENSION.values()]
)


def _load_one_element(filename):
    path = Path(filename)
    element = None
    for key, extensions in RECOGNIZED_EXTENSION.items():
        if path.suffix in extensions:
            try:
                element = extensions["loader"](filename)
            except Exception:
                warnings.warn(f"Could not load {key} in {filename}.")
            break
    else:
        warnings.warn(f"File in {filename} is not recognized.")
    return element


def _write_one_element(element, filename):
    path = Path(filename)
    type_name = type(element.raw).__name__

    if type_name not in RECOGNIZED_EXTENSION:
        warnings.warn("Cannot export elements of type '{RECOGNIZED_EXTENSION}'.")
        return None

    extensions = RECOGNIZED_EXTENSION[type_name]

    if path.suffix in (""):
        if path.stem[-1] == ".":
            path = path.with_stem(path.stem[:-1])
        path = path.with_suffix(extensions["default"])

    elif path.suffix not in extensions:
        warnings.warn(
            f"Suffix '{path.suffix}' invalid for elements of type {type_name}."
        )
        return None

    try:
        RECOGNIZED_EXTENSION[type_name]["writer"](element.raw, path.as_posix())
        return path

    except Exception:
        warnings.warn(f"Could not write element of {type_name} to {filename}.")
        return None


def _open_scene(input_path: Union[Path, str], editor_instance: Editor):
    input_path = Path(input_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(input_path, "r") as tar:
            tar.extractall(path=temp_dir)

            path_elements = Path(temp_dir) / "elements"
            path_elements_fixed = Path(temp_dir) / "elements_fixed"

            new_elements = []
            new_elements_fixed = []

            if path_elements.exists():
                for path in path_elements.glob("*"):
                    new_elements.append(_load_one_element(path))

            if path_elements_fixed.exists():
                for path in path_elements_fixed.glob("*"):
                    new_elements_fixed.append(_load_one_element(path))

            if len(editor_instance.elements) > 0:
                editor_instance.elements.pop_multiple(
                    range(len(editor_instance.elements)), from_gui=True
                )

            editor_instance.elements.insert_multiple(new_elements, to_gui=True)
            editor_instance._update_info()
            editor_instance._future_states = []
            editor_instance._past_states = []


def _save_scene(path: Union[Path, str], editor_instance: Editor):
    path = Path(path)
    if path.exists():
        path.unlink()

    elements = editor_instance.elements
    elements_fixed = editor_instance._elements_fixed
    # scene = editor_instance.scene

    if path.suffix == "":
        path = path.with_suffix(SCENE_FILE_EXTENSION)

    elif path.suffix != SCENE_FILE_EXTENSION:
        warnings.warn(f"Extension '{path.suffix}' invalid.")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(path, "w") as tar:
            temp_dir = Path(temp_dir)
            # json_file_path = temp_dir / "preferences.json"
            # with open(json_file_path, "w") as json_file:
            #     json_data = {}
            #     for param in editor_instance._settings._dict.values():
            #         value = param.value
            #         if isinstance(value, np.ndarray):
            #             value = value.tolist()
            #         json_data[param.name] = value
            #     json.dump(json_data, json_file, indent=4)
            #     tar.add(json_file_path, arcname="preferences.json")

            if len(elements) > 0:
                elements_directory = temp_dir / "elements"
                elements_directory.mkdir()
                for i, element in enumerate(elements):
                    element_path = elements_directory / f"element_{i}"
                    path_out = _write_one_element(element, element_path)
                    try:
                        path_out = _write_one_element(element, element_path)
                        tar.add(
                            path_out, arcname=f"elements/element_{i}" + path_out.suffix
                        )
                    except Exception:
                        traceback.print_exc()
                        pass

            if len(elements_fixed) > 0:
                elements_directory = temp_dir / "elements_fixed"
                elements_directory.mkdir()
                for i, element in enumerate(elements_fixed):
                    element_path = elements_directory / f"element_{i}"
                    try:
                        path_out = _write_one_element(element, element_path)
                        tar.add(
                            path_out,
                            arcname=f"elements_fixed/element_{i}" + path_out.suffix,
                        )
                    except:
                        pass

    # else:
    #     raise ValueError(
    #         f"Acceptable extensions are 'tar' and 'json', got {path.suffix}."
    #     )
