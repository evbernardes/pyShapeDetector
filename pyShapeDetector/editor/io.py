#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-13 11:12:42

Helpers for saving and loading Elements and scenes.

Attributes
----------
SCENE_FILE_EXTENSION
RECOGNIZED_EXTENSION


Methods
-------
_create_overwrite_warning
_load_one_element
_write_one_element
_open_scene
_save_scene

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
from open3d.visualization import gui
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
        "default": ".ply",
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


def _create_overwrite_warning(
    editor_instance: Editor, path: str, quitting: bool = False
):
    """Creates Warning dialog when trying to save to existing file."""
    window = editor_instance._window

    dlg = gui.Dialog("Warning")

    em = window.theme.font_size
    dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

    title = gui.Horiz()
    title.add_stretch()
    title.add_child(gui.Label("Overwrite warning"))
    title.add_stretch()

    dlg_layout.add_child(title)
    dlg_layout.add_child(gui.Label(f"File {path} already exists. Overwrite?"))

    def _on_yes():
        editor_instance._scene_file_path = path
        editor_instance._close_dialog()
        editor_instance._internal_functions._cb_save_scene()
        if quitting:
            editor_instance._closing_app = True
            editor_instance._window.close()

    def _on_no():
        editor_instance._close_dialog()

    yes = gui.Button("Yes")
    yes.set_on_clicked(_on_yes)
    no = gui.Button("No")
    no.set_on_clicked(_on_no)

    title = gui.Horiz()
    title.add_stretch()
    title.add_child(yes)
    title.add_stretch()
    title.add_child(no)
    title.add_stretch()
    dlg_layout.add_child(title)

    dlg.add_child(dlg_layout)
    window.show_dialog(dlg)


def _load_one_element(filename):
    """Loads one element into the ElementContainer."""
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
    """Write one element from the ElementContainer."""
    path = Path(filename)
    if isinstance(element.raw, Primitive):
        type_name = "Primitive"
    else:
        type_name = type(element.raw).__name__

    if type_name not in RECOGNIZED_EXTENSION:
        warnings.warn(f"Cannot export elements of type '{type_name}'.")
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
    """Replace all elements from the current ElementContainer with elements from a file."""
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

            json_path = Path(temp_dir) / "preferences.json"
            if not json_path.exists():
                return

            try:
                with open(json_path, "r") as json_file:
                    json_data = json.load(json_file)
            except Exception:
                warnings.warn("Could not load preferences file from '{input_path}'.")
                return

            keys = ["PointCloud_density"]
            for key in keys:
                if key not in json_data:
                    warnings.warn(
                        f"{key} not found in preferences from '{input_path}'."
                    )
                    continue

                try:
                    editor_instance._settings.set_setting(key, json_data[key])
                except Exception:
                    warnings.warn(
                        f"Could set '{key}':{json_data[key]}  from preferences in "
                        f"file '{input_path}' into settings."
                    )
                    traceback.print_exc()


def _save_scene(path: Union[Path, str], editor_instance: Editor):
    """Save all elements from the current ElementContainer into a file, overwriting."""
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
            json_file_name = "preferences.json"
            temp_json_file_path = temp_dir / json_file_name

            json_data = {}
            for key, param in editor_instance._settings._dict.items():
                if isinstance(param.value, np.ndarray):
                    json_data[key] = param.value.tolist()
                else:
                    json_data[key] = param.value

            print(json_data)

            with open(temp_json_file_path, "w") as fp:
                json.dump(json_data, fp, indent=4)

            tar.add(temp_json_file_path, arcname=json_file_name)

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
                    except Exception:
                        pass

    # else:
    #     raise ValueError(
    #         f"Acceptable extensions are 'tar' and 'json', got {path.suffix}."
    #     )
