#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-13 11:12:42

@author: evbernardes
"""
import warnings
from pathlib import Path
from importlib.util import find_spec
from pyShapeDetector.geometry import PointCloud, TriangleMesh
from pyShapeDetector.primitives import Primitive

if has_h5py := find_spec("h5py") is not None:
    import h5py

RECOGNIZED_EXTENSION = {
    "Primitive": {
        ".tar": "Primitive / inliers bundled descriptors (.tar)",
        ".json": "Primitive descriptors (.json)",
        "loader": Primitive.load,
        "writer": Primitive.save,
        "default": ".tar"
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
        "default": ".stl"
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
        return False

    extensions = RECOGNIZED_EXTENSION[type_name]

    if path.suffix in (""):
        if path.stem[-1] == ".":
            path = path.with_stem(path.stem[:-1])
        path = path.with_suffix(extensions["default"])

    elif path.suffix not in extensions:
        warnings.warn(
            f"Suffix '{path.suffix}' invalid for elements of type {type_name}."
        )
        return False

    try:
        RECOGNIZED_EXTENSION[type_name]["writer"](element.raw, path.as_posix())
        return True
    except Exception:
        warnings.warn(f"Could not write element of {type_name} to {filename}.")
        return False
