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
        ".json": "Primitive descriptors (.json)",
        ".tar": "Primitive / inliers bundled descriptors (.tar)",
        "loader": Primitive.load,
    },
    "PointCloud": {
        ".ply": "Polygon files (.ply)",
        ".xyz": "ASCII point cloud files (.xyz)",
        ".xyzn": "ASCII point cloud with normals (.xyzn)",
        ".xyzrgb": "ASCII point cloud files with colors (.xyzrgb)",
        ".pcd": "Point Cloud Data files (.pcd)",
        ".pts": "3D Points files (.pts)",
        "loader": PointCloud.read_point_cloud,
    },
    "TriangleMesh": {
        ".stl": "Stereolithography files (.stl)",
        ".fbx": "Autodesk Filmbox files (.fbx)",
        ".obj": "Wavefront OBJ files (.obj)",
        ".off": "Object file format (.off)",
        ".gltf": "OpenGL transfer files (.gltf)",
        ".glb": "OpenGL binary transfer files (.glb)",
        "loader": TriangleMesh.read_triangle_mesh,
    },
}

if has_h5py:
    RECOGNIZED_EXTENSION["PointCloud"][
        ".h5"
    ] = "Point Cloud in Hierarchical Data Format (.h5)"

for type, extensions in RECOGNIZED_EXTENSION.items():
    extensions["all"] = " ".join([key for key in extensions.keys() if key[0] == "."])
    extensions["all_description"] = f"{type} files ({extensions['all']})"
RECOGNIZED_EXTENSION["all"] = " ".join(
    [type["all"] for type in RECOGNIZED_EXTENSION.values()]
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
