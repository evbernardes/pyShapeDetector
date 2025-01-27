#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-14 10:13:36

@author: evbernardes
"""
from typing import TYPE_CHECKING
import numpy as np
from pyShapeDetector.primitives import Primitive
from pyShapeDetector.geometry import PointCloud

if TYPE_CHECKING:
    from pyShapeDetector.editor import Editor

extensions = []


def estimate_pointcloud_density(
    editor_instance: "Editor", PointCloud_density, number_of_neighbors, split
):
    pcds = []
    for element in editor_instance.element_container.elements:
        if not element.is_selected:
            continue

        raw = element.raw
        if isinstance(raw, PointCloud):
            pcds.append(raw)
        elif isinstance(raw, Primitive) and len(raw.inliers.points) > 0:
            pcds.append(raw.inliers)

    if len(pcds) == 0:
        raise RuntimeError("No pointclouds selected, cannot estimate density.")

    density = np.mean(
        [pcd.average_nearest_dist(k=number_of_neighbors, split=split) for pcd in pcds]
    )
    editor_instance._settings.print_debug(f"Estimated PointCloud density: {density}")
    editor_instance._settings.set_setting("PointCloud_density", density)


extensions.append(
    {
        "function": estimate_pointcloud_density,
        "menu": "Edit",
        "inputs": "internal",
        "parameters": {
            "PointCloud_density": {"type": "preference"},
            "number_of_neighbors": {"type": int, "limits": (3, 50), "default": 10},
            "split": {
                "type": int,
                "default": 30,
                "limits": (1, 30),
            },
        },
    }
)
