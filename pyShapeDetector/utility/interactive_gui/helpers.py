#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import numpy as np

from open3d.utility import Vector3dVector
from open3d.visualization import gui


def get_pretty_name(func):
    return func.__name__.replace("_", " ").capitalize()


def parse_parameters_as_kwargs(parameters):
    return {name: value["default"] for name, value in parameters.items()}


def get_key_name(key):
    if isinstance(key, gui.KeyName):
        return str(key).split(".")[1]
    elif isinstance(key, int):
        return chr(key)


def extract_element_colors(drawable_element):
    if hasattr(drawable_element, "vertex_colors"):
        return np.asarray(drawable_element.vertex_colors).copy()
    if hasattr(drawable_element, "mesh"):
        return np.asarray(drawable_element.mesh.vertex_colors).copy()
    if hasattr(drawable_element, "color"):
        return np.asarray(drawable_element.color).copy()
    if hasattr(drawable_element, "colors"):
        return np.asarray(drawable_element.colors).copy()

    warnings.warn("Could not get color from element {element}.")
    return None


def set_element_colors(element, input_color):
    if input_color is None:
        return

    color = np.clip(input_color, 0, 1)

    if hasattr(element, "vertex_colors"):
        if color.ndim == 2:
            element.vertex_colors = Vector3dVector(color)
        else:
            element.paint_uniform_color(color)

    elif hasattr(element, "colors"):
        if color.ndim == 2:
            element.colors = Vector3dVector(color)
        else:
            element.paint_uniform_color(color)

    elif hasattr(element, "color"):
        element.color = color


def get_painted_element(element, color, random_color_brightness=1):
    from ..helpers_visualization import get_painted

    if isinstance(element, list):
        raise RuntimeError("Expected single element, not list.")

    # lower luminance of random colors to not interfere with highlights
    if isinstance(color, str) and color == "random":
        color = np.random.random(3) * random_color_brightness

    color = np.clip(color, 0, 1)

    return get_painted(element, color)


def get_distance_checker(element, number_points_distance):
    from pyShapeDetector.primitives import Primitive
    from pyShapeDetector.geometry import TriangleMesh, PointCloud

    # number_points_distance = self._settings.number_points_distance

    if isinstance(element, Primitive):
        return element

    # assert our PointCloud class instead of Open3D PointCloud class
    elif TriangleMesh.is_instance_or_open3d(element):
        pcd = element.sample_points_uniformly(number_points_distance)
        return PointCloud(pcd)

    elif PointCloud.is_instance_or_open3d(element):
        if len(element.points) > number_points_distance:
            ratio = int(len(element.points) / number_points_distance)
            pcd = element.uniform_down_sample(ratio)
        else:
            pcd = element
        return PointCloud(pcd)

    else:
        return None
