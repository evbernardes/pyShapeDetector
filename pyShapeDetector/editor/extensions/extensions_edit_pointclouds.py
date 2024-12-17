#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:12:48 2024

@author: ebernardes
"""
import warnings
import numpy as np
from typing import Union
from multiprocessing import cpu_count
from pyShapeDetector.utility import get_inputs, select_function_with_gui
from pyShapeDetector.geometry import PointCloud
from pyShapeDetector.primitives import Primitive, Plane
from helpers import (
    _apply_to,
    _extract_element_by_type,
)

extensions = []

MENU_NAME = "Edit Pointclouds"


@_apply_to(PointCloud)
def uniform_downsample(pcds_input, every_k_points):
    return [pcd.uniform_downsample(every_k_points) for pcd in pcds_input]


extensions.append(
    {
        "function": uniform_downsample,
        "menu": MENU_NAME,
        "parameters": {
            "every_k_points": {
                "type": int,
                "default": 5,
                "limits": (1, 500),
            },
        },
    }
)


@_apply_to(PointCloud)
def segment_distances_with_DBSCAN(
    pcds_input: list[PointCloud], PointCloud_density, dbscan_ratio, min_points, debug
):
    if debug:
        print("Calling 'segment_dbscan' with following parameters: ")
        print(f"eps: {PointCloud_density}")
        print(f"dbscan_distance_ratio: {dbscan_ratio}")
        print(f"min_points: {min_points}")
        print(f"debug: {debug}")
        print(f"{len(pcds_input)} pcds input. Number of points of each: ")
        print([len(pcd.points) for pcd in pcds_input])

    pcds_output = []
    for pcd in pcds_input:
        pcds_output += pcd.segment_dbscan(PointCloud_density * dbscan_ratio)

    if debug:
        print(f"{len(pcds_output)} pcds found...")

    pcds_output = [pcd for pcd in pcds_output if len(pcd.points) >= min_points]

    if debug:
        print(f"{len(pcds_output)} pcds with more than {min_points}.")

    if len(pcds_output) == 0:
        warnings.warn("Point pointclouds deleted, returning input")
        return pcds_input

    return pcds_output


extensions.append(
    {
        "name": "Segment distances (DBSCAN)",
        "function": segment_distances_with_DBSCAN,
        "menu": MENU_NAME,
        "parameters": {
            "PointCloud_density": {"type": "preference"},
            "dbscan_ratio": {
                "type": float,
                "default": 3,
                "limits": (0.001, 50),
            },
            "min_points": {
                "type": int,
                "default": 5,
                "limits": (1, 500),
            },
            "debug": {"type": "preference"},
        },
    }
)


@_apply_to(PointCloud)
def segment_colors_with_DBSCAN(pcds_input: list[PointCloud], num_clusters):
    if num_clusters < 2:
        warnings.warn("Min number of clusters should be 2, got {num_clusters}.")
        return pcds_input

    pcds_output = []
    for pcd in pcds_input:
        pcds_output += pcd.segment_kmeans_colors(num_clusters)

    return pcds_output


extensions.append(
    {
        "name": "Segment colors (DBSCAN)",
        "function": segment_colors_with_DBSCAN,
        "menu": MENU_NAME,
        "parameters": {
            "num_clusters": {
                "type": int,
                "default": 5,
                "limits": (2, 50),
            },
        },
    }
)


@_apply_to(PointCloud)
def segment_region_growing(
    pcds_input: list[PointCloud],
    PointCloud_density,
    dbscan_ratio,
    threshold_angle_degrees,
    cores,
):
    pcds_output = []
    for pcd in pcds_input:
        pcds_output += pcd.segment_with_region_growing(
            mode="radius",
            radius=dbscan_ratio * PointCloud_density,
            k_retest=15,
            threshold_angle=np.radians(threshold_angle_degrees),
            cores=cores,
            min_points=10,
            seeds_max=1000,
            debug=True,
        )

    return pcds_output


extensions.append(
    {
        "name": "Segment with Region Growing",
        "function": segment_region_growing,
        "menu": MENU_NAME,
        "parameters": {
            "PointCloud_density": {"type": "preference"},
            "dbscan_ratio": {
                "type": float,
                "default": 3,
                "limits": (0.001, 50),
            },
            "threshold_angle_degrees": {
                "type": float,
                "default": 10,
                "limits": (0, 180),
            },
            "cores": {
                "type": int,
                "default": 5,
                "limits": (1, cpu_count()),
            },
        },
    }
)


@_apply_to(PointCloud)
def segment_curvature_with_DBSCAN(
    pcds_input: list[PointCloud],
    PointCloud_density,
    dbscan_ratio,
    std_ratio,
    min_points,
):
    pcds_output = []
    for pcd in pcds_input:
        if not pcd.has_normals():
            pcd.estimate_normals()

        pcds_output += pcd.segment_with_curvature_threshold(
            std_ratio, PointCloud_density * dbscan_ratio, min_points
        )
    return pcds_output


extensions.append(
    {
        "name": "Segment curvature (DBSCAN)",
        "function": segment_curvature_with_DBSCAN,
        "menu": MENU_NAME,
        "parameters": {
            "PointCloud_density": {"type": "preference"},
            "dbscan_ratio": {
                "type": float,
                "default": 3,
                "limits": (0.001, 50),
            },
            "std_ratio": {
                "type": float,
                "default": -0.1,
                "limits": (-1.0, 1.0),
            },
            "min_points": {
                "type": int,
                "default": 5,
                "limits": (1, 500),
            },
        },
    }
)


@_apply_to(PointCloud)
def separate_edges_with_curvature(
    pcds_input: list[PointCloud], PointCloud_density, dbscan_ratio, std_ratio
):
    output = []
    for pcd in pcds_input:
        pcd_low, pcd_high = pcd.separate_by_curvature(
            std_ratio, PointCloud_density * dbscan_ratio
        )
        output += [pcd_low, pcd_high]
    return output


extensions.append(
    {
        "name": "Separate edges with curvature DBSCAN",
        "function": separate_edges_with_curvature,
        "menu": MENU_NAME,
        "parameters": {
            "PointCloud_density": {"type": "preference"},
            "dbscan_ratio": {
                "type": float,
                "default": 3,
                "limits": (0.001, 50),
            },
            "std_ratio": {
                "type": float,
                "default": -0.1,
                "limits": (-1.0, 1.0),
            },
        },
    }
)


@_apply_to(PointCloud)
def split_along_axis(pcds_input: list[PointCloud], axis, number_of_boxes):
    dim = "xyz".index(axis)

    return [
        pcd
        for input_pcd in pcds_input
        for pcd in input_pcd.split(num_boxes=number_of_boxes, dim=dim)
    ]


extensions.append(
    {
        "function": split_along_axis,
        "menu": MENU_NAME,
        "hotkey": "S",
        "parameters": {
            "axis": {
                "type": list,
                "options": ["x", "y", "z"],
                "default": "z",
            },
            "number_of_boxes": {
                "type": int,
                "default": 2,
                "limits": (2, 20),
            },
        },
    }
)


@_apply_to(PointCloud)
def split_pcd_in_half_along_axis(pcds_input: list[PointCloud], axis, resolution):
    dim = "xyz".index(axis)

    return [
        pcd
        for input_pcd in pcds_input
        for pcd in input_pcd.split_in_half(dim=dim, resolution=resolution)
    ]


extensions.append(
    {
        "function": split_pcd_in_half_along_axis,
        "menu": MENU_NAME,
        "parameters": {
            "axis": {
                "type": list,
                "options": ["x", "y", "z"],
                "default": "z",
            },
            "resolution": {
                "type": int,
                "default": 5,
                "limits": (2, 20),
            },
        },
    }
)


@_apply_to(PointCloud)
def uniform_down_sample(pcds_input: list[PointCloud], every_k_points):
    return [pcd.uniform_down_sample(every_k_points) for pcd in pcds_input]


extensions.append(
    {
        "function": uniform_down_sample,
        "menu": MENU_NAME,
        "parameters": {
            "every_k_points": {
                "type": int,
                "default": 1,
                "limits": (1, 100),
            },
        },
    }
)


@_apply_to(PointCloud)
def box_planes_from_bounding_box(
    pcds_input: list[PointCloud], keep_empty_planes, keep_as_rectangular
):
    pcd_full = PointCloud.fuse_pointclouds(pcds_input)
    planes = pcd_full.obb.as_planes()

    distances = np.asarray([plane.get_distances(pcd_full.points) for plane in planes])
    idx = np.argmin(distances, axis=0)

    for i, plane in enumerate(planes):
        pcd = pcd_full.select_by_index(np.where(idx == i)[0])
        plane.color = pcd.colors.mean(axis=0)
        plane.set_inliers(pcd)

    if not keep_as_rectangular:
        planes = [plane for plane in planes if len(plane.inliers.points) > 3]

        for plane in planes:
            plane.set_vertices(plane.inliers.points, convex=True)

    if keep_empty_planes:
        return planes
    else:
        return [plane for plane in planes if len(plane.inliers.points) > 0]


extensions.append(
    {
        "function": box_planes_from_bounding_box,
        "menu": MENU_NAME,
        "parameters": {
            "keep_empty_planes": {
                "type": bool,
                "default": True,
            },
            "keep_as_rectangular": {
                "type": bool,
                "default": True,
            },
        },
    }
)


def separate_pointclouds_with_planes(elements: list[Union[Plane, PointCloud]]):
    planes, other = _extract_element_by_type(elements, Plane)
    pcds, other = _extract_element_by_type(other, PointCloud)

    new_pcds = []

    for plane in planes:
        for pcd in pcds:
            indices = np.where(plane.get_signed_distances(pcd.points) > 0)[0]

            pcd1 = pcd.select_by_index(indices)
            pcd2 = pcd.select_by_index(indices, invert=True)
            if len(pcd1.points) > 0:
                new_pcds.append(pcd1)
            if len(pcd2.points) > 0:
                new_pcds.append(pcd2)
    return new_pcds + planes + other


extensions.append({"function": separate_pointclouds_with_planes, "menu": MENU_NAME})


def flatten_pointclouds_to_shape(elements: list[Union[Plane, PointCloud]]):
    shapes, other = _extract_element_by_type(elements, Primitive)
    pcds, other = _extract_element_by_type(other, PointCloud)

    if len(shapes) != 1:
        raise ValueError(f"Expected 1 primitive, got {len(shapes)}.")

    shape = shapes[0].copy()

    new_pcds = []
    for pcd in pcds:
        new_pcds.append(shape.flatten_PointCloud(pcd))

    return new_pcds + [shape] + other


extensions.append({"function": flatten_pointclouds_to_shape, "menu": MENU_NAME})
