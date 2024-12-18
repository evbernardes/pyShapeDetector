#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:16:15 2024

@author: ebernardes
"""
import warnings
import numpy as np
from pyShapeDetector.geometry import PointCloud
from pyShapeDetector.primitives import (
    PlaneBounded,
    PlaneTriangulated,
    PlaneRectangular,
    Sphere,
    Cylinder,
    Cone,
)
from pyShapeDetector.methods import list_methods_RANSAC
from pyShapeDetector.utility import MultiDetector
from helpers import _apply_to


MENU_NAME = "Create shapes"

dict_methods = {method.__name__: method for method in list_methods_RANSAC}
extensions = []

extensions.append(
    {
        "name": "From center and radius",
        "function": lambda center, radius: Sphere.from_center_radius(center, radius),
        "menu": MENU_NAME + "/Create Sphere...",
        "inputs": None,
        "parameters": {
            "radius": {
                "type": float,
                "default": 1,
                "limits": (0, 5),
            },
            "center": {"type": np.ndarray, "default": [0.0, 0.0, 0.0]},
        },
    }
)

extensions.append(
    {
        "name": "From center, half vector and radius",
        "function": lambda center,
        vector,
        radius: Cylinder.from_center_half_vector_radius(center, vector, radius),
        "menu": MENU_NAME + "/Create Cylinder...",
        "inputs": None,
        "parameters": {
            "radius": {
                "type": float,
                "default": 1,
                "limits": (0, 5),
            },
            "center": {
                "type": np.ndarray,
                "default": [0.0, 0.0, 0.0],
            },
            "vector": {
                "type": np.ndarray,
                "default": [0.0, 0.0, 1.0],
            },
        },
    }
)

extensions.append(
    {
        "name": "From base, top and radius",
        "function": lambda base, top, radius: Cylinder.from_base_top_radius(
            base, top, radius
        ),
        "menu": MENU_NAME + "/Create Cylinder...",
        "inputs": None,
        "parameters": {
            "radius": {
                "type": float,
                "default": 1,
                "limits": (0, 5),
            },
            "base": {
                "type": np.ndarray,
                "default": [0.0, 0.0, 0.0],
            },
            "top": {
                "type": np.ndarray,
                "default": [0.0, 0.0, 1.0],
            },
        },
    }
)

extensions.append(
    {
        "name": "From base, vector and radius",
        "function": lambda base, vector, radius: Cylinder.from_base_vector_radius(
            base, vector, radius
        ),
        "menu": MENU_NAME + "/Create Cylinder...",
        "inputs": None,
        "parameters": {
            "radius": {
                "type": float,
                "default": 1,
                "limits": (0, 5),
            },
            "base": {
                "type": np.ndarray,
                "default": [0.0, 0.0, 0.0],
            },
            "vector": {
                "type": np.ndarray,
                "default": [0.0, 0.0, 1.0],
            },
        },
    }
)


extensions.append(
    {
        "name": "Create Cone",
        "function": lambda appex,
        vector,
        half_angle_degrees: Cone.from_appex_vector_half_angle(
            appex, vector, np.deg2rad(half_angle_degrees)
        ),
        "menu": MENU_NAME,
        "inputs": None,
        "parameters": {
            "half_angle_degrees": {
                "type": float,
                "default": 10,
                "limits": (0.01, 89.5),
            },
            "appex": {
                "type": np.ndarray,
                "default": [0.0, 0.0, 0.0],
            },
            "vector": {
                "type": np.ndarray,
                "default": [0.0, 0.0, 1.0],
            },
        },
    }
)


@_apply_to(PointCloud)
def detect_shapes(
    pcds_input,
    PointCloud_density,
    method,
    shapes_per_cluster,
    inliers_min,
    use_adaptative_threshold,
    threshold_distance_ratio,
    adaptative_threshold_k,
    max_sample_distance_ratio,
    threshold_angle_degrees,
    threshold_refit_ratio,
    num_iterations,
    num_samples,
    downsample,
    compare_metric,
    metric_min,
    debug,
    detect_PlaneBounded,
    detect_Sphere,
    detect_Cylinder,
    detect_Cone,
    detect_PlaneTriangulated,
    detect_PlaneRectangular,
):
    detector = dict_methods[method]()
    detector.options.inliers_min = inliers_min

    detector.options.max_sample_distance = (
        max_sample_distance_ratio * PointCloud_density
    )
    detector.options.threshold_angle_degrees = threshold_angle_degrees
    detector.options.threshold_refit_ratio = threshold_refit_ratio
    detector.options.num_iterations = num_iterations
    detector.options.num_samples = num_samples
    detector.options.downsample = downsample
    if use_adaptative_threshold:
        detector.options.adaptative_threshold_k = adaptative_threshold_k
    else:
        detector.options.threshold_distance = (
            threshold_distance_ratio * PointCloud_density
        )

    if detect_PlaneBounded:
        detector.add(PlaneBounded)
    if detect_Sphere:
        detector.add(Sphere)
    if detect_Cylinder:
        detector.add(Cylinder)
    if detect_Cone:
        detector.add(Cone)
    if detect_PlaneTriangulated:
        detector.add(PlaneTriangulated)
    if detect_PlaneRectangular:
        detector.add(PlaneRectangular)

    if debug:
        print("Calling 'detect_shapes_with_BDSAC' with following parameters: ")
        print(f"PointCloud_density = {PointCloud_density}")
        print(detector.options)

    shape_detector = MultiDetector(
        detector,
        pcds_input,
        debug=False,
        normals_reestimate=False,
        points_min=10,
        shapes_per_cluster=shapes_per_cluster,
        compare_metric=compare_metric,
        metric_min=metric_min,
        fuse_shapes=False,
    )

    shapes = shape_detector.shapes
    for shape in shapes:
        shape.color = shape.inliers.colors.mean(axis=0)

    return shape_detector.shapes + shape_detector.pcds_rest


extensions.append(
    {
        "name": "Detect shapes with RANSAC-based method",
        "function": detect_shapes,
        "menu": MENU_NAME,
        "parameters": {
            "method": {
                "type": list,
                "options": list(dict_methods.keys()),
                "default": "BDSAC",
            },
            "PointCloud_density": {"type": "preference"},
            "shapes_per_cluster": {"type": int, "default": 1, "limits": (1, 50)},
            "inliers_min": {"type": int, "default": 100, "limits": (1, 1000)},
            "use_adaptative_threshold": {
                "name": "test",
                "type": bool,
                "subpanel": "RANSAC Options",
            },
            "threshold_distance_ratio": {
                "type": float,
                "default": 1,
                "limits": (0.01, 100),
                "subpanel": "RANSAC Options",
            },
            "adaptative_threshold_k": {
                "type": int,
                "default": 1,
                "limits": (1, 50),
                "subpanel": "RANSAC Options",
            },
            "max_sample_distance_ratio": {
                "type": float,
                "default": 1,
                "limits": (0.01, 100),
                "subpanel": "RANSAC Options",
            },
            "threshold_angle_degrees": {
                "type": float,
                "default": 20,
                "limits": (0, 180),
                "subpanel": "RANSAC Options",
            },
            "threshold_refit_ratio": {
                "type": float,
                "default": 5,
                "limits": (1, 10),
                "subpanel": "RANSAC Options",
            },
            "num_iterations": {
                "type": int,
                "default": 20,
                "limits": (1, 100),
                "subpanel": "RANSAC Options",
            },
            "num_samples": {
                "type": int,
                "default": 10,
                "limits": (1, 100),
                "subpanel": "RANSAC Options",
            },
            "downsample": {
                "type": int,
                "default": 1,
                "limits": (1, 1000),
                "subpanel": "RANSAC Options",
            },
            "compare_metric": {
                "type": list,
                "options": ["fitness", "weight"],
                "default": "fitness",
                "subpanel": "RANSAC Options",
            },
            "metric_min": {
                "type": float,
                "default": 0.1,
                "subpanel": "RANSAC Options",
            },
            "detect_PlaneBounded": {
                "type": bool,
                "default": True,
                "subpanel": "Primitives",
            },
            "detect_Sphere": {"type": bool, "subpanel": "Primitives"},
            "detect_Cylinder": {"type": bool, "subpanel": "Primitives"},
            "detect_Cone": {"type": bool, "subpanel": "Primitives"},
            "detect_PlaneTriangulated": {"type": bool, "subpanel": "Primitives"},
            "detect_PlaneRectangular": {"type": bool, "subpanel": "Primitives"},
            "debug": {"type": "preference"},
        },
    }
)
