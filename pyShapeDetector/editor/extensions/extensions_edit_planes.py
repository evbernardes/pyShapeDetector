#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:01:29 2024

@author: ebernardes
"""
import warnings
import numpy as np
from pyShapeDetector.primitives import Plane, PlaneBounded
from .helpers import _apply_to

MENU_NAME = "Edit Planes"

extensions = []


@_apply_to(PlaneBounded)
def _get_concave_alpha_shapes_with_alpha(
    shapes,
    alpha,
    angle_colinear_degrees,
    contract_boundary,
    detect_holes,
    add_inliers,
    min_inliers,
    min_area,
    downsample_k,
):
    convex_shapes = [shape for shape in shapes if shape.is_convex]
    concave_shapes = [shape for shape in shapes if not shape.is_convex]

    shapes = get_convex(concave_shapes) + convex_shapes

    extra_options = {
        "detect_holes": detect_holes,
        "add_inliers": add_inliers,
        "angle_colinear": np.deg2rad(angle_colinear_degrees),
        "min_point_dist": (1 / alpha) * 2,
        "contract_boundary": contract_boundary,
        "min_inliers": min_inliers,
        "min_area": min_area,  # 0.0035,
    }

    if downsample_k == 1:
        downsample_k = None

    planes_bounded_alpha_groups = []
    for s in shapes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            plane_triangulated = s.get_triangulated_plane_from_alpha_shape(
                alpha, downsample_k=downsample_k
            )
            plane_triangulated.color = s.color
            plane_triangulated._inliers = s._inliers

            planes_bounded_alpha_groups.append(
                plane_triangulated.get_bounded_planes_from_boundaries(**extra_options)
            )

    concave_planes = [p for group in planes_bounded_alpha_groups for p in group]
    print(f"Number of planes: {len(concave_planes)}")
    return concave_planes


extensions.append(
    {
        "name": "With alpha",
        "function": _get_concave_alpha_shapes_with_alpha,
        "menu": MENU_NAME + "/Get concave alpha shapes...",
        "parameters": {
            "alpha": {
                "type": float,
                "default": 1,
                "limits": (0.001, 1000),
            },
            "angle_colinear_degrees": {
                "type": float,
                "default": 15,
                "limits": (0, 180),
            },
            "contract_boundary": {"type": bool},
            "detect_holes": {"type": bool, "default": True, "subpanel": "Extra"},
            "add_inliers": {"type": bool, "default": True, "subpanel": "Extra"},
            "min_inliers": {
                "type": int,
                "default": 1,
                "limits": (1, 1000),
                "subpanel": "Extra",
            },
            "min_area": {
                "type": float,
                "default": 0.0035,
                "limits": (0.0001, 100),
                "subpanel": "Extra",
            },
            "downsample_k": {
                "type": int,
                "default": 1,
                "limits": (1, 50),
            },
        },
    }
)


@_apply_to(PlaneBounded)
def _get_concave_alpha_shapes_with_radius_ratio(
    shapes,
    PointCloud_density,
    radius_ratio,
    angle_colinear_degrees,
    contract_boundary,
    detect_holes,
    add_inliers,
    min_inliers,
    min_area,
    downsample_k,
):
    alpha_inv = PointCloud_density * radius_ratio
    alpha = 1 / alpha_inv

    return _get_concave_alpha_shapes_with_alpha(
        shapes,
        alpha,
        angle_colinear_degrees,
        contract_boundary,
        detect_holes,
        add_inliers,
        min_inliers,
        min_area,
        downsample_k,
    )


extensions.append(
    {
        "name": "With radius ratio",
        "function": _get_concave_alpha_shapes_with_radius_ratio,
        "menu": MENU_NAME + "/Get concave alpha shapes...",
        "parameters": {
            "PointCloud_density": {"type": "preference"},
            "radius_ratio": {
                "type": float,
                "default": 2,
                "limits": (0, 100),
            },
            "angle_colinear_degrees": {
                "type": float,
                "default": 15,
                "limits": (0, 180),
            },
            "contract_boundary": {"type": bool},
            "detect_holes": {"type": bool, "default": True, "subpanel": "Extra"},
            "add_inliers": {"type": bool, "default": True, "subpanel": "Extra"},
            "min_inliers": {
                "type": int,
                "default": 1,
                "limits": (1, 1000),
                "subpanel": "Extra",
            },
            "min_area": {
                "type": float,
                "default": 0.0035,
                "limits": (0.0001, 100),
                "subpanel": "Extra",
            },
            "downsample_k": {
                "type": int,
                "default": 1,
                "limits": (1, 50),
            },
        },
    }
)


@_apply_to(PlaneBounded)
def _get_concave_alpha_shapes_with_area_percentage(
    shapes,
    area_percentage,
    angle_colinear_degrees,
    contract_boundary,
    detect_holes,
    add_inliers,
    min_inliers,
    min_area,
    downsample_k,
):
    convex_shapes = [shape for shape in shapes if shape.is_convex]
    concave_shapes = [shape for shape in shapes if not shape.is_convex]

    shapes = get_convex(concave_shapes) + convex_shapes

    output_shapes = []
    for shape in shapes:
        area = shape.surface_area
        alpha_inv = area * area_percentage / 100
        alpha = 1 / alpha_inv

        output_shapes += _get_concave_alpha_shapes_with_alpha(
            [shape],
            alpha,
            angle_colinear_degrees,
            contract_boundary,
            detect_holes,
            add_inliers,
            min_inliers,
            min_area,
            downsample_k,
        )
    return output_shapes


extensions.append(
    {
        "name": "With Area Percentage",
        "function": _get_concave_alpha_shapes_with_area_percentage,
        "menu": MENU_NAME + "/Get concave alpha shapes...",
        "parameters": {
            "area_percentage": {
                "type": float,
                "default": 10,
                "limits": (0.001, 15),
            },
            "angle_colinear_degrees": {
                "type": float,
                "default": 15,
                "limits": (0, 180),
            },
            "contract_boundary": {"type": bool},
            "detect_holes": {"type": bool, "default": True, "subpanel": "Extra"},
            "add_inliers": {"type": bool, "default": True, "subpanel": "Extra"},
            "min_inliers": {
                "type": int,
                "default": 1,
                "limits": (1, 1000),
                "subpanel": "Extra",
            },
            "min_area": {
                "type": float,
                "default": 0.0035,
                "limits": (0.0001, 100),
                "subpanel": "Extra",
            },
            "downsample_k": {
                "type": int,
                "default": 1,
                "limits": (1, 50),
            },
        },
    }
)


@_apply_to(PlaneBounded)
def get_convex(shapes):
    convex_shapes = []
    for shape in shapes:
        convex_shape = shape.get_convex()
        convex_shape._inliers = shape._inliers
        convex_shapes.append(convex_shape)
    return convex_shapes


extensions.append(
    {
        "name": "Revert concave planes to convex",
        "function": get_convex,
        "menu": MENU_NAME,
    }
)


@_apply_to(PlaneBounded)
def split_planes(selected_planes):
    intersections = PlaneBounded.get_plane_intersections(selected_planes)
    split_planes = []
    for (i, j), line in intersections.items():
        plane_i = selected_planes[i]
        plane_j = selected_planes[j]

        line_i = line.get_fitted_to_points(plane_i.vertices)
        line_j = line.get_fitted_to_points(plane_j.vertices)
        line = line_i.get_segment_union(line_j)

        split_planes += plane_i.split(line) + plane_j.split(line)

    split_planes = [
        p for p in split_planes if (p is not None and p.surface_area > 1e-8)
    ]

    return split_planes


extensions.append(
    {
        "name": "Split planes at each intersection",
        "function": split_planes,
        "menu": MENU_NAME,
    }
)


@_apply_to(PlaneBounded)
def glue_planes(selected_planes, split_planes, fit_mode, to_biggest):
    intersections = PlaneBounded.get_plane_intersections(selected_planes)
    if to_biggest:
        areas = [p.surface_area for p in selected_planes]
        i = np.argmax(areas)
        intersections = {key: value for key, value in intersections.items() if i in key}
    planes_glued = [plane.copy() for plane in selected_planes]
    PlaneBounded.glue_planes_with_intersections(
        planes_glued, intersections, split=split_planes, fit_mode=fit_mode, eps=1e-3
    )
    return planes_glued


extensions.append(
    {
        "name": "Glue planes at each intersection",
        "function": glue_planes,
        "menu": MENU_NAME,
        "parameters": {
            "fit_mode": {
                "type": list,
                "options": [
                    "direct",
                    "separated",
                    "segment_union",
                    "segment_intersection",
                ],
                "default": "segment_intersection",
            },
            "split_planes": {"type": bool},
            "to_biggest": {"type": bool},
        },
    }
)


@_apply_to(Plane)
def remove_holes_from_planes(planes_input):
    planes = [plane.copy() for plane in planes_input]
    for plane in planes:
        while len(plane.holes) > 0:
            plane.remove_hole(0)
    return planes


extensions.append(
    {
        "function": remove_holes_from_planes,
        "menu": MENU_NAME,
    }
)


@_apply_to(PlaneBounded)
def simplify_vertices(planes_input, angle_colinear_degrees, min_point_dist):
    new_planes = [p.copy() for p in planes_input]
    for plane in new_planes:
        plane.simplify_vertices(
            np.deg2rad(angle_colinear_degrees),
            min_point_dist,
        )

    return new_planes


def _get_line_lenghts(planes):
    lenghts = [line.length for plane in planes for line in plane.vertices_lines]
    return (min(lenghts), 1000 * min(lenghts))


extensions.append(
    {
        "function": simplify_vertices,
        "menu": MENU_NAME,
        "parameters": {
            "angle_colinear_degrees": {
                "type": float,
                "default": 180,
                "limits": (0, 180),
            },
            "min_point_dist": {
                "type": float,
                "limits": (0, 10),
                "limit_setter": _get_line_lenghts,
            },
        },
    }
)


@_apply_to(PlaneBounded)
def smoothen_boundaries_with_fft(planes_input, number_of_frequencies_to_keep):
    new_planes = [p.copy() for p in planes_input]
    for plane in new_planes:
        plane.smooth_boundary_with_fft(number_of_frequencies_to_keep)

    return new_planes


extensions.append(
    {
        "name": "Smoothen boundaries with FFT",
        "function": smoothen_boundaries_with_fft,
        "menu": MENU_NAME,
        "parameters": {
            "number_of_frequencies_to_keep": {
                "type": int,
                "default": 3,
                "limits": (1, 1000),
            },
        },
    }
)
