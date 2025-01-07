#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:35:16 2023

@author: ebernardes
"""
import copy
import warnings
import functools
import numpy as np
import json
from itertools import compress

from pyShapeDetector.primitives import (
    Primitive,
    PlaneBounded,
    PlaneRectangular,
)  # , Sphere, Plane, Cylinder
from pyShapeDetector.geometry import PointCloud, TriangleMesh
from pyShapeDetector import utility as util
from pyShapeDetector import methods
from datetime import datetime

format_int = lambda i, N: "0" * (len(str(N)) - len(str(i))) + str(i)


def _get_pointcloud_sizes(elements):
    sizes = [
        len(elem.points) for elem in elements if PointCloud.is_instance_or_open3d(elem)
    ]
    min_ = min(sizes)
    return min_, min([max(sizes), min_ + 1000])


def _get_shape_areas(elements):
    areas = [
        elem.surface_area
        for elem in elements
        if isinstance(elem, (Primitive, TriangleMesh))
    ]
    return (min(areas), max(areas))


def _extract_element_by_type(elements, element_type):
    mask = np.array([isinstance(elem, element_type) for elem in elements])
    elements_selected = []
    other = []
    for element in elements:
        if isinstance(element, element_type):
            elements_selected.append(element)
        else:
            other.append(element)

    return elements_selected, other


def _apply_to(element_type):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(elements, *args, **kwargs):
            elements_selected, other = _extract_element_by_type(elements, element_type)

            if len(elements_selected) == 0:
                return other

            return other + func(elements_selected, *args, **kwargs)

        return wrapper

    return decorator


def load_current_state(path):
    try:
        ground = Primitive.load(path / "ground.tar")
        shapes = [Primitive.load(f) for f in (path).glob("shape_*.tar")]
        pcds = [PointCloud.read_point_cloud(f) for f in (path).glob("pcd_*.ply")]
        with open(path / "parameters.json", "r") as json_file:
            parameters = json.load(json_file)
    except:
        warnings.warn("Could not find elements, returning zero...")
        ground = None
        shapes = []
        pcds = []
        parameters = None
    return ground, shapes + pcds, parameters


def save_current_state(path, ground, elements, parameters):
    """Quick save elements."""
    util.check_existance(path, remove_dir=True)
    pcds = [elem for elem in elements if PointCloud.is_instance_or_open3d(elem)]
    shapes = [elem for elem in elements if isinstance(elem, Primitive)]
    assert len(pcds) + len(shapes) == len(elements)

    for i, pcd in enumerate(pcds):
        pcd.write_point_cloud(path / f"pcd_{format_int(i, len(pcds))}.ply")

    for i, shape in enumerate(shapes):
        shape.save(path / f"shape_{format_int(i, len(shapes))}.tar")

    ground.save(path / "ground.tar")

    with open(path / "parameters.json", "w") as fp:
        json.dump(parameters, fp)


def get_camera_from_plane(plane, zoom=0.8):
    pcd = plane.inliers

    camera_options = {
        "lookat": pcd.midrange.tolist(),
        "up": PlaneRectangular(plane).parallel_vectors[0].tolist(),
        "front": (plane.normal).tolist(),
        "zoom": zoom,
    }

    c = np.cross(camera_options["up"], camera_options["front"])
    d = pcd.points.dot(c) / np.linalg.norm(c)
    # camera_options["dist"] = (max(d) - min(d)) * 0.5
    camera_options["dist"] = max(plane.aabb.get_extent()) / 5
    return camera_options


def extract_options(options):
    draw = options.get("draw", True)
    ask = options.get("ask", True)
    camera_options = options.get("camera_options", {})
    timer = options.get("timer", None)

    if timer is None:
        timer = Timer()

    return draw, ask, camera_options, timer


class Timer:
    def __init__(self):
        self.start_time = datetime.now()  # Initialize start time

    def elapsed_time(self):
        elapsed = datetime.now() - self.start_time
        return elapsed

    def hours_minutes_seconds(self):
        elapsed = self.elapsed_time()
        minutes = elapsed.seconds // 60
        hours = minutes // 60
        minutes = minutes % 60
        seconds = elapsed.seconds % 60

        return hours, minutes, seconds

    def print(self, text=""):
        if text == "":
            text = self.__repr__()
        else:
            text = f"{self.__repr__()} - {text}"
        print(text)

    def reset(self):
        self.start_time = datetime.now()  # Reset the start time

    def __repr__(self):
        hours, minutes, seconds = self.hours_minutes_seconds()
        hours = "0" * (2 - len(str(hours))) + str(hours)
        minutes = "0" * (2 - len(str(minutes))) + str(minutes)
        seconds = "0" * (2 - len(str(seconds))) + str(seconds)
        return f"{hours}:{minutes}:{seconds}"


def separate_ground_pcd_camera(
    pcd,
    k=15,
    threshold_angle_degrees=15,
    threshold_refit_ratio=2,
    downsample=50,
    debug=False,
    method=methods.BDSAC,
    **options,
):
    draw, ask, camera_options, timer = extract_options(options)

    # Detect plane with downsampled pointcloud
    plane_detector = method()
    plane_detector.options.downsample = downsample
    plane_detector.options.threshold_angle_degrees = threshold_angle_degrees
    plane_detector.options.threshold_refit_ratio = threshold_refit_ratio
    plane_detector.options.adaptative_threshold_k = k
    plane_detector.add(PlaneBounded)

    print(f"{timer} - Detecting ground...")
    ground, inliers, ground_metrics = plane_detector.fit(
        pcd, set_inliers=True, debug=debug
    )
    print(f"{timer} - Done, {len(inliers)} inliers found!")

    # Remove ground from full pcd
    pcd_no_ground = pcd.select_by_index(inliers, invert=True)

    print(f"{timer} - Correcting ground normal direction...")
    below_ground = (pcd_no_ground.points - ground.centroid).dot(ground.normal) < 0
    if sum(below_ground) > len(pcd.points) / 2:
        below_ground = ~below_ground
        ground._model = -ground._model
    print(f"{timer} - Normal corrected!")

    indices_below_ground = np.where(below_ground)[0]
    pcd_above_ground = pcd_no_ground.select_by_index(indices_below_ground, invert=True)
    pcd_below_ground = pcd_no_ground.select_by_index(indices_below_ground)

    # ground.set_inliers(pcd.select_by_index(inliers))
    ground.set_vertices(
        pcd.points, flatten=True, convex=True
    )  # Use inliers and outliers for boudaries, to assure nothing is "floating"

    if len(camera_options) == 0:
        camera_options = get_camera_from_plane(ground, zoom=0.8)

    if draw:
        util.draw_two_columns(
            [ground.inliers, pcd_no_ground],
            [
                ground,
                pcd_above_ground,
                util.get_painted(pcd_below_ground, color=(0, 0, 0)),
            ],
            # draw_inliers=True,
            window_name=f"Ground with {len(ground.inliers.points)} inliers, {len(pcd_no_ground.points)} remaining points.",
            **camera_options,
        )

    return ground, pcd_above_ground, pcd_below_ground, camera_options


def choose_threshold_parameters(pcds, eps, **options):
    from open3d import visualization

    draw, ask, camera_options, timer = extract_options(options)

    if not isinstance(pcds, list):
        pcds = [pcds]

    for pcd in pcds:
        if not pcd.has_curvature():
            if not pcd.has_normals():
                pcd.estimate_normals()
            pcd.estimate_curvature()

    pcd = PointCloud.fuse_pointclouds(pcds)

    curvature = pcd.curvature
    pcd = util.get_painted(pcd, (0.9, 0.9, 0.9))
    pcd.curvature = curvature

    pcd_high, pcd_low = pcd.separate_by_curvature(std_ratio=0)
    pcd_high = util.get_painted(pcd_high, (1, 0, 0))

    global data

    data = {
        "pcd_low": pcd_low.as_open3d,
        "pcd_high": pcd_high.as_open3d,
        "pcd_close": PointCloud().as_open3d,
        "eps": eps,
        "std_ratio": 0,
        "distance_threshold_ratio": 1,
        "std": np.std(pcd.curvature),
        "mean": np.mean(pcd.curvature),
    }

    def update(vis):
        global data

        threshold = data["mean"] + data["std"] * data["std_ratio"]

        vis.remove_geometry(data["pcd_low"], reset_bounding_box=False)
        vis.remove_geometry(data["pcd_high"], reset_bounding_box=False)
        vis.remove_geometry(data["pcd_close"], reset_bounding_box=False)

        indices = np.where(pcd.curvature < threshold)[0]
        pcd_low = pcd.select_by_index(indices)
        pcd_high = pcd.select_by_index(indices, invert=True)

        if data["distance_threshold_ratio"] > 0:
            distance_threshold = data["distance_threshold_ratio"] * eps

            is_close = (
                pcd_low.compute_point_cloud_distance(pcd_high) <= distance_threshold
            )

            new_indices = np.where(is_close)[0]
            pcd_close = pcd_low.select_by_index(new_indices)
            pcd_low = pcd_low.select_by_index(new_indices, invert=True)
        else:
            pcd_close = PointCloud()

        data["pcd_low"] = pcd_low.as_open3d
        data["pcd_high"] = util.get_painted(pcd_high, (1, 0, 0)).as_open3d
        data["pcd_close"] = util.get_painted(pcd_close, (0, 0, 1)).as_open3d

        vis.add_geometry(data["pcd_low"], reset_bounding_box=False)
        vis.add_geometry(data["pcd_high"], reset_bounding_box=False)
        vis.add_geometry(data["pcd_close"], reset_bounding_box=False)

        print(
            f"distance_threshold_ratio = {data['distance_threshold_ratio']}, std_ratio = {data['std_ratio']}"
        )

    def std_decrease(vis):
        global data
        data["std_ratio"] -= 0.1
        update(vis)

    def std_increase(vis):
        global data
        data["std_ratio"] += 0.1
        update(vis)

    def distance_threshold_decrease(vis):
        global data
        data["distance_threshold_ratio"] = max(
            data["distance_threshold_ratio"] - 0.1, 0
        )
        update(vis)

    def distance_threshold_increase(vis):
        global data
        data["distance_threshold_ratio"] += 0.1
        update(vis)

    window_name = "(W) std + | (S) std - | (A) dist - | (D) dist +"

    key_to_callback = {}
    key_to_callback[ord("W")] = std_increase
    key_to_callback[ord("S")] = std_decrease
    key_to_callback[ord("A")] = distance_threshold_decrease
    key_to_callback[ord("D")] = distance_threshold_increase

    visualization.draw_geometries_with_key_callbacks(
        [data["pcd_low"], data["pcd_high"], data["pcd_close"]],
        # data['elements_painted'],
        key_to_callback,
        window_name=window_name,
    )

    return data
