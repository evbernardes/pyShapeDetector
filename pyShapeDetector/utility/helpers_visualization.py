#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:59:02 2024

@author: ebernardes
"""
import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np
from open3d import visualization
from pyShapeDetector.geometry import PointCloud
# from pyShapeDetector.primitives import Primitive, Line


def get_painted(elements, color="random"):
    """Get painted copy of each pointcloud/mesh/shape.

    If color is not input, then

    Parameters
    ----------
    elements : list of geomery elements
        Elements to be painted
    color : array_like, 'random' or cmap name
        If color is an array_like, it will be used to define the color of
        everything. If 'random', colors will be random. If anything else, will
        be assumed to be the name of a color map. See: matplotlib.pyplot.get_cmap

    Returns
    -------
    list
    """
    from pyShapeDetector.primitives import Primitive

    elements = copy.deepcopy(elements)

    if not (is_list := isinstance(elements, list)):
        elements = [elements]

    if isinstance(color, str) or color is None:
        if color == "random":
            colors = np.random.random((len(elements), 3))
        else:
            color_map = plt.get_cmap(color)
            colors = [color_map(i)[:3] for i in range(len(elements))]

    else:
        colors = [color] * len(elements)

    for color, element in zip(colors, elements):
        try:
            element.paint_uniform_color(color)

        except AttributeError:
            try:
                element.color = color
            except AttributeError:
                pass

        color = np.random.random(3)
        if isinstance(element, Primitive):
            element.inliers.paint_uniform_color(color)

    if is_list:
        return elements
    else:
        return elements[0]


# def paint_random(elements, paint_inliers=False, return_copy=False):
#     """ Paint each pointcloud/mesh with a different random color.

#     Parameters
#     ----------
#     elements : list of geomery elements
#         Elements to be painted
#     """

#     from pyShapeDetector.primitives import Primitive

#     if return_copy:
#         elements = copy.deepcopy(elements)

#     if not isinstance(elements, list):
#         elements = [elements]

#     for element in elements:
#         color = np.random.random(3)
#         if isinstance(element, Primitive):
#             element._color = color
#             if paint_inliers:
#                 element._inlier_colors[:] = color
#         else:
#             element.paint_uniform_color(color)

#     if return_copy:
#         return elements

# def _treat_up_normal(camera_options):

#     normal = camera_options.pop('normal', None)
#     up = camera_options.get('up', None)

#     if normal is not None and  up is not None:
#             raise ValueError("Cannot enter both 'up' and 'normal'")

#     elif normal is not None:
#         x = np.cross(np.random.random(3), normal)
#         x /= np.linalg.norm(x)
#         # camera_options['up'] = np.cross(normal, x)
#         camera_options['up'] = x

#     elif 'up' in camera_options:
#         camera_options['up'] = up


def draw_geometries(elements, **camera_options):
    from pyShapeDetector.primitives import Primitive, Line, Plane, PlaneBounded

    # _treat_up_normal(camera_options)
    _ = camera_options.pop("dist", None)

    try:
        draw_inliers = camera_options.pop("draw_inliers")
    except KeyError:
        draw_inliers = False

    try:
        draw_boundary_lines = camera_options.pop("draw_boundary_lines")
    except KeyError:
        draw_boundary_lines = False

    try:
        draw_planes = camera_options.pop("draw_planes")
    except KeyError:
        draw_planes = True

    pcds = []
    geometries = []
    lines = []
    boundary_lines = []
    hole_boundary_lines = []

    if not isinstance(elements, (list, tuple)):
        elements = [elements]

    for element in elements:
        if hasattr(element, "as_open3d"):
            geometries.append(element.as_open3d)
        elif isinstance(element, Line):
            lines.append(element)
        elif isinstance(element, Plane):
            if draw_planes:
                geometries.append(element.mesh.as_open3d)
        elif isinstance(element, Primitive):
            geometries.append(element.mesh.as_open3d)
        elif isinstance(element, np.ndarray):
            if len(element.shape) != 2 or element.shape[1] != 3:
                raise ValueError(
                    "3D arrays are interpreted as PointClouds, "
                    "but they need to have a shape of (N, 3), got "
                    f"{element.shape}."
                )
            pcd = PointCloud.from_points_normals_colors(element)
            geometries.append(pcd.as_open3d)
        else:
            geometries.append(element)

        if draw_inliers and isinstance(element, Primitive):
            pcds.append(element.inliers.as_open3d)

        if draw_boundary_lines and isinstance(element, PlaneBounded):
            boundary_LineSet = element.bound_LineSet
            boundary_LineSet.paint_uniform_color((1, 0, 0))
            boundary_lines.append(boundary_LineSet)

        if draw_boundary_lines and isinstance(element, Plane):
            for hole in element.holes:
                hole_boundary_LineSet = hole.bound_LineSet
                hole_boundary_LineSet.paint_uniform_color((0, 0, 1))
                hole_boundary_lines.append(hole_boundary_LineSet)

    if len(lines) > 0:
        geometries.append(Line.get_LineSet_from_list(lines))

    if draw_inliers:
        geometries += pcds

    if "mesh_show_back_face" not in camera_options:
        camera_options["mesh_show_back_face"] = True

    if draw_boundary_lines:
        geometries += boundary_lines + hole_boundary_lines

    visualization.draw_geometries(geometries, **camera_options)


def draw_two_columns(objs_left, objs_right, **camera_options):
    # _treat_up_normal(camera_options)
    lookat = camera_options.pop("lookat", None)
    up = camera_options.pop("up", None)
    front = camera_options.pop("front", None)
    zoom = camera_options.pop("zoom", None)
    dist = camera_options.pop("dist", 5)

    has_options = not any(v is None for v in [lookat, up, front, zoom])

    if not isinstance(objs_left, list):
        objs_left = [objs_left]

    if not isinstance(objs_right, list):
        objs_right = [objs_right]

    # precalculating meshes just in case
    for elem in objs_left + objs_right:
        try:
            elem.mesh
        except AttributeError:
            pass

    objs_left = copy.deepcopy(objs_left)
    objs_right = copy.deepcopy(objs_right)

    if has_options:
        translate = 0.5 * dist * np.cross(up, front)
    else:
        translate = np.array([0, 0.5 * dist, 0])

    for i in range(len(objs_left)):
        objs_left[i].translate(-translate)
    for i in range(len(objs_right)):
        objs_right[i].translate(translate)

    if not has_options:
        draw_geometries(objs_right + objs_left, **camera_options)
    else:
        draw_geometries(
            objs_right + objs_left,
            lookat=lookat,
            up=up,
            front=front,
            zoom=zoom,
            **camera_options,
        )


def select_manually(
    elements, fixed_elements=[], bbox_expand=0.0, paint_selected=False, **camera_options
):
    if not isinstance(elements, list):
        elements = [elements]

    bboxes = []
    for element in elements:
        bbox = element.get_axis_aligned_bounding_box()
        bbox = bbox.expanded(bbox_expand)
        bbox.color = (1, 0, 0)
        bbox = bbox.as_open3d
        bboxes.append(bbox)

    elements = [elem.as_open3d for elem in elements]

    if not isinstance(fixed_elements, list):
        fixed_elements = [fixed_elements]
    fixed_elements = [elem.as_open3d for elem in fixed_elements]

    color_bbox_selected = (0, 0.8, 0)
    color_bbox_unselected = (1, 0, 0)

    color_selected = (0, 0.4, 0)
    color_selected_current = color_bbox_selected

    color_unselected = (0.9, 0.9, 0.9)
    color_unselected_current = (0.0, 0.0, 0.6)

    elements_painted = get_painted(elements, color_unselected)
    if paint_selected:
        elements_painted[0] = get_painted(elements_painted[0], color_unselected_current)
    else:
        elements_painted[0] = elements[0]

    global data

    window_name = f"{len(elements)} elements. Green: selected. White: unselected. (T)oggle | (D) next | (A) previous"

    data = {
        "selected": [False] * len(elements),
        "elements_painted": elements_painted,
        "i_old": 0,
        "i": 0,
    }

    def update(vis):
        global data
        i_old = data["i_old"]
        i = data["i"]

        element = data["elements_painted"][i_old]
        vis.remove_geometry(element, reset_bounding_box=False)
        vis.remove_geometry(bboxes[i_old], reset_bounding_box=False)
        if not data["selected"][i_old]:
            bboxes[i_old].color = color_bbox_unselected
            element = get_painted(element, color_unselected)

        else:
            bboxes[i_old].color = color_bbox_selected
            if paint_selected:
                element = get_painted(element, color_selected)
            else:
                element = elements[i_old]

        data["elements_painted"][i_old] = element

        vis.add_geometry(element, reset_bounding_box=False)

        element = data["elements_painted"][i]
        vis.remove_geometry(element, reset_bounding_box=False)
        if not paint_selected:
            element = elements[i]
        elif data["selected"][i]:
            element = get_painted(element, color_selected_current)
        else:
            element = get_painted(element, color_unselected_current)
        data["elements_painted"][i] = element

        vis.add_geometry(element, reset_bounding_box=False)
        vis.add_geometry(bboxes[i], reset_bounding_box=False)

        data["i_old"] = i

    def toggle(vis):
        global data
        data["selected"][data["i"]] = not data["selected"][data["i"]]
        update(vis)

    def next(vis):
        global data
        data["i_old"] = data["i"]
        data["i"] = min(data["i_old"] + 1, len(elements) - 1)
        update(vis)

    def previous(vis):
        global data
        data["i_old"] = data["i"]
        data["i"] = max(data["i_old"] - 1, 0)
        update(vis)

    key_to_callback = {}
    key_to_callback[ord("S")] = toggle
    key_to_callback[ord("D")] = next
    key_to_callback[ord("A")] = previous

    visualization.draw_geometries_with_key_callbacks(
        fixed_elements + data["elements_painted"] + [bboxes[0]],
        # data['elements_painted'],
        key_to_callback,
        window_name=window_name,
    )

    return data["selected"]


# def draw_and_ask(elements, return_not_selected=False, **camera_options):
#     elements_original = elements

#     def _paint_element(element, color):
#         try:
#             element.paint_uniform_color(color)
#         except Exception:
#             element.color = color

#     color_remaining = (0, 0, 1)
#     color_selected = (0, 1, 0)
#     color_discarded = (0.9, 0.9, 0.9)
#     color_test = (1, 0, 0)

#     instructions = (
#         " - Red: current, Blue: remaining, Green: selected, White: Unselected."
#     )

#     if not isinstance(elements, (list, tuple)):
#         elements = [elements]

#     window_name = camera_options.get("window_name", "")
#     if window_name != "":
#         window_name += " - "

#     elements = copy.deepcopy(elements)
#     for element in elements:
#         _paint_element(element, color_remaining)

#     N = len(elements)
#     indices_selected = []
#     indices_not_selected = []
#     for i, element in enumerate(elements):
#         # element_red = copy.deepcopy(element)
#         _paint_element(element, color_test)

#         # try:
#         #     element_red.paint_uniform_color(color_test)
#         # except:
#         #     element_red.color = color_test

#         camera_options["window_name"] = window_name + f"{i+1}/{N}" + instructions
#         draw_two_columns(
#             elements[:i] + [element] + elements[(i + 1) :],
#             elements_original[i],
#             **camera_options,
#         )
#         out = input(f"Get element {i+1}/{N}? (y)es, (N)o, (s)top: ").lower()
#         if out == "y" or out == "yes":
#             indices_selected.append(i)
#             _paint_element(element, color_selected)
#         elif out == "s" or out == "stop":
#             indices_not_selected.append(i)
#             indices_not_selected += list(range(i + 1, len(elements)))
#             break
#         elif return_not_selected:
#             indices_not_selected.append(i)
#             _paint_element(element, color_discarded)

#     if return_not_selected:
#         return indices_selected, indices_not_selected
#     return indices_selected


def select_combinations_manually(elements, return_grouped=False):
    if not isinstance(elements, list) or len(elements) < 2:
        raise ValueError(
            "'elements' must be a list with at least 2 elements, " f"got {elements}."
        )

    elements_test = [[elem] for elem in get_painted(elements)]

    partitions = np.arange(N := len(elements))
    for i, j in itertools.combinations(range(N), 2):
        if partitions[i] == partitions[j]:
            continue
        elif len(elements_test[i]) == 0 or len(elements_test[j]) == 0:
            continue

        elements_test[i] = get_painted(elements_test[i], color=(0, 0, 1))
        elements_test[j] = get_painted(elements_test[j], color=(0, 1, 0))

        draw_geometries(elements_test[i] + elements_test[j], window_name=f"{i} and {j}")

        option = input(f"Fuse {i} and {j}?  (y)es, (N)o, (s)top: ").lower()

        if option == "s" or option == "stop":
            break
        elif option == "y" or option == "yes":
            partitions[j] = partitions[i]
            # elements_test[j].paint_uniform_color(elements_test[i].colors[0])
            elements_test[i] += elements_test[j]
            elements_test[j] = []

            # pcds_no_ground[i] += pcds_no_ground[j]
            # pcds_no_ground[j].points = []

    if return_grouped:
        grouped = []
        elements = np.array(elements)
        for partition in set(partitions):
            grouped.append(elements[partition == partitions].tolist())
        return partitions, grouped

    return partitions
