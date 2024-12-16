#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions that help with visualization or manual selection.

Methods
-------
get_painted
get_open3d_geometries
draw_geometries
draw_two_columns
select_manually
apply_function_manually

Created on Wed Feb 28 10:59:02 2024

@author: ebernardes
"""
import copy
import warnings
from importlib.util import find_spec
import numpy as np
from multiprocessing import Manager, Process
from open3d import visualization

from .interactive_gui.editor_app import Editor

if has_matplotlib := find_spec("matplotlib") is not None:
    import matplotlib.pyplot as plt
if not has_matplotlib:
    warnings.warn(
        "Optional dependency 'Matplotlib' not found",
    )

# from .interactive_window import InteractiveWindow
from .input_selector import InputSelector, SingleChoiceSelector


GLFW_KEY_LEFT_SHIFT = 340
GLFW_KEY_LEFT_CONTROL = 341


def get_inputs(specs, window_name="Enter values", as_dict=False):
    """Get values from user with a graphical interface, using a dictionary that
    defines values to get.

    Runs in a separate process for comatibility with Open3D.Visualizer.

    For example, defining the specs dictionary for two different variables:
        specs = {
           'name': (str, 'default string'),
           'number': (int, 1)}

    And then entering the dict as input:
        get_inputs(specs)

    Opens a window asking for both "name" and "number", with "default string"
    and "1" pre-entered.


    Parameters
    ----------
    specs : dict
        A dictionary where each key is the variable name, and each value is a
        tuple (expected_type, default_value), specifying the type and default
        value for the input variable.

    window_name : string, optional
        Name of window. Default: "Enter values".

    as_dict : boolean, optional
        If True, get results as dictionary. Else, get them as list. Default: False.

    Returns
    -------
    list

    """
    manager = Manager()
    results = manager.list()

    def _get_inputs_worker(specs, results):
        try:
            for result in InputSelector(specs, window_name=window_name).get_results():
                results.append(result)
        except KeyboardInterrupt:
            results.append(None)

    process = Process(target=_get_inputs_worker, args=(specs, results))
    process.start()
    process.join()
    results = list(results)

    if results == [None]:
        raise KeyboardInterrupt

    if as_dict:
        results = {name: value for name, value in zip(specs.keys(), results)}

    return results


def select_function_with_gui(functions, default_function=None):
    """Make user select from a list of functions with a GUI window.

    Parameters
    ----------
    functions : list
        List of functions.
    default_function : function, optional
        Default function.

    Returns
    -------
    function
        The selected function.
    """
    for func in functions:
        if not callable(func):
            raise ValueError(f"Expected functions, got {type(func)}.")

    if default_function is None or default_function not in functions:
        default_function = functions[0]

    # Get function names for display
    function_names = [f.__name__ for f in functions]

    # Create a multiprocessing Manager for safely sharing data between processes
    manager = Manager()
    results = manager.list()

    def _get_choice_selector_worker(results):
        try:
            # Initialize the selector with function names
            selector = SingleChoiceSelector(
                choices=function_names,
                default_value=default_function.__name__,  # Default to the first function
                window_name="Choose a Function",
            )
            # Get the selected result and store it
            results.append(selector.get_result())
        except KeyboardInterrupt:
            results.append(None)  # Append None if the process is interrupted

    # Start a separate process for the GUI
    process = Process(target=_get_choice_selector_worker, args=(results,))
    process.start()
    process.join()

    # Handle the result after the process ends
    if not results or results[0] is None:
        raise KeyboardInterrupt("No function selected.")

    # Get the index of the selected function
    selected_function_name = results[0]
    idx = function_names.index(selected_function_name)

    return functions[idx]


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

    if not (is_list := isinstance(elements, (list, tuple))):
        elements = [elements]

    if isinstance(color, str) or color is None:
        if color == "random":
            colors = np.random.random((len(elements), 3))
        else:
            if not has_matplotlib:
                raise ImportError(
                    "Calling get_painted called with cmap string requires matplotlib."
                )
            color_map = plt.get_cmap(color)
            colors = [color_map(i)[:3] for i in range(len(elements))]

    else:
        colors = [color] * len(elements)

    for color, element in zip(colors, elements):
        color = np.asarray(color)

        if hasattr(element, "paint_uniform_color"):
            element.paint_uniform_color(color)
        elif hasattr(element, "color"):
            element.color = color
        else:
            warnings.warn("Could not paint element {element}.")

        # TODO: Why did I put this here again?
        # color = np.random.random(3)
        if isinstance(element, Primitive):
            element.inliers.paint_uniform_color(color)

    if is_list:
        return elements
    else:
        return elements[0]


def get_open3d_geometries(elements, **camera_options):
    """Get Open3D compatible/drawable forms of input elements:

    Open3D-compatible geometries:
        Transformed into the Open3D legacy geometry internal element.
        Currently valid for:
            PointCloud
            TriangleMesh
            LineSet
            AxisAlignedBoundingBox
            OrientedBoundingBox

    Primitives:
        Transformed into TriangleMeshes

    Nx3 arrays:
        Transformed into PointClouds

    Parameters
    ----------
    elements : list
        Elements to be drawn.
    draw_inliers : boolean, optional
        If True, returns inliers of every Primitive instance. Default: False.
    draw_boundary_lines : boolean, optional
        If True, returns LineSet instances for every plane, showing their
         boundaries. Default: False.
    draw_planes : boolean, optional
        If False, ignores planes. Default: True.

    Returns
    -------
    list
        Elements to be drawn.
    """

    from pyShapeDetector.primitives import Primitive, Line, Plane, PlaneBounded
    from pyShapeDetector.geometry import PointCloud

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

    if "mesh_show_back_face" not in camera_options:
        camera_options["mesh_show_back_face"] = True

    pcds = []
    geometries = []
    lines = []
    boundary_lines = []
    hole_boundary_lines = []

    if not isinstance(elements, (list, tuple)):
        elements = [elements]

    for element in elements:
        if element is None:
            continue
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
            boundary_LineSet = element.vertices_LineSet
            boundary_LineSet.paint_uniform_color((1, 0, 0))
            boundary_lines.append(boundary_LineSet.as_open3d)

        if draw_boundary_lines and isinstance(element, Plane):
            for hole in element.holes:
                hole_boundary_LineSet = hole.vertices_LineSet
                hole_boundary_LineSet.paint_uniform_color((0, 0, 1))
                hole_boundary_lines.append(hole_boundary_LineSet.as_open3d)

    if len(lines) > 0:
        geometries.append(Line.get_LineSet_from_list(lines))

    if draw_inliers:
        geometries += pcds

    if draw_boundary_lines:
        geometries += boundary_lines + hole_boundary_lines

    return geometries, camera_options


def draw_geometries(elements, **camera_options):
    """Get Open3D compatible/drawable forms of input elements and then
    draw them with Open3D.

    See also:
        get_open3d_geometries, open3d.visualization.draw_geometries

    Parameters
    ----------
    elements : list
        Elements to be drawn.
    draw_inliers : boolean, optional
        If True, returns inliers of every Primitive instance. Default: False.
    draw_boundary_lines : boolean, optional
        If True, returns LineSet instances for every plane, showing their
         boundaries. Default: False.
    draw_planes : boolean, optional
        If False, ignores planes. Default: True.
    other camera options : optional
        See camera options on open3d.visualization.draw_geometries

    Returns
    -------
    list
        Elements to be drawn.
    """
    geometries, camera_options = get_open3d_geometries(elements, **camera_options)
    visualization.draw_geometries(geometries, **camera_options)


def draw_two_columns(objs_left, objs_right, **camera_options):
    """Get Open3D compatible/drawable forms of two lists of input elements and
    then draw them side-by-side with Open3D.

    See also:
        get_open3d_geometries, draw_geometries, open3d.visualization.draw_geometries

    Parameters
    ----------
    objs_left : list
        Elements to be drawn to the left.
    objs_right : list
        Elements to be drawn to the right.
    dist : float, optional
        Distance between left and right elements. Default: 5.
    draw_inliers : boolean, optional
        If True, returns inliers of every Primitive instance. Default: False.
    draw_boundary_lines : boolean, optional
        If True, returns LineSet instances for every plane, showing their
         boundaries. Default: False.
    draw_planes : boolean, optional
        If False, ignores planes. Default: True.
    other camera options : optional
        See camera options on open3d.visualization.draw_geometries

    Returns
    -------
    list
        Elements to be drawn.
    """

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
    elements,
    fixed_elements=[],
    pre_selected=None,
    return_finish_flag=False,
    **kwargs,
):
    """Plots elements on an interactive visualizer so that separate elements
    can be manually selected with the mouse and/or keyboard.

    Keys:
        (Space) Toggle
        (<-) Previous
        (->) Next
        (F) Finish
        (LCtrl) Enables mouse selection
        (LCtrl) + (H) Print Help
        (LCtrl) + (I) Print Info
        (LCtrl) + (C) Center current
        (LCtrl) + (Z) Undo
        (LCtrl) + (Y) Redo
        (LCtrl) + (A) Toggle all
        (LCtrl) + (L) Toggle last
        (LCtrl) + (T) Toggle type
        (LCtrl) + (LShift) Toggle click

    See also:
        open3d.visualization.VisualizerWithKeyCallback
        utility.ElementSelector
        apply_function_manually

    Parameters
    ----------
    elements : list
        Elements to be drawn.
    fixed_elements : list, optional
        Unselectable elements to be drawn. Default: empty.
    pre_selected : list, optional
        List of boolean values for elements to be pre-selected when starting
        vizualization. Has to be the same length as the number of elements.
    other options : optional
        For other options, see: utility.ElementSelector

    Returns
    -------
    list
        List of booleans showing which of the elements where selected.
    """

    if "print_instructions" in kwargs:
        warnings.warn("print_instructions option not used anymore")

    element_selector = Editor(**kwargs)
    element_selector.add_elements(elements, pre_selected=pre_selected)
    element_selector.add_elements(fixed_elements, fixed=True)
    element_selector.run()

    if "functions" in kwargs:
        warnings.warn("To use a function, try 'apply_function_manually' instead.")

    if return_finish_flag:
        return element_selector.selected, element_selector.finish
    return element_selector.selected


def apply_function_manually(
    elements,
    functions=[],
    function_submenus={},
    fixed_elements=[],
    pre_selected=None,
    return_indices=False,
    **kwargs,
):
    """Plots elements on an interactive visualizer so that separate elements
    can be manually selected with the mouse and/or keyboard, and then apply
    input function fo them.

    At each step, the selected elements are removed from the list, the input
    function is applied to the list of selected elements, and the list of
    output elements is appended at the end of the list.

    Keys:
        (Space) Toggle
        (<-) Previous
        (->) Next
        (F) Finish
        (LCtrl) Enables mouse selection
        (LCtrl) + (H) Print Help
        (LCtrl) + (I) Print Info
        (LCtrl) + (C) Center current
        (LCtrl) + (Z) Undo
        (LCtrl) + (Y) Redo
        (LCtrl) + (A) Toggle all
        (LCtrl) + (L) Toggle last
        (LCtrl) + (T) Toggle type
        (LCtrl) + (LShift) Toggle click
        (LCtrl) + (1...0) Apply one of the selected functions

    See also:
        open3d.visualization.VisualizerWithKeyCallback
        utility.ElementSelector
        select_manually

    Parameters
    ----------
    elements : list
        Elements to be drawn.
    functions : function or list of functions
        Function(s) to be applied on elements.
    fixed_elements : list, optional
        Unselectable elements to be drawn. Default: empty.
    pre_selected : list, optional
        List of boolean values for elements to be pre-selected when starting
        vizualization. Has to be the same length as the number of elements.
    return_indices : boolean, optional
        If True, also return indices to which the functions are applied at each
        step. Default: False.
    other options : optional
        For other options, see: utility.ElementSelector

    Returns
    -------
    list
        Modified elements

    list
        List containing sublists of indices
    """
    if pre_selected is None:
        pre_selected = [False] * len(elements)

    if "print_instructions" in kwargs:
        warnings.warn("print_instructions option not used anymore")

    element_selector = Editor(**kwargs)
    # element_selector.add_elements(elements, pre_selected=pre_selected)
    # for element, selected in zip(elements, pre_selected):
    element_selector.elements.insert_multiple(
        elements, selected=pre_selected, to_gui=False
    )
    element_selector.elements_fixed.insert_multiple(fixed_elements, to_gui=False)

    for function in functions:
        element_selector.add_extension(function)

    for menu, functions_list in function_submenus.items():
        for function in functions_list:
            element_selector.add_extension({"function": function, "menu": menu})

    element_selector.run()

    if return_indices:
        indices = [state["indices"] for state in element_selector._past_states]
        return element_selector.element_dicts, indices

    return element_selector.get_elements()


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


def select_combinations_manually(
    elements,
    return_grouped=False,
    BBOX_expand=0.0,
    paint_selected=True,
    window_name="",
    **camera_options,
):
    if not isinstance(elements, list):
        raise ValueError("'elements' must be a list.")

    if len(elements) == 0:
        warnings.warn("'elements' has no elements, doing nothing...")

        if return_grouped:
            return [], []
        else:
            return []

    if len(elements) == 1:
        warnings.warn("'elements' has only one element, doing nothing...")

        if return_grouped:
            return [0], elements
        else:
            return [0]

    # elements_test = [[elem] for elem in get_painted(elements)]
    elements = copy.deepcopy(elements)
    elements_fused = [[elem] for elem in elements]

    partitions = np.arange(N := len(elements))

    if window_name != "":
        window_name += " - "

    for i in range(N):
        if elements[i] is None:
            # if len(elements_test[i]) == 0:
            continue

        if partitions[i] != i:
            continue

        if len(elements[i + 1 :]) == 0:
            continue

        selected, finish_flag = select_manually(
            elements[i + 1 :],
            fixed_elements=elements_fused[i],
            window_name=f"{window_name}Element {i}",
            BBOX_expand=BBOX_expand,
            paint_selected=paint_selected,
            return_finish_flag=True,
            **camera_options,
        )

        indices = np.where(selected)[0]
        partitions[i + 1 :][indices] = partitions[i]

        for idx in indices:
            if isinstance(elements[i + 1 :][idx], list):
                elements_fused[i] += elements[i + 1 :][idx]
            else:
                elements_fused[i].append(elements[i + 1 :][idx])
            elements[i + 1 :][idx] = None

        if finish_flag:
            break

        # for i, j in itertools.combinations(range(N), 2):

        # if partitions[i] == partitions[j]:
        # continue
        # elif len(elements_test[i]) == 0 or len(elements_test[j]) == 0:
        # continue

        # elements_test[i] = get_painted(elements_test[i], color=(0, 0, 1))
        # elements_test[j] = get_painted(elements_test[j], color=(0, 1, 0))

        # draw_geometries(elements_test[i] + elements_test[j], window_name=f"{i} and {j}")

        # option = input(f"Fuse {i} and {j}?  (y)es, (N)o, (s)top: ").lower()

        # if option == "s" or option == "stop":
        #     break
        # elif option == "y" or option == "yes":
        #     partitions[j] = partitions[i]
        #     # elements_test[j].paint_uniform_color(elements_test[i].colors[0])
        #     elements_test[i] += elements_test[j]
        #     elements_test[j] = []

        #     # pcds_no_ground[i] += pcds_no_ground[j]
        #     # pcds_no_ground[j].points = []

    if return_grouped:
        grouped = []
        elements = np.array(elements)
        for partition in set(partitions):
            grouped.append(elements[partition == partitions].tolist())
        return partitions, grouped

    return partitions
