#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tur Oct 17 13:17:55 2024

@author: ebernardes
"""
import time
import copy
import signal
import sys
import warnings
import inspect
from abc import ABC
import numpy as np
from open3d import visualization
from open3d.utility import Vector3dVector

COLOR_BBOX_SELECTED = (0, 0.8, 0)
COLOR_BBOX_UNSELECTED = (1, 0, 0)

COLOR_SELECTED_CURRENT = np.array([255, 234, 0]) / 255  # flashy yellow
COLOR_SELECTED = 0.7 * COLOR_SELECTED_CURRENT

COLOR_UNSELECTED_CURRENT = (0.6, 0.6, 0.6)
COLOR_UNSELECTED = (0.3, 0.3, 0.3)

KEYS_NORMAL = {
    "Toggle": ["Space", 32],  # GLFW_KEY_SPACE = 32
    "Previous": ["<-", 263],  # GLFW_KEY_LEFT = 263
    "Next": ["->", 262],  # GLFW_KEY_RIGHT = 262
    "Finish": ["F", ord("F")],
    "Preferences": ["P", ord("P")],
    "Color mode": ["M", ord("M")],
}

EXTRA_KEY = ["LCtrl", 341]  # GLFW_KEY_LCTRL = 341

KEYS_EXTRA = {
    "Print Help": ["H", ord("H")],
    "Print Info": ["I", ord("I")],
    "Center current": ["C", ord("C")],
    "Functions menu": ["Enter", 257],  # GLFW_KEY_ENTER = 257
    "Undo": ["Z", ord("Z")],
    "Redo": ["Y", ord("Y")],
    "Hide/Unhide": ["U", ord("U")],
    "Toggle all": ["A", ord("A")],
    "Toggle last": ["L", ord("L")],
    "Toggle type": ["T", ord("T")],
    "Toggle click": ["LShift", 340],  # GLFW_KEY_LSHIFT = 340
}

all_keys = [key[1] for key in KEYS_NORMAL.values()]
all_keys += [key[1] for key in KEYS_EXTRA.values()]
assert len(all_keys) == len(set(all_keys))


def _unproject_screen_to_world(vis, x, y):
    """
    Convert screen coordinates (x, y, depth) to 3D coordinates.
    """

    depth = vis.capture_depth_float_buffer(True)
    depth = np.asarray(depth)[y, x]

    intrinsic = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic
    extrinsic = vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic

    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    # Convert screen space to camera space
    z = depth
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    # Convert camera space to world space
    camera_space_point = np.array([x, y, z, 1.0]).reshape(4, 1)
    world_space_point = np.dot(np.linalg.inv(extrinsic), camera_space_point)
    point = world_space_point[:3].flatten()

    # Extract the camera forward direction (view normal vector)
    camera_forward = np.array([0, 0, -1, 1.0])  # The forward direction in camera space
    world_forward_vector = np.dot(np.linalg.inv(extrinsic), camera_forward)[:3]
    normal_view_vector = world_forward_vector / np.linalg.norm(world_forward_vector)

    return point, normal_view_vector


class InteractiveWindow:
    """
    Visualizer class used to manually select elements and apply functions to
    them.

    Elements can be selected either with a keyboard or a mouse.

    This has only been tested for:
        PointCloud
        TriangleMesh
        2D Primitives

    1D primitives (like Line instances) cannot be used with this.

    To better understand how to use it, see also:
        pyShapeDetector.utility.select_manually
        pyShapeDetector.utility.apply_function_manually

    Parameters
    ----------
    window_name : str, optional
        Name of window. If empty, just gives the number of elements. Default: "".

    draw_boundary_lines : boolean, optional
        If True, draws the boundaries of planes as LineSets. Default: False.
    mesh_show_back_face : optional, boolean
        If True, shows back faces of surfaces and meshes. Default: True.
    bbox_expand : float, optional
        Expands bounding boxes in all directions with this value. Default: 0.0.
    paint_selected : boolean, optional
        If True, paint selected elements, and not only their bounding boxes.
        Default: True
    paint_random : boolean, optional
        If True, paint all elements with a random color. Default: False
    number_points_for_distance : int, optional
        Number of points in element distance calculator. Default: 30.
    number_undo_states : int, optional
        Number of states to save for undoing. Default: 10.
    number_redo_states : int, optional
        Number of states to save for redoing. Default: 5.
    random_color_multiplier : float, optional
        Random colors are multiplied by this value to reduce their brightness,
        creating bigger contrast with selected objects. Default: 2/3.
    highlight_color_multiplier : float, optional
        Multiplier for color to show selected and current elements. Default: 0.3.
    debug : boolean, optional
        If True, prints debug information. Default: False.
    verbose : boolean, optional
        If both this and 'debug' are True, prints extra debug information.
        Default: False.

    return_finish_flag : boolean, optional
        Should be deprecated.
    """

    def __init__(
        self,
        window_name="",
        draw_boundary_lines=True,
        mesh_show_back_face=True,
        bbox_expand=0.0,
        paint_selected=True,
        paint_random=False,
        number_points_for_distance=30,
        number_undo_states=10,
        number_redo_states=5,
        random_color_multiplier=2 / 3,
        highlight_color_multiplier=0.3,
        debug=False,
        verbose=False,
        return_finish_flag=False,
    ):
        self._past_states = []
        self._future_states = []
        self._element_dicts = []
        self._elements_input = []
        self._plane_boundaries = []
        self._hidden_elements = []
        self._fixed_elements = []
        self._fixed_elements_drawable = []
        self._pre_selected = []
        self._bbox = None
        self._functions = None
        self._select_filter = lambda x: True
        self._instructions = ""
        self.window_name = window_name
        self.return_finish_flag = return_finish_flag

        # self._elements_distance = []
        self.i_old = 0
        self.i = 0
        self.finish = False
        self.extra_functions = False
        self.mouse_toggle = False
        self.mouse_position = (0, 0)
        self._started = False

        self._vis = None

        # Set up colors and elements
        self.color_bbox_selected = COLOR_BBOX_SELECTED
        self.color_bbox_unselected = COLOR_BBOX_UNSELECTED
        self.color_selected = COLOR_SELECTED
        self.color_selected_current = COLOR_SELECTED_CURRENT
        self.color_unselected = COLOR_UNSELECTED
        self.color_unselected_current = COLOR_UNSELECTED_CURRENT

        # preferences dict:
        self._preferences = {
            "draw_boundary_lines": bool(draw_boundary_lines),
            "mesh_show_back_face": bool(mesh_show_back_face),
            "bbox_expand": float(bbox_expand),
            "paint_selected": bool(paint_selected),
            "paint_random": bool(paint_random),
            "number_points_for_distance": int(number_points_for_distance),
            "random_color_multiplier": np.clip(float(random_color_multiplier), 0, 1),
            "highlight_color_multiplier": np.max(float(highlight_color_multiplier), 0),
            "number_undo_states": int(number_undo_states),
            "number_redo_states": int(number_redo_states),
            "debug": bool(debug),
            "verbose": bool(verbose),
        }

        # (selected, current) -> color
        self._colors_selected_current = {
            (False, False): self.color_unselected,
            (False, True): self.color_unselected_current,
            (True, False): self.color_selected,
            (True, True): self.color_selected_current,
        }

    def print_debug(self, text, require_verbose=False):
        is_debug_activated = self._preferences["debug"]
        is_verbose_activated = self._preferences["verbose"]

        if not is_debug_activated or (require_verbose and not is_verbose_activated):
            return

        text = str(text)
        print("[DEBUG] " + text)

    @property
    def function_key_mappings(self):
        key_mappings = dict()
        if self.functions is not None:
            for n, f in enumerate(self.functions):
                key = ord(str((n + 1) % 10))
                key_mappings[key] = f
        return key_mappings

    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, new_functions):
        if new_functions is None:
            self._functions = None
            return

        if isinstance(new_functions, tuple):
            new_functions = list(new_functions)
        elif callable(new_functions):
            new_functions = [new_functions]

        if (N := len(new_functions)) > 10:
            warnings.warn(
                f"Got: {N} functions, only the first 10 will have separete keys. "
                "Use full menu to access others."
            )

        if not isinstance(new_functions, list):
            raise ValueError(
                f"Expected function or list of functions, got {type(new_functions)}."
            )

        for function in new_functions:
            if not callable(function):
                raise ValueError(
                    f"Expected function or list of functions, got {type(function)}."
                )

            elif (N := len(inspect.signature(function).parameters)) < 1:
                raise ValueError(
                    f"Expected function with at least 1 parameter (list of elements), got {N}."
                )

        self._functions = new_functions

    @property
    def select_filter(self):
        return self._select_filter

    @select_filter.setter
    def select_filter(self, new_function):
        if new_function is None:
            new_function = lambda x: True
        elif (N := len(inspect.signature(new_function).parameters)) != 1:
            raise ValueError(
                f"Expected filter function with 1 parameter (element), got {N}."
            )
        self._select_filter = lambda elem: new_function(elem["raw"])

    def get_elements(self):
        return [elem["raw"] for elem in self.elements]

    @property
    def elements(self):
        return self._element_dicts

    @property
    def current_element(self):
        num_elems = len(self.elements)
        if self.i in range(num_elems):
            return self.elements[self.i]
        else:
            warnings.warn(
                f"Tried to update index {self.i}, but {num_elems} elements present."
            )
            return None

    @property
    def is_current_selected(self):
        if self.current_element is None:
            return None
        return self.current_element["selected"]

    @is_current_selected.setter
    def is_current_selected(self, boolean_value):
        if not isinstance(boolean_value, bool):
            raise RuntimeError(f"Expected boolean, got {type(boolean_value)}.")
        if self.current_element is None:
            raise RuntimeError(
                f"Error setting selected, current element does not exist."
            )
        self.current_element["selected"] = boolean_value

    def _add_geometry_to_vis(self, elem, reset_bounding_box=False):
        # override reset_bounding_box if has not started yet
        reset_bounding_box = reset_bounding_box or not self._started
        self._vis.add_geometry(elem, reset_bounding_box=reset_bounding_box)

    def _remove_geometry_from_vis(self, elem, reset_bounding_box=False):
        self._vis.remove_geometry(elem, reset_bounding_box=reset_bounding_box)

    def add_elements(self, elems_raw, pre_selected=None, fixed=False):
        if fixed and pre_selected is not None:
            raise ValueError("Cannot select fixed elements.")

        if isinstance(elems_raw, (list, tuple)):
            new_raw_elements = copy.deepcopy(list(elems_raw))
        else:
            new_raw_elements = [copy.deepcopy(elems_raw)]

        if pre_selected is None:
            pre_selected = [False] * len(new_raw_elements)
        elif isinstance(pre_selected, bool):
            pre_selected = [pre_selected]
        elif len(pre_selected) != len(new_raw_elements):
            raise ValueError(
                f"Got {len(new_raw_elements)} elements but {len(pre_selected)} "
                " pre-selections."
            )

        if fixed:
            self._fixed_elements += new_raw_elements
        else:
            # self._insert_elements(new_elements, to_vis=False)
            self._elements_input += new_raw_elements
            self._pre_selected += pre_selected

    def _pop_elements(self, indices, from_vis=False):
        # update_old = self.i in indices
        # idx_new = self.i
        elements_popped = []
        for n, i in enumerate(indices):
            elem = self._element_dicts.pop(i - n)
            if from_vis:
                self._remove_geometry_from_vis(elem["drawable"])
            elements_popped.append(elem["raw"])

        idx_new = self.i - sum([idx < self.i for idx in indices])
        self.print_debug(
            f"popped: {indices}",
        )
        self.print_debug(f"old index: {self.i}, new index: {idx_new}")

        if len(self.elements) == 0:
            warnings.warn("No elements left after popping, not updating.")
            self.i = 0
            self.i_old = 0
        else:
            idx_new = max(min(idx_new, len(self.elements) - 1), 0)
            self._update_current_idx(idx_new, update_old=False)
            self.i_old = self.i
        return elements_popped

    # def _get_element_dict(self, elem_raw, selected=False, idx=None):
    #     elem_dict = {
    #         "raw": elem_raw,
    #         "selected": selected,
    #         "drawable": self._get_open3d(elem_raw),
    #         "distance_checker": self._get_element_distances([elem_raw])[0],
    #     }

    #     # save original colors
    #     elem_dict["color"] = self._extract_element_colors(elem_dict["drawable"])

    #     if self._preferences["paint_selected"]:
    #         if idx is None:
    #             is_current = False
    #         else:
    #             is_current = self.i == idx
    #         color = self._colors_selected_current[selected, is_current]

    #         elem_dict["drawable"] = self._get_painted(elem_dict["drawable"], color)

    #     return elem_dict

    def _insert_elements(self, elems_raw, indices=None, selected=False, to_vis=False):
        if indices is None:
            indices = range(len(self.elements), len(self.elements) + len(elems_raw))

        if isinstance(selected, bool):
            selected = [selected] * len(indices)

        idx_new = self.i

        self.print_debug(
            f"Adding {len(elems_raw)} elements to the existing {len(self.elements)}",
            require_verbose=True,
        )

        for i, idx in enumerate(indices):
            if idx_new > idx:
                idx_new += 1

            elem = {
                "raw": elems_raw[i],
                "selected": selected[i],
                "drawable": self._get_open3d(elems_raw[i]),
                "distance_checker": self._get_element_distances([elems_raw[i]])[0],
            }

            # save original colors
            elem["color"] = self._extract_element_colors(elem["drawable"])

            if self._preferences["paint_random"]:
                self.print_debug(
                    f"[_insert_elements] Randomly painting element at index {i}.",
                    require_verbose=True,
                )
                elem["drawable"] = self._get_painted_element(
                    elem["drawable"], color="random"
                )

            if self._preferences["paint_selected"] and selected[i]:
                self.print_debug(
                    f"[_insert_elements] Painting and inserting element at index {i}.",
                    require_verbose=True,
                )
                is_current = self._started and (self.i == idx)
                color = self._colors_selected_current[True, is_current]
                elem["drawable"] = self._get_painted_element(elem["drawable"], color)

            self.print_debug(
                f"Added {elem['raw']} at index {idx}.", require_verbose=True
            )
            self._element_dicts.insert(idx, elem)

        for idx in indices:
            # Updating vis explicitly in order not to remove it
            self._update_element(idx, update_vis=False)
            if to_vis:
                self._add_geometry_to_vis(self.elements[idx]["drawable"])

        self.print_debug(f"{len(self.elements)} now.", require_verbose=True)

        # self.i += sum([idx <= self.i for idx in indices])
        # if self._started:
        idx_new = max(min(idx_new, len(self.elements) - 1), 0)
        self._update_current_idx(idx_new, update_old=self._started)
        self.i_old = self.i

    @property
    def fixed_elements(self):
        return self._fixed_elements

    @property
    def selected_indices(self):
        return np.where(self.selected)[0].tolist()

    @property
    def elements_drawable(self):
        return [elem["drawable"] for elem in self.elements]

    @property
    def selected(self):
        return [elem["selected"] for elem in self.elements]

    @selected.setter
    def selected(self, values):
        if isinstance(values, bool):
            values = [values] * len(self.elements)

        elif len(values) != len(self.elements):
            raise ValueError(
                "Length of input expected to be the same as the "
                f"current number of elements ({len(self.elements)}), "
                f"got {len(values)}."
            )

        values = copy.deepcopy(values)

        for elem, value in zip(self.elements, values):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"Expected boolean, got {type(value)}")

            elem["selected"] = value and self.select_filter(elem)

    @property
    def all_drawable_elements(self):
        return self._fixed_elements + self.elements_drawable

    def _reset_selected(self, indices_true=[]):
        """Reset boolean array of selected values to False, then revert some to True is asked."""
        for i, elem in enumerate(self.elements):
            elem["selected"] = i in indices_true

    def distances_to_point(self, screen_point, screen_vector):
        from pyShapeDetector.geometry import PointCloud
        from pyShapeDetector.primitives import Primitive, Plane

        screen_plane = Plane.from_normal_point(screen_vector, screen_point)

        def _is_point_in_convex_region(elem, point=screen_point, plane=screen_plane):
            """Check if mouse-click was done inside of the element actual region."""

            if hasattr(elem, "points"):
                boundary_points = elem.points
            elif hasattr(elem, "vertices"):
                boundary_points = elem.vertices
            elif hasattr(elem, "mesh"):
                boundary_points = elem.mesh.vertices
            else:
                return False

            boundary_points = np.asarray(boundary_points)
            if len(boundary_points) < 3:
                return False

            plane_bounded = plane.get_bounded_plane(boundary_points, convex=True)
            return plane_bounded.contains_projections(point)

        def _distance_to_point(elem, point=screen_point, plane=screen_plane):
            """Check if mouse-click was done inside of the element actual region."""

            if not _is_point_in_convex_region(elem, point, plane):
                return np.inf

            try:
                return elem.get_distances(point)
            except AttributeError:
                warnings.warn(
                    f"Element of type {type(elem)} "
                    "found in distance elements, should not happen."
                )
                return np.inf

        distances = []
        for i, elem in enumerate(self.elements):
            if i == self.i:
                # for selecting smaller objects closer to bigger ones,
                # ignores currently selected one
                distances.append(np.inf)
            else:
                distances.append(_distance_to_point(elem["distance_checker"]))
        return distances

    def _save_state(self, indices, input_elements, num_outputs):
        """Save state for undoing."""
        current_state = {
            "indices": copy.deepcopy(indices),
            "elements": copy.deepcopy(input_elements),
            "num_outputs": num_outputs,
            "current_index": self.i,
        }

        self._past_states.append(current_state)
        self.print_debug(
            f"Saving state with {len(input_elements)} inputs and {num_outputs} outputs."
        )

        while len(self._past_states) > self._preferences["number_undo_states"]:
            self._past_states.pop(0)

    def _check_and_initialize_inputs(self):
        from pyShapeDetector.geometry import PointCloud

        if len(self._elements_input) != len(self._pre_selected):
            raise RuntimeError(
                f"{len(self._elements_input)} elements but "
                f"{len(self._pre_selected)} selected values fouund."
            )

        # check correct input of elements and fixed elements
        if isinstance(self._elements_input, tuple):
            self._elements_input = list(self._elements_input)
        elif not isinstance(self._elements_input, list):
            self._elements_input = [self._elements_input]

        for elem in self._elements_input:
            if PointCloud.is_instance_or_open3d(elem) and not elem.has_normals():
                elem.estimate_normals()

        if self._fixed_elements is None:
            self._fixed_elements = []
        elif isinstance(self._fixed_elements, tuple):
            self._fixed_elements = list(self._elements_input)
        elif not isinstance(self._fixed_elements, list):
            self._fixed_elements = [self._fixed_elements]

        if (L := len(self._elements_input)) == 0:
            raise ValueError("Elements cannot be an empty list.")

        # if L > self._ELEMENTS_NUMBER_WARNING:
        #     warnings.warn(
        #         f"There are {L} elements, this is too many and may "
        #         "cause segmentation fault errors."
        #     )

        if len(self.selected) != len(self.elements):
            raise ValueError("Pre-select and input elements must have same length.")

        # IMPORTANT: respect order, only get Open3D elements at the very end
        # self._fixed_elements = [self._get_open3d(elem) for elem in self._fixed_elements]
        # self._elements_as_open3d = [self._get_open3d(elem) for elem in self._elements]

    # def _get_drawable_elements(self):
    #     self._original_colors = [
    #         self._extract_element_colors(elem) for elem in self._elements_as_open3d
    #     ]

    #     # either get painted elements or keep original colors
    #     if self._preferences["paint_selected"]:
    #         mask = np.array(self.selected)[np.newaxis].T
    #         colors = self.color_selected * mask + self.color_unselected * ~mask

    #         if self.is_current_selected:
    #             colors[self.i] = self.color_selected_current
    #         else:
    #             colors[self.i] = self.color_unselected_current

    #         self._elements_drawable = [
    #             self._get_painted(elem, color)
    #             for (elem, color) in zip(self._elements_as_open3d, colors)
    #         ]
    #     else:
    #         self._elements_drawable = copy.copy(self._elements_as_open3d)

    def _get_open3d(self, elem):
        from pyShapeDetector.geometry import TriangleMesh
        from open3d.geometry import Geometry as Open3D_Geometry

        if isinstance(elem, Open3D_Geometry):
            return elem

        try:
            elem_new = elem.as_open3d
            mesh_show_back_face = self._preferences["mesh_show_back_face"]
            if mesh_show_back_face and TriangleMesh.is_instance_or_open3d(elem_new):
                mesh = TriangleMesh(elem_new)
                mesh.add_reverse_triangles()
                elem_new = mesh.as_open3d

        except Exception as e:
            warnings.warn(f"Could not convert element: {elem}, got: {str(e)}")
            elem_new = elem

        if self._preferences["paint_random"]:
            elem_new = self._get_painted_element(elem_new, color="random")

        return elem_new

    def _extract_element_colors(self, element):
        if hasattr(element, "color"):
            return np.asarray(element.color).copy()
        if hasattr(element, "colors"):
            return np.asarray(element.colors).copy()
        if hasattr(element, "vertex_colors"):
            return np.asarray(element.vertex_colors).copy()
        return None

    def _set_element_colors(self, element, input_color):
        if input_color is not None:
            input_color = np.clip(input_color, 0, 1)
            if hasattr(element, "color"):
                element.color = Vector3dVector(input_color)
            if hasattr(element, "colors"):
                element.colors = Vector3dVector(input_color)
            if hasattr(element, "vertex_colors"):
                element.vertex_colors = Vector3dVector(input_color)

    def _get_painted_element(self, element, color):
        from .helpers_visualization import get_painted

        if isinstance(element, list):
            raise RuntimeError("Expected single element, not list.")

        # lower luminance of random colors to not interfere with highlights
        if isinstance(color, str) and color == "random":
            multiplier = self._preferences["random_color_multiplier"]
            color = np.random.random(3) * multiplier

        color = np.clip(color, 0, 1)

        return get_painted(element, color)

    def _get_element_distances(self, elems):
        from pyShapeDetector.primitives import Primitive
        from pyShapeDetector.geometry import TriangleMesh, PointCloud

        number_points_for_distance = self._preferences["number_points_for_distance"]

        elements_distance = []
        for elem in elems:
            if isinstance(elem, Primitive):
                elements_distance.append(elem)

            # assert our PointCloud class instead of Open3D PointCloud class
            elif TriangleMesh.is_instance_or_open3d(elem):
                pcd = elem.sample_points_uniformly(number_points_for_distance)
                elements_distance.append(PointCloud(pcd))

            elif PointCloud.is_instance_or_open3d(elem):
                if len(elem.points) > number_points_for_distance:
                    ratio = int(len(elem.points) / number_points_for_distance)
                    pcd = elem.uniform_down_sample(ratio)
                else:
                    pcd = elem
                elements_distance.append(PointCloud(pcd))

            else:
                elements_distance.append(None)

        return elements_distance

    def _get_visualizer(self):
        vis = visualization.VisualizerWithKeyCallback()

        # Register signal handler to gracefully stop the program
        signal.signal(signal.SIGINT, self._signal_handler)

        # Normal keys
        vis.register_key_action_callback(KEYS_NORMAL["Toggle"][1], self._cb_toggle)
        vis.register_key_action_callback(KEYS_NORMAL["Next"][1], self._cb_next)
        vis.register_key_action_callback(KEYS_NORMAL["Previous"][1], self._cb_previous)
        vis.register_key_action_callback(
            KEYS_NORMAL["Finish"][1], self._cb_finish_process
        )
        vis.register_key_action_callback(
            KEYS_NORMAL["Preferences"][1], self._cb_set_preferences
        )
        vis.register_key_action_callback(
            KEYS_NORMAL["Color mode"][1], self._cb_set_color_mode
        )
        vis.register_key_action_callback(EXTRA_KEY[1], self._cb_switch_extra)

        vis.register_key_action_callback(
            KEYS_EXTRA["Toggle click"][1], self._cb_switch_mouse_toggle
        )

        # Extra (LCtrl) keys
        vis.register_key_action_callback(
            KEYS_EXTRA["Print Help"][1], self._cb_print_help
        )
        vis.register_key_action_callback(
            KEYS_EXTRA["Print Info"][1], self._cb_print_info
        )
        vis.register_key_action_callback(
            KEYS_EXTRA["Center current"][1], self._cb_center_current
        )
        vis.register_key_action_callback(
            KEYS_EXTRA["Hide/Unhide"][1], self._cb_hide_unhide
        )
        vis.register_key_action_callback(
            KEYS_EXTRA["Toggle all"][1], self._cb_toggle_all
        )
        vis.register_key_action_callback(
            KEYS_EXTRA["Toggle last"][1], self._cb_toggle_last
        )
        vis.register_key_action_callback(
            KEYS_EXTRA["Toggle type"][1], self._cb_toggle_type
        )

        vis.register_key_action_callback(KEYS_EXTRA["Undo"][1], self._cb_undo)
        vis.register_key_action_callback(KEYS_EXTRA["Redo"][1], self._cb_redo)
        vis.register_key_action_callback(
            KEYS_EXTRA["Functions menu"][1], self._cb_functions_menu
        )

        # Extra keys for functions
        for key, f in self.function_key_mappings.items():
            vis.register_key_action_callback(
                key,
                lambda vis, action, mods, func=f: self._apply_function_to_elements(
                    func, action
                ),
            )

        window_name = self.window_name
        if window_name == "":
            window_name = "Element selector."

        vis.create_window(window_name)

        self._vis = vis

    def _signal_handler(self, sig, frame):
        self._vis.destroy_window()
        self._vis.close()
        sys.exit(0)

    def _update_get_plane_boundaries(self):
        for plane in self._plane_boundaries:
            self._remove_geometry_from_vis(plane)

        plane_boundaries = []

        if self._preferences["draw_boundary_lines"]:
            for elem in self.elements:
                try:
                    lineset = elem["raw"].vertices_LineSet.as_open3d
                    if hasattr(elem["raw"], "holes"):
                        for hole in elem["raw"].holes:
                            lineset += hole.vertices_LineSet.as_open3d
                except AttributeError:
                    continue
                lineset.paint_uniform_color((0.0, 0.0, 0.0))
                plane_boundaries.append(lineset)

        self._plane_boundaries = plane_boundaries

        for boundary in self._plane_boundaries:
            self._add_geometry_to_vis(boundary)

    def _update_bounding_box(self):
        """Remove bounding box and get new one for current element."""
        from pyShapeDetector.geometry import (
            LineSet,
            OrientedBoundingBox,
            AxisAlignedBoundingBox,
        )

        bbox_expand = self._preferences["bbox_expand"]
        self.print_debug("Updating bounding box...", require_verbose=True)

        vis = self._vis
        if vis is None:
            return

        if self._bbox is not None:
            self._remove_geometry_from_vis(self._bbox)

        with warnings.catch_warnings():
            if self._started:
                warnings.simplefilter("ignore")
            element = self.current_element

        if element is None or isinstance(element, LineSet):
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                bbox_original = element["raw"].get_oriented_bounding_box()
                bbox = OrientedBoundingBox(bbox_original).expanded(bbox_expand)
            except Exception:
                bbox_original = element["raw"].get_axis_aligned_bounding_box()
                bbox = AxisAlignedBoundingBox(bbox_original).expanded(bbox_expand)

        if self.is_current_selected:
            bbox.color = self.color_bbox_selected
        else:
            bbox.color = self.color_bbox_unselected

        self._bbox = bbox.as_open3d

        if self._bbox is not None:
            self._add_geometry_to_vis(self._bbox)

    def _update_element(self, idx, update_vis=True):
        num_elems = len(self.elements)
        if num_elems == 0 or idx not in range(num_elems):
            warnings.warn(
                "Tried to update index {idx}, but {num_elems} elements present."
            )
            return

        elem = self.elements[idx]

        if update_vis:
            self._remove_geometry_from_vis(elem["drawable"])

        is_selected = elem["selected"] and self.select_filter(elem)
        is_current = self._started and (idx == self.i)

        self.print_debug(
            "[_update_element]"
            f"Updating element at index: {idx}.\n"
            f"Selected = {is_selected}, current = {is_current}.",
            require_verbose=True,
        )

        if self._preferences["paint_selected"] and is_selected:
            color = self._colors_selected_current[True, is_current]
            elem["drawable"] = self._get_painted_element(elem["drawable"], color)
            self.print_debug(
                f"[_update_element] Painting drawable to color: {color}.",
                require_verbose=True,
            )
        else:
            highlight_offset = self._preferences["highlight_color_multiplier"] * (
                int(is_selected) + int(is_current)
            )
            color = elem["color"] + highlight_offset
            self._set_element_colors(elem["drawable"], color)

        if update_vis:
            self.print_debug(
                "[_update_element] Adding to geometry.", require_verbose=True
            )
            self._add_geometry_to_vis(elem["drawable"])

    def _update_current_idx(self, idx=None, update_old=True):
        if idx is not None:
            self.i_old = self.i
            self.i = idx
            self.print_debug(
                f"Updating index, from {self.i_old} to {self.i}", require_verbose=True
            )
        else:
            self.print_debug(f"Updating current index: {self.i}", require_verbose=True)

        if self.i >= len(self.elements):
            warnings.warn(
                f"Index error, tried accessing {self.i} out of "
                f"{len(self.elements)} elements. Getting last one."
            )
            idx = len(self.elements) - 1

        self._update_element(self.i)
        if update_old:
            self._update_element(self.i_old)
        self._update_bounding_box()

    def _get_range(self, indices_or_slice):
        if isinstance(indices_or_slice, (range, list, np.ndarray)):
            return indices_or_slice

        if indices_or_slice is None:
            # if indices are not given, update everything
            return range(len(self.elements))

        if isinstance(indices_or_slice, slice):
            start, stop, stride = indices_or_slice.indices(len(self.elements))
            return range(start, stop, stride)

        raise ValueError("Invalid input, expected index list/array, range or slice.")

    # TODO: clean this up
    # def _update_indices(self, indices_or_slice=None):
    #     indices = self._get_range(indices_or_slice)

    #     i_old = self.i

    #     for i in indices:
    #         self._update_current_idx(i, update_old=self._started)
    #     self._update_current_idx(i_old, update_old=self._started)

    def _toggle_indices(self, indices_or_slice):
        indices = self._get_range(indices_or_slice)

        selectable = [self.select_filter(elem) for elem in self.elements]

        selected = np.logical_or(self.selected, ~np.asarray(selectable))
        selected[indices] = not np.sum(selected[indices]) == len(selected[indices])
        self.selected = selected.tolist()

        # self._update_indices(indices)
        # TODO: this used to be self._update_indices. Remove?
        i_old = self.i
        for i in indices:
            self._update_current_idx(i, update_old=self._started)
        self._update_current_idx(i_old, update_old=self._started)

    def _apply_function_to_elements(self, func, action):
        """Apply function to selected elements."""
        if not self.extra_functions or action == 1 or self.functions is None:
            return

        indices = self.selected_indices
        self.print_debug(
            f"Applying {func.__name__} function to {len(indices)} elements, indices: "
        )
        self.print_debug(indices)
        input_elements = [self.elements[i]["raw"] for i in indices]

        try:
            output_elements = func(input_elements)
        except KeyboardInterrupt:
            return
        except Exception as e:
            warnings.warn(
                f"Failed to apply {func.__name__} function to "
                f"elements in indices {indices}, got following error: {str(e)}"
            )
            time.sleep(0.5)
            return

        # assures it's a list
        if isinstance(output_elements, tuple):
            output_elements = list(output_elements)
        elif not isinstance(output_elements, list):
            output_elements = [output_elements]

        self._save_state(indices, input_elements, len(output_elements))
        assert self._pop_elements(indices, from_vis=True) == input_elements
        self._insert_elements(output_elements, to_vis=True)

        self._future_states = []
        self._update_get_plane_boundaries()

    ###########################################################################
    ################### Callback functions for Visualizer #####################
    ###########################################################################
    def _cb_toggle(self, vis, action, mods):
        """Toggle the current highlighted element between selected/unselected."""
        if action == 1 or not self.select_filter(self.current_element):
            return

        self.is_current_selected = not self.is_current_selected
        self._update_current_idx()

    def _cb_toggle_all(self, vis, action, mods):
        """Toggle the all elements between all selected/all unselected."""
        if not self.extra_functions or action == 1:
            return

        self._toggle_indices(None)

    def _cb_toggle_last(self, vis, action, mods):
        """Toggle the elements from last output."""
        if not self.extra_functions or action == 1:
            return

        if len(self._past_states) == 0:
            return

        num_outputs = self._past_states[-1]["num_outputs"]
        self._toggle_indices(slice(-num_outputs, None))

    def _cb_toggle_type(self, vis, action, mods):
        if not self.extra_functions or action == 1:
            return

        from .helpers_visualization import get_inputs

        elems_raw = [elem["raw"] for elem in self.elements]

        types = set([t for elem in elems_raw for t in elem.__class__.mro()])
        types.discard(ABC)
        types.discard(object)
        types = list(types)
        types_names = [type_.__name__ for type_ in types]

        try:
            (selected_type_name,) = get_inputs(
                {"type": [types_names, types_names[0]]}, window_name="Select type"
            )
        except KeyboardInterrupt:
            return

        selected_type = types[types_names.index(selected_type_name)]
        idx = np.where([isinstance(elem, selected_type) for elem in elems_raw])[0]
        self._toggle_indices(idx)

    def _cb_next(self, vis, action, mods):
        """Highlight next element in list."""
        if action == 1:
            return
        self._update_current_idx(min(self.i + 1, len(self.elements) - 1))

    def _cb_previous(self, vis, action, mods):
        """Highlight previous element in list."""
        if action == 1:
            return
        self._update_current_idx(max(self.i - 1, 0))

    def _cb_finish_process(self, vis, action, mods):
        """Signal ending."""
        if action == 1:
            return
        self.finish = True
        vis.close()

    def _cb_switch_extra(self, vis, action, mods):
        """Switch to mouse selection + extra LCtrl functions."""
        if self.extra_functions == bool(action):
            return

        self.extra_functions = bool(action)

        if self.extra_functions:
            vis.register_mouse_button_callback(self.on_mouse_button)
            vis.register_mouse_move_callback(self.on_mouse_move)
        else:
            vis.register_mouse_button_callback(None)
            vis.register_mouse_move_callback(None)

    def _cb_switch_mouse_toggle(self, vis, action, mods):
        """Switch to mouse selection + extra LCtrl mode."""
        self.mouse_toggle = bool(action)

    def _cb_print_help(self, vis, action, mods):
        if not self.extra_functions or action == 1:
            return

        print(self._instructions)
        time.sleep(0.5)

    def _cb_print_info(self, vis, action, mods):
        if not self.extra_functions or action == 1:
            return

        elem = self.current_element

        print()
        print(f"Current element: ({self.i + 1}/{len(self.elements)}): {elem['raw']}")
        self.print_debug(f"drawable: {elem['drawable']}")
        self.print_debug(f"Current index: {self.i}, old index: {self.i_old}")
        print(f"Current selected: {self.is_current_selected}")
        print(f"Current bbox: {self._bbox}")
        print(f"{len(self.elements)} current elements")
        print(f"{len(self._fixed_elements)} fixed elements")
        print(f"{len(self._hidden_elements)} hidden elements")
        print(f"{len(self._past_states)} past states (for undoing)")
        print(f"{len(self._future_states)} future states (for redoing)")
        time.sleep(0.5)

    def _cb_center_current(self, vis, action, mods):
        if not self.extra_functions or action == 1:
            return

        ctr = self._vis.get_view_control()
        ctr.set_lookat(self._bbox.get_center())
        ctr.set_front([0, 0, -1])  # Define the camera front direction
        ctr.set_up([0, 1, 0])  # Define the camera "up" direction
        ctr.set_zoom(0.1)  # Adjust zoom level if necessary

    def _cb_functions_menu(self, vis, action, mods):
        if not self.extra_functions or action == 1:
            return

        if self.functions is None or len(self.functions) == 0:
            warnings.warn("No functions, cannot call menu.")
            return

        from .helpers_visualization import select_function_with_gui

        try:
            func = select_function_with_gui(self.functions)
        except KeyboardInterrupt:
            return

        if func is not None:
            self._apply_function_to_elements(func, action)

    def _cb_hide_unhide(self, vis, action, mods):
        """Hide selected elements or unhide all hidden elements."""
        if not self.extra_functions or action == 1:
            return

        indices = self.selected_indices
        num_elements = len(indices)

        if num_elements == 0:
            self._insert_elements(self._hidden_elements, selected=True, to_vis=True)
            self._hidden_elements = []

        else:
            self._hidden_elements += self._pop_elements(indices, from_vis=True)
            self.selected = False

        # TODO: find a way to make hiding work with undoing
        self._past_states = []
        self._future_states = []
        self._update_get_plane_boundaries()

    def _cb_set_color_mode(self, vis, action, mods):
        if action == 1:
            return

        self._preferences["paint_random"] = not self._preferences["paint_random"]
        self._reset_visualiser_elements()

    def _cb_set_preferences(self, vis, action, mods):
        if action == 1:
            return

        from .helpers_visualization import get_inputs

        try:
            new_preferences = get_inputs(
                {
                    name: (type(value), value)
                    for name, value in self._preferences.items()
                },
                window_name="Set preferences",
                as_dict=True,
            )

            new_preferences["random_color_multiplier"] = np.clip(
                new_preferences["random_color_multiplier"], 0, 1
            )

        except KeyboardInterrupt:
            return

        if new_preferences != self._preferences:
            self._preferences = new_preferences
            self._reset_visualiser_elements()

    def _cb_redo(self, vis, action, mods):
        if not self.extra_functions or action == 1 or len(self._future_states) == 0:
            return

        future_state = self._future_states.pop()

        modified_elements = future_state["modified_elements"]
        indices = future_state["indices"]

        self.print_debug(
            f"Redoing last operation, removing {len(indices)} inputs and "
            f"resetting {len(modified_elements)} inputs."
        )

        input_elements = [self.elements[i]["raw"] for i in indices]
        self._save_state(indices, input_elements, len(modified_elements))

        self.i = future_state["current_index"]
        self._pop_elements(indices, from_vis=True)
        self._insert_elements(modified_elements, to_vis=True)

        self._update_current_idx(len(self.elements) - 1)
        self._update_get_plane_boundaries()

    def _cb_undo(self, vis, action, mods):
        if not self.extra_functions or action == 1 or len(self._past_states) == 0:
            return

        last_state = self._past_states.pop()
        indices = last_state["indices"]
        elements = last_state["elements"]
        num_outputs = last_state["num_outputs"]
        num_elems = len(self.elements)

        self.print_debug(
            f"Undoing last operation, removing {num_outputs} outputs and "
            f"resetting {len(elements)} inputs."
        )

        indices_to_pop = range(num_elems - num_outputs, num_elems)
        modified_elements = self._pop_elements(indices_to_pop, from_vis=True)
        self._insert_elements(elements, indices, selected=True, to_vis=True)

        self._future_states.append(
            {
                "modified_elements": modified_elements,
                "indices": indices,
                "current_index": self.i,
            }
        )

        while len(self._future_states) > self._preferences["number_redo_states"]:
            self._future_states.pop(0)

        self._update_current_idx(indices[-1])
        self._update_get_plane_boundaries()

    def on_mouse_move(self, vis, x, y):
        """Get mouse position."""
        self.mouse_position = (int(x), int(y))

    def on_mouse_button(self, vis, button, action, mods):
        if action == 1:
            return

        point, vector = _unproject_screen_to_world(vis, *self.mouse_position)

        distances = self.distances_to_point(point, vector)

        i_min_distance = np.argmin(distances)
        if distances[i_min_distance] is np.inf:
            return

        self.i_old = self.i
        self.i = np.argmin(distances)
        if self.mouse_toggle:
            self._cb_toggle(vis, 0, None)
        self._update_current_idx()

    def _reset_visualiser_elements(self, startup=False, reset_fixed=False):
        """Prepare elements for visualization"""

        if not startup:
            self.i = min(self.i, len(self.elements) - 1)

        if startup or reset_fixed:
            for elem in self._fixed_elements_drawable:
                self._remove_geometry_from_vis(elem)

            self._fixed_elements_drawable = [
                self._get_open3d(elem) for elem in self._fixed_elements
            ]

            for elem in self._fixed_elements_drawable:
                self._add_geometry_to_vis(elem)

        if startup:
            current_idx = 0
            elems_raw = self._elements_input
            pre_selected = self._pre_selected

        else:
            # pre_selected = [False] * len(elems_raw)
            current_idx = copy.copy(self.i)
            pre_selected = self.selected
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No elements left")
                elems_raw = self._pop_elements(range(len(self.elements)), from_vis=True)
            assert len(self.elements) == 0

        self.print_debug(
            f"\ninserting elements at startup, there are {len(elems_raw)}.",
            require_verbose=True,
        )
        self._insert_elements(elems_raw, selected=pre_selected, to_vis=True)
        self.print_debug(f"Finished inserting {len(self.elements)} elements.")

        self._update_get_plane_boundaries()
        self._update_bounding_box()

        self._update_current_idx(current_idx)
        self._started = True

    def run(self, print_instructions=True):
        self.print_debug("Starting ElementSelector instance.")

        if len(self._elements_input) == 0:
            raise RuntimeError("No elements added!")

        # Ensure proper format of inputs, check elements and raise warnings
        self._check_and_initialize_inputs()

        # Set up the visualizer
        self._get_visualizer()
        self._reset_visualiser_elements(startup=True)

        self._instructions = (
            # "**************************************************"
            # + "\nStarting manual selector. Instructions:"
            # + "\nGreen: selected. White: unselected. Blue: current."
            # + "\n******************** KEYS: ***********************\n"
            "******************** KEYS: ***********************\n"
            + "\n".join([f"({key}) {desc}" for desc, (key, _) in KEYS_NORMAL.items()])
            + f"\n({EXTRA_KEY[0]}) Enables mouse selection"
            + "\n"
            + "\n".join(
                [
                    f"({EXTRA_KEY[0]}) + ({key}) {desc}"
                    for desc, (key, _) in KEYS_EXTRA.items()
                ],
            )
            # adding one line for each function
            + "\n"
            + "\n".join(
                [
                    f"({EXTRA_KEY[0]}) + ({chr(key)}) {func.__name__}"
                    for key, func in self.function_key_mappings.items()
                ],
            )
            + "\n**************************************************"
        )

        if print_instructions:
            print(self._instructions)
            time.sleep(0.2)

        try:
            self._vis.run()
            # add hidden elements back to elements list
            self._insert_elements(self._hidden_elements)
        except Exception as e:
            raise e
        finally:
            self._vis.close()
            self._vis.destroy_window()
