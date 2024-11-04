#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tur Oct 17 13:17:55 2024

@author: ebernardes
"""
import copy
import signal
import sys
import warnings
import inspect
import numpy as np
from open3d import visualization
# from .helpers_visualization import get_painted
# from pyShapeDetector.geometry import OrientedBoundingBox, TriangleMesh, PointCloud
# from pyShapeDetector.primitives import Primitive


ELEMENTS_NUMBER_WARNING = 350
NUM_POINTS_FOR_DISTANCE_CALC = 30

COLOR_BBOX_SELECTED = (0, 0.8, 0)
COLOR_BBOX_UNSELECTED = (1, 0, 0)
COLOR_SELECTED = (0, 0.4, 0)
COLOR_SELECTED_CURRENT = COLOR_BBOX_SELECTED
COLOR_UNSELECTED = (0.9, 0.9, 0.9)
COLOR_UNSELECTED_CURRENT = (0.0, 0.0, 0.6)

KEYS_NORMAL = {
    "Toggle": ["Space", 32],  # GLFW_KEY_SPACE = 32
    "Previous": ["<-", 263],  # GLFW_KEY_LEFT = 263
    "Next": ["->", 262],  # GLFW_KEY_RIGHT = 262
    "Finish": ["F", ord("F")],
}

EXTRA_KEY = ["LCtrl", 341]  # GLFW_KEY_LCTRL = 341

KEYS_EXTRA = {
    "Apply": ["Enter", 257],  # GLFW_KEY_ENTER = 257
    "Undo": ["Z", ord("Z")],
    "Redo": ["Y", ord("Y")],
    "Toggle all": ["A", ord("A")],
    "Toggle click": ["LShift", 340],  # GLFW_KEY_LSHIFT = 340
}


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


class ElementSelector:
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
    bbox_expand : float, optional
        Expands bounding boxes in all directions with this value. Default: 0.0.
    window_name : str, optional
        Name of window. If empty, just gives the number of elements. Default: "".
    paint_selected : boolean, optional
        If True, paint selected elements, and not only their bounding boxes.
        Default: True
    draw_boundary_lines : boolean, optional
        If True, draws the boundaries of planes as LineSets. Default: False.
    return_finish_flag : boolean, optional
        Should be deprecated.
    mesh_show_back_face : optional, boolean
        If True, shows back faces of surfaces and meshes. Default: True.
    """

    def __init__(
        self,
        bbox_expand=0.0,
        window_name="",
        paint_selected=True,
        draw_boundary_lines=False,
        return_finish_flag=False,
        **camera_options,
    ):
        self._past_states = []
        self._future_states = []
        self._elements = []
        self._elements_as_open3d = []
        self._colors = []
        self._fixed_elements = []
        self._elements_painted = []
        self._selected = []
        self._selectable = []
        self._bbox = None
        self._function = None
        self._select_filter = None
        self.bbox_expand = bbox_expand
        self.paint_selected = paint_selected
        self.window_name = window_name
        self.return_finish_flag = return_finish_flag
        self.draw_boundary_lines = draw_boundary_lines
        self._ELEMENTS_NUMBER_WARNING = ELEMENTS_NUMBER_WARNING
        self._NUM_POINTS_FOR_DISTANCE_CALC = NUM_POINTS_FOR_DISTANCE_CALC
        self.camera_options = camera_options

        # self.selected = self.pre_selected.copy()
        self._elements_distance = []
        self.i_old = 0
        self.i = 0
        self.finish = False
        self.extra_functions = False
        self.mouse_toggle = False
        self.mouse_position = (0, 0)

        self._vis = None

        # Set up colors and elements
        self.color_bbox_selected = COLOR_BBOX_SELECTED
        self.color_bbox_unselected = COLOR_BBOX_UNSELECTED
        self.color_selected = COLOR_SELECTED
        self.color_selected_current = COLOR_SELECTED_CURRENT
        self.color_unselected = COLOR_UNSELECTED
        self.color_unselected_current = COLOR_UNSELECTED_CURRENT

        # (selected, current) -> color
        self._colors_selected_current = {
            (False, False): self.color_unselected,
            (False, True): self.color_unselected_current,
            (True, False): self.color_selected,
            (True, True): self.color_selected_current,
        }

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, new_function):
        if new_function is None:
            pass
        elif (N := len(inspect.signature(new_function).parameters)) != 1:
            raise ValueError(
                f"Expected function with 1 parameter (list of elements), got {N}."
            )
        self._function = new_function

    @property
    def select_filter(self):
        return self._select_filter

    @select_filter.setter
    def select_filter(self, new_function):
        if new_function is None:
            pass
        elif (N := len(inspect.signature(new_function).parameters)) != 1:
            raise ValueError(
                f"Expected filter function with 1 parameter (element), got {N}."
            )
        self._select_filter = new_function

    @property
    def elements(self):
        return self._elements

    def pop_element(self, idx):
        self._elements_as_open3d.pop(idx)
        return self._elements.pop(idx)

    def insert_element(self, idx, elem):
        self._elements_as_open3d.insert(idx, self._get_open3d(elem))
        self._elements.insert(idx, elem)

    def append_elements(self, elems):
        self._elements_as_open3d += [self._get_open3d(elem) for elem in elems]
        self._elements += elems

    @property
    def fixed_elements(self):
        return self._fixed_elements

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, selected_values):
        if isinstance(selected_values, bool):
            selected_values = [selected_values] * len(self._elements)
        elif len(selected_values) != len(self.elements):
            raise ValueError(
                "Length of input expected to be the same as the "
                f"current number of elements ({len(self.elements)}), "
                f"got {len(selected_values)}."
            )

        for value in selected_values:
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"Expected boolean, got {type(value)}")

        if self._selectable is not None and len(self._selectable) == len(
            selected_values
        ):
            for i in np.where(~self._selectable)[0]:
                selected_values[i] = False

        self._selected = copy.deepcopy(selected_values)

    @property
    def current_element(self):
        return self.elements[self.i]

    @property
    def is_current_selected(self):
        return self.selected[self.i]

    @is_current_selected.setter
    def is_current_selected(self, boolean_value):
        if not isinstance(boolean_value, bool):
            raise RuntimeError()
        self.selected[self.i] = boolean_value

    @property
    def all_drawable_elements(self):
        return self._fixed_elements + self._plane_boundaries + self._elements_painted
        # return self._fixed_elements + self._elements_as_open3d + self._plane_boundaries

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

        return [_distance_to_point(elem) for elem in self._elements_distance]

    def add_elements(self, elem, fixed=False):
        if isinstance(elem, (list, tuple)):
            new_elements = copy.deepcopy(list(elem))
        else:
            new_elements = [copy.deepcopy(elem)]

        if fixed:
            self._fixed_elements += new_elements
        else:
            self._elements += new_elements
            self._selected += [False] * len(new_elements)

    def remove_element(self, idx):
        del self._elements[idx]
        del self._selected[idx]

    def _check_and_initialize_inputs(self):
        from pyShapeDetector.geometry import PointCloud

        # check correct input of elements and fixed elements
        if isinstance(self._elements, tuple):
            self._elements = list(self._elements)
        elif not isinstance(self._elements, list):
            self._elements = [self._elements]

        for elem in self._elements:
            if PointCloud.is_instance_or_open3d(elem) and not elem.has_normals():
                elem.estimate_normals()

        if self._fixed_elements is None:
            self._fixed_elements = []
        elif isinstance(self._fixed_elements, tuple):
            self._fixed_elements = list(self._elements)
        elif not isinstance(self._fixed_elements, list):
            self._fixed_elements = [self._fixed_elements]

        if (L := len(self._elements)) == 0:
            raise ValueError("Elements cannot be an empty list.")

        if L > self._ELEMENTS_NUMBER_WARNING:
            warnings.warn(
                f"There are {L} elements, this is too many and may "
                "cause segmentation fault errors."
            )

        if len(self.selected) != len(self.elements):
            raise ValueError("Pre-select and input elements must have same length.")

        if "mesh_show_back_face" not in self.camera_options:
            self.camera_options["mesh_show_back_face"] = True

        # IMPORTANT: respect order, only get Open3D elements at the very end
        self._colors = self._get_colors(self._elements)
        self._fixed_elements = [self._get_open3d(elem) for elem in self._fixed_elements]
        self._elements_as_open3d = [self._get_open3d(elem) for elem in self._elements]

    def _get_drawable_and_painted_elements(self):
        # self._elements_drawable = [self._get_open3d(elem) for elem in self._elements]
        # painted = self._get_painted(self._elements_as_open3d, self.color_unselected)

        if self.paint_selected:
            mask = np.array(self.selected)[np.newaxis].T
            colors = self.color_selected * mask + self.color_unselected * ~mask

            if self.is_current_selected:
                colors[self.i] = self.color_selected_current
            else:
                colors[self.i] = self.color_unselected_current

            self._elements_painted = [
                self._get_painted(elem, color)
                for (elem, color) in zip(self._elements_as_open3d, colors)
            ]
        else:
            self._elements_painted = copy.copy(self._elements_as_open3d)

        # if self

        # if self.paint_selected:
        #     for i in range(len(painted)):
        #         if i == self.i:
        #             if self.is_current_selected:
        #                 color = self.color_selected_current
        #             else:
        #                 color = self.color_unselected_current

        #         elif self.selected[i]:
        #             color = self.color_selected
        #         else:
        #             continue

        #         painted[i] = self._get_painted(painted[i], color)

        # else:
        #     painted[self.i] = self._elements_as_open3d[self.i]

        # self._elements_painted = painted

    def _get_plane_boundaries(self):
        if not self.draw_boundary_lines:
            self._plane_boundaries = []

        plane_boundaries = []
        for element in self._elements:
            try:
                lineset = element.vertices_LineSet.as_open3d
                if hasattr(element, "holes"):
                    for hole in element.holes:
                        lineset += hole.vertices_LineSet.as_open3d
            except AttributeError:
                continue
            lineset.paint_uniform_color((0, 0, 0))
            plane_boundaries.append(lineset)
        self._plane_boundaries = plane_boundaries

    def _get_open3d(self, elem):
        from pyShapeDetector.geometry import TriangleMesh

        try:
            elem_new = elem.as_open3d
            mesh_show_back_face = self.camera_options["mesh_show_back_face"]
            if mesh_show_back_face and TriangleMesh.is_instance_or_open3d(elem_new):
                mesh = TriangleMesh(elem_new)
                mesh.add_reverse_triangles()
                elem_new = mesh.as_open3d

        except Exception:
            elem_new = elem

        return elem_new

    def _get_colors(self, elements):
        colors = []
        for elem in elements:
            if hasattr(elem, "color"):
                colors.append(elem.color)
            elif hasattr(elem, "colors"):
                colors_mean = np.asarray(elem.colors).mean(axis=0)
                colors.append(colors_mean)
            else:
                warnings.warn("Element with no color found.")
                colors.append(np.random.random(3))
        return colors

    def _get_painted(self, elements, color):
        from .helpers_visualization import get_painted

        return get_painted(elements, color)

    def _get_element_distances(self):
        from pyShapeDetector.primitives import Primitive
        from pyShapeDetector.geometry import TriangleMesh, PointCloud

        elements_distance = []
        for elem in self._elements:
            if isinstance(elem, Primitive):
                elements_distance.append(elem)

            # assert our PointCloud class instead of Open3D PointCloud class
            elif TriangleMesh.is_instance_or_open3d(elem):
                pcd = elem.sample_points_uniformly(self._NUM_POINTS_FOR_DISTANCE_CALC)
                elements_distance.append(PointCloud(pcd))

            elif PointCloud.is_instance_or_open3d(elem):
                pcd = elem.uniform_down_sample(self._NUM_POINTS_FOR_DISTANCE_CALC)
                elements_distance.append(PointCloud(pcd))

            else:
                elements_distance.append(None)

        self._elements_distance = elements_distance

    def _get_visualizer(self):
        vis = visualization.VisualizerWithKeyCallback()

        # Register signal handler to gracefully stop the program
        signal.signal(signal.SIGINT, self._signal_handler)

        # Normal keys
        vis.register_key_action_callback(KEYS_NORMAL["Toggle"][1], self.toggle)
        vis.register_key_action_callback(KEYS_NORMAL["Next"][1], self.next)
        vis.register_key_action_callback(KEYS_NORMAL["Previous"][1], self.previous)
        vis.register_key_action_callback(KEYS_NORMAL["Finish"][1], self.finish_process)
        vis.register_key_action_callback(EXTRA_KEY[1], self.switch_extra)
        vis.register_key_action_callback(
            KEYS_EXTRA["Toggle click"][1], self.switch_mouse_toggle
        )

        # Extra (LCtrl) keys
        vis.register_key_action_callback(KEYS_EXTRA["Toggle all"][1], self.toggle_all)
        vis.register_key_action_callback(KEYS_EXTRA["Undo"][1], self.undo)
        vis.register_key_action_callback(KEYS_EXTRA["Redo"][1], self.redo)
        vis.register_key_action_callback(KEYS_EXTRA["Apply"][1], self.apply_function)

        window_name = self.window_name
        if window_name != "":
            window_name += " - "

        window_name += f"{len(self.elements)} elements. "

        vis.create_window(window_name)

        self._vis = vis

    def _signal_handler(self, sig, frame):
        self._vis.destroy_window()
        self._vis.close()
        sys.exit(0)

    def update_all(self, vis):
        idx = min(self.i, len(self._elements) - 1)
        # value = not np.sum(self.selected) == (L := len(self.selected))
        for i in range(len(self._elements)):
            # self.selected[i] = value
            self.update(vis, i)
        self.update(vis, idx)

    def _update_bounding_box(self):
        """Remove bounding box and get new one for current element."""
        from pyShapeDetector.geometry import (
            LineSet,
            OrientedBoundingBox,
            AxisAlignedBoundingBox,
        )

        vis = self._vis
        if vis is None:
            return

        if self._bbox is not None:
            vis.remove_geometry(self._bbox, reset_bounding_box=False)

        element = self.current_element

        if element is None or isinstance(element, LineSet):
            return

        try:
            bbox_original = element.get_oriented_bounding_box()
            bbox = OrientedBoundingBox(bbox_original).expanded(self.bbox_expand)
        except Exception:
            bbox_original = element.get_axis_aligned_bounding_box()
            bbox = AxisAlignedBoundingBox(bbox_original).expanded(self.bbox_expand)

        if self.is_current_selected:
            bbox.color = self.color_bbox_selected
        else:
            bbox.color = self.color_bbox_unselected

        self._bbox = bbox.as_open3d

        if self._bbox is not None:
            vis.add_geometry(self._bbox, reset_bounding_box=False)

    def _update_element(self, idx):
        vis = self._vis
        element = self._elements_painted[idx]
        vis.remove_geometry(element, reset_bounding_box=False)

        is_selected = self._selected[idx] and self._selectable[idx]
        is_current = idx == self.i

        if self.paint_selected:
            color = self._colors_selected_current[is_selected, is_current]
            element = self._get_painted(element, color)

        self._elements_painted[idx] = element
        vis.add_geometry(element, reset_bounding_box=False)

    def update(self, vis, idx=None):
        if idx is not None:
            self.i_old = self.i
            self.i = idx

        if self.i >= len(self._elements):
            warnings.warn(
                f"Index error, tried accessing {self.i} out of "
                f"{len(self._elements)} elements. Getting last one."
            )
            idx = len(self._elements) - 1

        self._update_element(self.i)
        self._update_element(self.i_old)
        self._update_bounding_box()

    def toggle(self, vis, action, mods):
        """Toggle the current highlighted element between selected/unselected."""
        if action == 1 or not self._selectable[self.i]:
            return

        self.is_current_selected = not self.is_current_selected
        self.update(vis)

    def next(self, vis, action, mods):
        """Highlight next element in list."""
        if action == 1:
            return
        self.update(vis, min(self.i + 1, len(self._elements) - 1))

    def previous(self, vis, action, mods):
        """Highlight previous element in list."""
        if action == 1:
            return
        self.update(vis, max(self.i - 1, 0))

    def finish_process(self, vis, action, mods):
        """Signal ending."""
        if action == 1:
            return
        self.finish = True
        vis.close()

    def switch_extra(self, vis, action, mods):
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

    def switch_mouse_toggle(self, vis, action, mods):
        """Switch to mouse selection + extra LCtrl mode."""
        self.mouse_toggle = bool(action)

    def apply_function(self, vis, action, mods):
        """Apply function to selected elements."""
        if not self.extra_functions or action == 1 or self.function is None:
            return

        indices = np.where(self._selected)[0].tolist()
        input_elements = [self._elements[i] for i in indices]
        output_elements = self.function(input_elements)

        # assures it's a list
        if isinstance(output_elements, tuple):
            output_elements = list(output_elements)
        elif not isinstance(output_elements, list):
            output_elements = [output_elements]

        current_state = {
            "indices": copy.deepcopy(indices),
            "elements": copy.deepcopy(input_elements),
            "num_outputs": len(output_elements),
        }

        self._remove_all_visualiser_elements(vis)
        for n, i in enumerate(indices):
            self._elements.pop(i - n)
            self._elements_as_open3d.pop(i - n)

        self._elements += output_elements
        self._elements_as_open3d += [self._get_open3d(elem) for elem in output_elements]

        if self.i in indices:
            # print("[INFO] Current idx in modified indices, getting last element")
            self.i = max(indices[0] - 1, 0)
        else:
            # print("[INFO] Current idx not modified indices")
            self.i -= len([i for i in indices if i <= self.i])

        if self.i >= len(self._elements):
            warnings.warn(
                f"Index error, tried accessing {self.i} out of "
                f"{len(self._elements)} elements. Getting last one."
            )
            self.i = len(self._elements) - 1

        self._past_states.append(current_state)
        self._future_states = []
        self.selected = False

        self._reset_visualiser_elements(vis)

    def toggle_all(self, vis, action, mods):
        """Toggle the all elements between all selected/all unselected."""
        if not self.extra_functions or action == 1:
            return

        selected = np.logical_or(self.selected, ~self._selectable)

        value = not np.sum(selected) == len(self._elements)
        self.selected = value
        # for i in range(L):
        # self.selected[i] = value

        self.update_all(vis)

    def redo(self, vis, action, mods):
        if not self.extra_functions or action == 1 or len(self._future_states) == 0:
            return

        self._remove_all_visualiser_elements(vis)

        future_state = self._future_states.pop()

        modified_elements = future_state["modified_elements"]
        indices = future_state["indices"]
        self.i = future_state["current_index"]

        input_elements = [self._elements[i] for i in indices]
        current_state = {
            "indices": copy.deepcopy(indices),
            "elements": copy.deepcopy(input_elements),
            "num_outputs": len(modified_elements),
        }

        for n, i in enumerate(indices):
            self.pop_element(i - n)

        self.append_elements(modified_elements)
        self._past_states.append(current_state)
        self.selected = False

        self._reset_visualiser_elements(vis)

    def undo(self, vis, action, mods):
        if not self.extra_functions or action == 1 or len(self._past_states) == 0:
            return

        self._remove_all_visualiser_elements(vis)

        last_state = self._past_states.pop()
        indices = last_state["indices"]
        elements = last_state["elements"]
        num_outputs = last_state["num_outputs"]

        if self.i >= len(self._elements) - num_outputs:
            # print("[INFO] Current idx in modified indices, getting last element")
            self.i = indices[-1]
        else:
            # print("[INFO] Current idx not modified indices")
            self.i += len([i for i in indices if i < self.i])

        if num_outputs > 0:
            modified_elements = self._elements[-num_outputs:]
            self._elements = self._elements[:-num_outputs]
            self._elements_as_open3d = self._elements_as_open3d[:-num_outputs]
        else:
            modified_elements = []

        for i, elem in zip(indices, elements):
            self.insert_element(i, elem)

        self.selected = False
        for i in indices:
            self._selected[i] = True

        self._future_states.append(
            {
                "modified_elements": modified_elements,
                "indices": indices,
                "current_index": self.i,
            }
        )

        self._reset_visualiser_elements(vis)

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
            self.toggle(vis, 0, None)
        self.update(vis)

    def _remove_all_visualiser_elements(self, vis):
        for elem in self.all_drawable_elements:
            if elem is not None:
                try:
                    vis.remove_geometry(elem, reset_bounding_box=False)
                except Exception:
                    pass

    def _reset_visualiser_elements(self, vis, startup=False):
        # Prepare elements for visualization

        self._get_plane_boundaries()
        self._get_drawable_and_painted_elements()
        self._get_element_distances()

        if self.select_filter is None:
            self._selectable = [True] * len(self.elements)
        else:
            self._selectable = [self.select_filter(elem) for elem in self.elements]
        self._selectable = np.asarray(self._selectable)

        for elem in self.all_drawable_elements:
            if elem is not None:
                vis.add_geometry(elem, reset_bounding_box=startup)

        self._update_bounding_box()

        if not startup:
            self.update_all(vis)

    def run(self, print_instructions=True):
        if len(self.elements) == 0:
            raise RuntimeError("No elements added!")

        # Ensure proper format of inputs, check elements and raise warnings
        self._check_and_initialize_inputs()

        # Set up the visualizer
        self._get_visualizer()
        self._reset_visualiser_elements(self._vis, startup=True)

        if print_instructions:
            instructions = (
                "**************************************************"
                + "\nStarting manual selector. Instructions:"
                + "\nGreen: selected. White: unselected. Blue: current."
                + "\n******************** KEYS: ***********************\n"
                + "\n".join(
                    [f"({key}) {desc}" for desc, (key, _) in KEYS_NORMAL.items()]
                )
                + f"\n({EXTRA_KEY[0]}) Enables mouse selection"
                + "\n"
                + "\n".join(
                    [
                        f"({EXTRA_KEY[0]}) + ({key}) {desc}"
                        for desc, (key, _) in KEYS_EXTRA.items()
                    ],
                )
                + "\n**************************************************"
            )
            print(instructions)

        try:
            self._vis.run()
        except Exception as e:
            raise e
        finally:
            self._vis.close()
            self._vis.destroy_window()
