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

KEYS_DESCRIPTOR = {
    "TOGGLE": ord("S"),
    "TOGGLE ALL": ord("T"),
    "NEXT": ord("D"),
    "PREVIOUS": ord("A"),
    "FINISH": ord("F"),
}

GLFW_LSHIFT = 340
GLFW_LCTRL = 341
GLFW_ENTER = 257

INSTRUCTIONS = (
    " Green: selected. White: unselected. Blue: current. "
    + " | ".join([f"({chr(k)}) {desc.lower()}" for desc, k in KEYS_DESCRIPTOR.items()])
    + " | (LCtrl) Mouse select + (LShift) Toggle"
    + " | (LCtrl) + (Enter) Apply function"
)


def unproject_screen_to_world(vis, x, y):
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
    return point


class ElementSelector:
    def __init__(
        self,
        bbox_expand=0.0,
        window_name="",
        paint_selected=True,
        return_finish_flag=False,
        draw_boundary_lines=False,
        **camera_options,
    ):
        self._past_states = []
        self._elements = []
        self._fixed_elements = []
        self._elements_painted = []
        self._selected = []
        self._function = None
        self.bbox_expand = bbox_expand
        self.paint_selected = paint_selected
        self.window_name = window_name
        self.return_finish_flag = return_finish_flag
        self.draw_boundary_lines = draw_boundary_lines
        self._ELEMENTS_NUMBER_WARNING = ELEMENTS_NUMBER_WARNING
        self._NUM_POINTS_FOR_DISTANCE_CALC = NUM_POINTS_FOR_DISTANCE_CALC
        self.camera_options = camera_options

        # self.selected = self.pre_selected.copy()
        self.elements_distance = []
        self.i_old = 0
        self.i = 0
        self.finish = False
        self.mouse_select = False
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
    def elements(self):
        return self._elements

    @property
    def fixed_elements(self):
        return self._fixed_elements

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, pre_selected_values):
        if isinstance(pre_selected_values, bool):
            pre_selected_values = [pre_selected_values] * len(self._elements)
        elif len(pre_selected_values) != len(self.elements):
            raise ValueError(
                "Length of input expected to be the same as the "
                f"current number of elements ({len(self.elements)}), "
                f"got {len(pre_selected_values)}."
            )

        for value in pre_selected_values:
            if not isinstance(value, bool):
                raise ValueError("Expected boolean, got {type(value)}")

        self._selected = copy.deepcopy(pre_selected_values)

    def distances_to_point(self, point):
        from pyShapeDetector.geometry import PointCloud

        distances = []
        for elem in self.elements_distance:
            if elem is None:
                distances.append(np.inf)
            elif PointCloud.is_instance_or_open3d(elem):
                distances.append(
                    PointCloud([point]).compute_point_cloud_distance(elem)[0]
                )
            else:  # it is a Primitive
                distances.append(elem.get_distances(point))

        return distances

    def add_elements(self, element, fixed=False):
        if isinstance(element, (list, tuple)):
            new_elements = copy.deepcopy(list(element))
        else:
            new_elements = [copy.deepcopy(element)]

        if fixed:
            self._fixed_elements += new_elements
        else:
            self._elements += new_elements
            self._selected += [False] * len(new_elements)

    def remove_element(self, idx):
        del self._elements[idx]
        del self._selected[idx]

    def _check_and_initialize_inputs(self):
        # check correct input of elements and fixed elements
        if isinstance(self._elements, tuple):
            self._elements = list(self._elements)
        elif not isinstance(self._elements, list):
            self._elements = [self._elements]

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
        self._fixed_bboxes = self._get_bboxes(self._fixed_elements, (0, 0, 0))
        self._fixed_elements = [self._get_open3d(elem) for elem in self._fixed_elements]

    def _get_drawable_and_painted_elements(self):
        self._elements_drawable = [self._get_open3d(elem) for elem in self._elements]
        painted = self._get_painted(self._elements_drawable, self.color_unselected)

        if self.paint_selected:
            painted[self.i] = self._get_painted(
                painted[self.i], self.color_unselected_current
            )
        else:
            painted[self.i] = self._elements_drawable[self.i]

        self._elements_painted = painted

    def _get_plane_boundaries(self):
        if not self.draw_boundary_lines:
            return []

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
        return plane_boundaries

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

    def _get_bboxes(self, elements, color):
        from open3d.geometry import LineSet
        from pyShapeDetector.geometry import OrientedBoundingBox

        bboxes = []
        for element in elements:
            if isinstance(element, LineSet):
                continue

            if element is None:
                bbox = None
            else:
                bbox_original = element.get_oriented_bounding_box()
                bbox = OrientedBoundingBox(bbox_original).expanded(self.bbox_expand)
                bbox.color = color
                bbox = bbox.as_open3d
                bboxes.append(bbox)
        return bboxes

    def _get_painted(self, elements, color):
        from .helpers_visualization import get_painted

        return get_painted(elements, color)

    def _compute_element_distances(self):
        from pyShapeDetector.primitives import Primitive
        from pyShapeDetector.geometry import TriangleMesh, PointCloud

        elements_distance = []
        for elem in self._elements:
            if isinstance(elem, Primitive):
                elements_distance.append(elem)
            elif TriangleMesh.is_instance_or_open3d(elem):
                elements_distance.append(
                    elem.sample_points_uniformly(self._NUM_POINTS_FOR_DISTANCE_CALC)
                )
            elif PointCloud.is_instance_or_open3d(elem):
                elements_distance.append(
                    elem.uniform_down_sample(self._NUM_POINTS_FOR_DISTANCE_CALC)
                )
            else:
                elements_distance.append(None)
        return elements_distance

    def _get_visualizer(self):
        vis = visualization.VisualizerWithKeyCallback()

        # Register signal handler to gracefully stop the program
        signal.signal(signal.SIGINT, self._signal_handler)

        vis.register_key_action_callback(KEYS_DESCRIPTOR["TOGGLE"], self.toggle)
        vis.register_key_action_callback(KEYS_DESCRIPTOR["TOGGLE ALL"], self.toggle_all)
        vis.register_key_action_callback(KEYS_DESCRIPTOR["NEXT"], self.next)
        vis.register_key_action_callback(KEYS_DESCRIPTOR["PREVIOUS"], self.previous)
        vis.register_key_action_callback(KEYS_DESCRIPTOR["FINISH"], self.finish_process)
        vis.register_key_action_callback(GLFW_LCTRL, self.switch_mode)
        vis.register_key_action_callback(GLFW_LSHIFT, self.switch_mouse_toggle)
        vis.register_key_action_callback(GLFW_ENTER, self.apply_function)
        vis.register_key_action_callback(ord("Z"), self.undo)

        # vis.register_key_action_callback(
        #     GLFW_KEY_LEFT_SHIFT, self.switch_mouse_selection
        # )
        # vis.register_key_action_callback(
        #     GLFW_KEY_LEFT_CONTROL, self.switch_mouse_toggle
        # )
        # vis.register_key_action_callback(GLFW_KEY_ENTER, self.apply_function)

        window_name = self.window_name
        if window_name != "":
            window_name += " - "

        window_name += f"{len(self.elements)} elements. " + INSTRUCTIONS

        vis.create_window(window_name)

        return vis

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

        # revert current element color to normal color...
        element = self._elements_painted[self.i_old]
        vis.remove_geometry(element, reset_bounding_box=False)
        vis.remove_geometry(self._bboxes[self.i_old], reset_bounding_box=False)
        if not self.selected[self.i_old]:
            # ... if not selected
            self._bboxes[self.i_old].color = self.color_bbox_unselected
            element = self._get_painted(element, self.color_unselected)
        else:
            # ... or if not selected
            self._bboxes[self.i_old].color = self.color_bbox_selected
            if self.paint_selected:
                element = self._get_painted(element, self.color_selected)
            else:
                element = self._elements_drawable[self.i_old]
        self._elements_painted[self.i_old] = element
        vis.add_geometry(element, reset_bounding_box=False)

        # change new current element color to highlighted color...
        element = self._elements_painted[self.i]
        vis.remove_geometry(element, reset_bounding_box=False)
        if not self.paint_selected:
            element = self._elements_drawable[self.i]
        elif self.selected[self.i]:
            element = self._get_painted(element, self.color_selected_current)
        else:
            element = self._get_painted(element, self.color_unselected_current)
        self._elements_painted[self.i] = element
        vis.add_geometry(element, reset_bounding_box=False)
        vis.add_geometry(self._bboxes[self.i], reset_bounding_box=False)

    def toggle(self, vis, action, mods):
        """Toggle the current highlighted element between selected/unselected."""
        if action == 1:
            return
        self.selected[self.i] = not self.selected[self.i]
        self.update(vis)

    def toggle_all(self, vis, action, mods):
        """Toggle the all elements between all selected/all unselected."""
        if action == 1:
            return

        value = not np.sum(self.selected) == len(self._elements)
        self.selected = value
        # for i in range(L):
        # self.selected[i] = value

        self.update_all(vis)

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

    def switch_mode(self, vis, action, mods):
        """Switch to mouse selection + extra LCtrl mode."""
        if self.mouse_select == bool(action):
            return

        self.mouse_select = bool(action)

        if self.mouse_select:
            # print("[Info] Mouse mode: selection")
            vis.register_mouse_button_callback(self.on_mouse_button)
            vis.register_mouse_move_callback(self.on_mouse_move)
        else:
            # print("[Info] Mouse mode: camera control")
            vis.register_mouse_button_callback(None)
            vis.register_mouse_move_callback(None)

    def switch_mouse_toggle(self, vis, action, mods):
        """Switch to mouse selection + extra LCtrl mode."""
        self.mouse_toggle = bool(action)

    def apply_function(self, vis, action, mods):
        """Apply function to selected elements."""
        if not self.mouse_select or action == 1 or self.function is None:
            return

        indices = np.where(self._selected)[0].tolist()
        input_elements = [self._elements[i] for i in indices]
        output_elements = self.function(input_elements)

        current_state = {
            "indices": copy.deepcopy(indices),
            "elements": copy.deepcopy(input_elements),
            "num_outputs": len(output_elements),
        }

        self.remove_all_visualiser_elements(vis)
        for n, i in enumerate(indices):
            self._elements.pop(i - n)
            self._bboxes.pop(i - n)

        self._elements += output_elements
        self._bboxes += self._get_bboxes(output_elements, (1, 0, 0))

        if self.i in indices:
            print("[INFO] Current idx in modified indices, getting last element")
            self.i = len(self._elements) - 1
        else:
            print("[INFO] Current idx not modified indices")
            self.i -= len([i for i in indices if i <= self.i])

        print(f"Current index is {self.i}, {len(self._elements)} elements.")

        if self.i >= len(self._elements):
            warnings.warn(
                f"Index error, tried accessing {self.i} out of "
                f"{len(self._elements)} elements. Getting last one."
            )
            self.i = len(self._elements) - 1

        self._past_states.append(current_state)

        self.reset_visualiser_elements(vis)

    def undo(self, vis, action, mods):
        if not self.mouse_select or action == 1 or len(self._past_states) == 0:
            return

        self.remove_all_visualiser_elements(vis)

        last_state = self._past_states.pop()
        indices = last_state["indices"]
        elements = last_state["elements"]
        num_outputs = last_state["num_outputs"]

        if self.i >= len(self._elements) - num_outputs:
            self.i = indices[-1]
        else:
            self.i += len([i for i in indices if i < self.i])

        self._elements = self._elements[:-num_outputs]
        self._bboxes = self._bboxes[:-num_outputs]

        for i, element in zip(indices, elements):
            self._elements.insert(i, element)
            self._bboxes.insert(i, self._get_bboxes([element], (1, 0, 0))[0])

        self.reset_visualiser_elements(vis)

    def on_mouse_move(self, vis, x, y):
        """Get mouse position."""
        self.mouse_position = (int(x), int(y))

    def on_mouse_button(self, vis, button, action, mods):
        if action == 1:
            return

        point = unproject_screen_to_world(vis, *self.mouse_position)

        distances = self.distances_to_point(point)

        self.i_old = self.i
        self.i = np.argmin(distances)
        if self.mouse_toggle:
            self.toggle(vis, 0, None)
        self.update(vis)

    def remove_all_visualiser_elements(self, vis):
        elements = (
            self._fixed_elements
            + self._fixed_bboxes
            + self._plane_boundaries
            + self._elements_painted
            + self._bboxes
            + [self._bboxes[self.i]]  # TODO: investigate, not sure why this is needed
        )

        for elem in elements:
            if elem is not None:
                try:
                    vis.remove_geometry(elem, reset_bounding_box=False)
                except Exception:
                    pass

    def reset_visualiser_elements(self, vis, startup=False):
        # Prepare elements for visualization
        if startup:
            self._bboxes = self._get_bboxes(self._elements, (1, 0, 0))
        self._plane_boundaries = self._get_plane_boundaries()
        self._get_drawable_and_painted_elements()
        self.elements_distance = self._compute_element_distances()

        # print(
        #     f"{len(self._elements)} _elements, "
        #     f"{len(self._elements_painted)} _elements_painted, "
        #     f"{len(self._elements_drawable)} _elements_drawable, "
        #     f"{len(self._bboxes)} _bboxes, "
        # )

        elements = (
            self._fixed_elements
            + self._fixed_bboxes
            + self._plane_boundaries
            + self._elements_painted
        )

        if startup:
            elements += [self._bboxes[self.i]]

        for elem in elements:
            if elem is not None:
                vis.add_geometry(elem, reset_bounding_box=startup)

        if not startup:
            self.selected = False
            self.update_all(vis)

    def run(self):
        if len(self.elements) == 0:
            raise RuntimeError("No elements added!")

        # Ensure proper format of inputs, check elements and raise warnings
        self._check_and_initialize_inputs()

        # Set up the visualizer
        vis = self._get_visualizer()
        self.reset_visualiser_elements(vis, startup=True)

        try:
            vis.run()
        except Exception as e:
            raise e
        finally:
            vis.close()
            vis.destroy_window()
