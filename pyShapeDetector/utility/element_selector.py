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
import numpy as np
from open3d import visualization
# from .helpers_visualization import get_painted
# from pyShapeDetector.geometry import OrientedBoundingBox, TriangleMesh, PointCloud
# from pyShapeDetector.primitives import Primitive


ELEMENTS_NUMBER_WARNING = 60
NUM_POINTS_FOR_DISTANCE_CALC = 30

COLOR_BBOX_SELECTED = (0, 0.8, 0)
COLOR_BBOX_UNSELECTED = (1, 0, 0)
COLOR_SELECTED = (0, 0.4, 0)
COLOR_SELECTED_CURRENT = COLOR_BBOX_SELECTED
COLOR_UNSELECTED = (0.9, 0.9, 0.9)
COLOR_UNSELECTED_CURRENT = (0.0, 0.0, 0.6)

KEYS_DESCRIPTOR = {
    "TOGGLE": "S",
    "TOGGLE ALL": "T",
    "NEXT": "D",
    "PREVIOUS": "A",
    "FINISH": "F",
}

KEYS_CONFIG = {k: ord(value) for k, value in KEYS_DESCRIPTOR.items()}

GLFW_KEY_LEFT_SHIFT = 340
GLFW_KEY_LEFT_CONTROL = 341
# GLFW_KEY_ENTER = 257

INSTRUCTIONS = (
    " Green: selected. White: unselected. Blue: current. "
    + " | ".join([f"({k}) {desc.lower()}" for desc, k in KEYS_DESCRIPTOR.items()])
    + " | (LShift) Mouse select + (LCtrl) Toggle"
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
        show_plane_boundaries=False,
        **camera_options,
    ):
        self._elements = []
        self._fixed_elements = []
        self._selected = []
        self.bbox_expand = bbox_expand
        self.paint_selected = paint_selected
        self.window_name = window_name
        self.return_finish_flag = return_finish_flag
        self.show_plane_boundaries = show_plane_boundaries
        self.ELEMENTS_NUMBER_WARNING = ELEMENTS_NUMBER_WARNING
        self.NUM_POINTS_FOR_DISTANCE_CALC = NUM_POINTS_FOR_DISTANCE_CALC
        self.camera_options = camera_options

        # self.selected = self.pre_selected.copy()
        # self._elements_painted = []
        self.elements_distance = []
        self.i_old = 0
        self.i = 0
        self.finish = False
        self.mouse_select = False
        self.mouse_toggle = False
        self.mouse_position = (0, 0)

        self.vis = None

        # Set up colors and elements
        self.color_bbox_selected = COLOR_BBOX_SELECTED
        self.color_bbox_unselected = COLOR_BBOX_UNSELECTED
        self.color_selected = COLOR_SELECTED
        self.color_selected_current = COLOR_SELECTED_CURRENT
        self.color_unselected = COLOR_UNSELECTED
        self.color_unselected_current = COLOR_UNSELECTED_CURRENT

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
        if len(pre_selected_values) != len(self.elements):
            raise ValueError(
                "Length of input expected to be the same as the "
                f"current number of elements ({len(self.elements)}), "
                f"got {len(pre_selected_values)}."
            )

        for value in pre_selected_values:
            if not isinstance(value, bool):
                raise ValueError("Expected boolean, got {type(value)}")

        self._selected = copy.deepcopy(pre_selected_values)

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

        if L > self.ELEMENTS_NUMBER_WARNING:
            warnings.warn(
                f"There are {L} elements, this is too many and may "
                "cause segmentation fault errors."
            )

        # if not isinstance(self._elements, list):
        #     raise ValueError("Input elements must be a list.")

        if len(self.selected) != len(self.elements):
            raise ValueError("Pre-select and input elements must have same length.")

        # if not np.array(self.pre_selected).dtype == bool:
        #     raise ValueError("Pre-select must be a list of booleans.")

        if "mesh_show_back_face" not in self.camera_options:
            self.camera_options["mesh_show_back_face"] = True

        self._bboxes = self._get_bboxes(self._elements, (1, 0, 0))
        self._fixed_bboxes = self._get_bboxes(self._fixed_elements, (0, 0, 0))

        if self.show_plane_boundaries:
            self._fixed_elements += self._add_plane_boundaries()

        # IMPORTANT: respect order, only get Open3D elements at the very end
        self._elements = [self._get_open3d(elem) for elem in self._elements]
        self._fixed_elements = [self._get_open3d(elem) for elem in self._fixed_elements]

    def _paint_elements(self):
        # self._bboxes = self._get_bboxes(self._elements, (1, 0, 0))
        # self._fixed_bboxes = self._get_bboxes(self._fixed_elements, (0, 0, 0))

        painted = self._get_painted(self._elements, self.color_unselected)

        if self.paint_selected:
            painted[0] = self._get_painted(painted[0], self.color_unselected_current)
        else:
            painted[0] = self._elements[0]

        self._elements_painted = painted

    def _add_plane_boundaries(self):
        plane_boundaries = []
        for element in self._elements:
            try:
                lineset = element.vertices_LineSet
                if hasattr(element, "holes"):
                    for hole in element.holes:
                        lineset += hole.vertices_LineSet
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

        distances = []
        for elem in self._elements:
            if isinstance(elem, Primitive):
                distances.append(elem)
            elif TriangleMesh.is_instance_or_open3d(elem):
                distances.append(
                    elem.sample_points_uniformly(self.NUM_POINTS_FOR_DISTANCE_CALC)
                )
            elif PointCloud.is_instance_or_open3d(elem):
                distances.append(
                    elem.uniform_down_sample(self.NUM_POINTS_FOR_DISTANCE_CALC)
                )
            else:
                distances.append(None)
        return distances

    def _get_visualizer(self):
        vis = visualization.VisualizerWithKeyCallback()

        # Register signal handler to gracefully stop the program
        signal.signal(signal.SIGINT, self._signal_handler)

        vis.register_key_action_callback(KEYS_CONFIG["TOGGLE"], self.toggle)
        vis.register_key_action_callback(KEYS_CONFIG["TOGGLE ALL"], self.toggle_all)
        vis.register_key_action_callback(KEYS_CONFIG["NEXT"], self.next)
        vis.register_key_action_callback(KEYS_CONFIG["PREVIOUS"], self.previous)
        vis.register_key_action_callback(KEYS_CONFIG["FINISH"], self.finish_process)
        vis.register_key_action_callback(
            GLFW_KEY_LEFT_SHIFT, self.switch_mouse_selection
        )
        vis.register_key_action_callback(
            GLFW_KEY_LEFT_CONTROL, self.switch_mouse_toggle
        )

        window_name = self.window_name
        if window_name != "":
            window_name += " - "

        window_name += f"{len(self.elements)} elements. " + INSTRUCTIONS

        vis.create_window(window_name)

        return vis

    def _signal_handler(self, sig, frame):
        self.vis.destroy_window()
        self.vis.close()
        sys.exit(0)

    def update(self, vis, idx=None):
        if idx is not None:
            self.i_old = self.i
            self.i = idx

        element = self._elements_painted[self.i_old]
        vis.remove_geometry(element, reset_bounding_box=False)
        vis.remove_geometry(self._bboxes[self.i_old], reset_bounding_box=False)

        if not self.selected[self.i_old]:
            self._bboxes[self.i_old].color = self.color_bbox_unselected
            element = self._get_painted(element, self.color_unselected)
        else:
            self._bboxes[self.i_old].color = self.color_bbox_selected
            if self.paint_selected:
                element = self._get_painted(element, self.color_selected)
            else:
                element = self._elements[self.i_old]

        self._elements_painted[self.i_old] = element
        vis.add_geometry(element, reset_bounding_box=False)

        element = self._elements_painted[self.i]
        vis.remove_geometry(element, reset_bounding_box=False)
        if not self.paint_selected:
            element = self._elements[self.i]
        elif self.selected[self.i]:
            element = self._get_painted(element, self.color_selected_current)
        else:
            element = self._get_painted(element, self.color_unselected_current)
        self._elements_painted[self.i] = element

        vis.add_geometry(element, reset_bounding_box=False)
        vis.add_geometry(self._bboxes[self.i], reset_bounding_box=False)

    def toggle(self, vis, action, mods):
        if action == 1:
            return
        self.selected[self.i] = not self.selected[self.i]
        self.update(vis)

    def toggle_all(self, vis, action, mods):
        if action == 1:
            return

        idx = self.i

        value = not np.sum(self.selected) == (L := len(self.selected))
        for i in range(L):
            self.selected[i] = value
            self.update(vis, i)
        self.update(vis, idx)

    def next(self, vis, action, mods):
        if action == 1:
            return
        self.update(vis, min(self.i + 1, len(self._elements) - 1))

    def previous(self, vis, action, mods):
        if action == 1:
            return
        self.update(vis, max(self.i - 1, 0))

    def finish_process(self, vis, action, mods):
        if action == 1:
            return
        self.finish = True
        vis.close()

    def switch_mouse_selection(self, vis, action, mods):
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
        self.mouse_toggle = bool(action)

    def on_mouse_move(self, vis, x, y):
        self.mouse_position = (int(x), int(y))

    def on_mouse_button(self, vis, button, action, mods):
        from pyShapeDetector.geometry import PointCloud

        if action == 1:
            return

        point = unproject_screen_to_world(vis, *self.mouse_position)

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

        self.i_old = self.i
        self.i = np.argmin(distances)
        if self.mouse_toggle:
            self.toggle(vis, 0, None)
        self.update(vis)

    def run(self):
        if len(self.elements) == 0:
            raise RuntimeError("No elements added!")

        # Ensure proper format of inputs, check elements and raise warnings
        self._check_and_initialize_inputs()

        # Prepare elements for visualization
        self._paint_elements()
        self.elements_distance = self._compute_element_distances()

        # Set up the visualizer
        vis = self._get_visualizer()

        elements = (
            self._fixed_elements
            + self._fixed_bboxes
            + self._elements_painted
            + [self._bboxes[0]]
        )

        # vis.create_window()

        for elem in elements:
            vis.add_geometry(elem)

        vis.run()
        vis.close()
        vis.destroy_window()
