#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tur Oct 17 13:17:55 2024

@author: ebernardes
"""
import time
import copy
import inspect
import signal
import sys
import warnings
import inspect
import numpy as np
from open3d.utility import Vector3dVector
from open3d.visualization import gui, rendering

from .settings import Settings
from .menu_functions import MenuFunctions
from .menu_help import MenuHelp
from .hotkeys import Hotkeys
from .helpers import (
    get_pretty_name,
    extract_element_colors,
    set_element_colors,
    get_painted_element,
    get_distance_checker,
)


class AppWindow:
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

    Extra Parameters (these can also be set in real time)
    -----------------------------------------------------
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
    number_points_distance : int, optional
        Number of points in element distance calculator. Default: 30.
    number_undo_states : int, optional
        Number of states to save for undoing. Default: 10.
    number_redo_states : int, optional
        Number of states to save for redoing. Default: 5.
    random_color_brightness : float, optional
        Random colors are multiplied by this value to reduce their brightness,
        creating bigger contrast with selected objects. Default: 2/3.
    highlight_color_brightness : float, optional
        Multiplier for color to show selected and current elements. Default: 0.3.
    debug : boolean, optional
        If True, prints debug information. Default: False.
    verbose : boolean, optional
        If both this and 'debug' are True, prints extra debug information.
        Default: False.


    return_finish_flag : boolean, optional
        Should be deprecated.
    """

    MENU_QUIT = 3
    MENU_SHOW_FUNCTIONS = 2
    NUMBER_SUBFUNCTIONS = 10
    MENU_SUBFUNCTIONS = range(
        MENU_SHOW_FUNCTIONS + 1, MENU_SHOW_FUNCTIONS + 1 + NUMBER_SUBFUNCTIONS
    )
    MENU_SHOW_SETTINGS = 20
    MENU_HELP = 21

    def __init__(
        self,
        window_name="",
        return_finish_flag=False,
        **kwargs,
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
        self._current_bbox = None
        self._functions = None
        self._function_submenu_names = []
        self._last_used_function = None
        self._select_filter = lambda x: True
        self.window_name = window_name
        self.return_finish_flag = return_finish_flag

        # self._elements_distance = []
        self.i_old = 0
        self.i = 0
        self.finish = False
        self._started = False

        self._settings = Settings(self, **kwargs)

    def print_debug(self, text, require_verbose=False):
        is_debug_activated = self._settings.debug
        is_verbose_activated = self._settings.verbose

        if not is_debug_activated or (require_verbose and not is_verbose_activated):
            return

        text = str(text)
        print("[DEBUG] " + text)

    @property
    def functions(self):
        return self._functions

    def add_function(self, function_descriptor):
        parsed_descriptor = {}

        if callable(function_descriptor):
            function = function_descriptor
            parsed_descriptor["function"] = function
            function_descriptor = {"function": function}

        elif not isinstance(function_descriptor, dict):
            raise TypeError("Expected either function or dictionary describing it.")

        elif "function" not in function_descriptor:
            raise AttributeError("Function not found in descriptor.")

        else:
            function = function_descriptor["function"]
            parsed_descriptor["function"] = function_descriptor["function"]

        parsed_descriptor["name"] = function_descriptor.get(
            "name", get_pretty_name(function)
        )

        menu = function_descriptor.get("menu")
        if menu is None:
            menu = MenuFunctions.DEFAULT_MENU_NAME

        parsed_descriptor["menu"] = menu

        hotkey = function_descriptor.get("hotkey", None)
        if hotkey is None:
            pass

        elif not isinstance(hotkey, int) or not (0 <= hotkey <= 9):
            warnings.warn(
                f"Expected integer hotkey between 0 and 9, got {hotkey}. "
                "Ignoring hotkey"
            )
            hotkey = None

        # elif hotkey in used_hotkeys:
        #     warnings.warn(
        #         f"hotkey {hotkey} already assigned to function {used_hotkeys[hotkey]}, "
        #         "resetting it to {parsed_descriptor['name']}."
        #     )

        if hotkey is not None:
            hotkey = ord(str(hotkey))
            if self.functions is not None:
                current_hotkeys = [desc['hotkey'] for desc in self.functions]
                idx = current_hotkeys.index(hotkey)
                if hotkey in current_hotkeys:
                    warnings.warn(
                        f"hotkey {chr(hotkey)} already assigned to function {self.functions[idx]['name']}, "
                        f"resetting it to {parsed_descriptor['name']}."
                    )

        parsed_descriptor["hotkey"] = hotkey

        all_parsed_parameters = {}

        if parameters := function_descriptor.get("parameters", None):
            for name, parameter in parameters.items():
                parsed_parameter = parameter.copy()

                parameter_type = parsed_parameter.get("type", None)

                if parameter_type is None:
                    raise TypeError("Expected explicit parameter type.")

                elif parameter_type is bool:
                    if "default" not in parsed_parameter:
                        parsed_parameter["default"] = False
                    if "limits" in parsed_parameter:
                        warnings.warn(
                            "Unexpected limits in boolean parameter descriptor."
                        )

                elif parameter_type is int or parameter_type is float:
                    if "limits" not in parsed_parameter:
                        raise AttributeError(
                            f"Expected limits in {parameter['type']} parameter descriptor."
                        )

                    limits = parsed_parameter["limits"]

                    if "default" not in parsed_parameter:
                        warnings.warn(
                            f"No default value found in {parameter['type']} "
                            "parameter descriptor, getting mean of limits."
                        )
                        parsed_parameter["default"] = sum(limits) / 2.0

                    else:
                        if not (limits[0] <= parsed_parameter["default"] <= limits[1]):
                            raise ValueError(
                                f"Default value {parsed_parameter['default']} is "
                                f"not within limits {limits}."
                            )

                elif parameter_type is str:
                    if "options" in parsed_parameter:
                        parsed_parameter["options"] = [
                            str(option) for option in parsed_parameter["options"]
                        ]
                else:
                    warnings.warn(f"{parameter_type} is not implemented, ignoring.")
                    continue

            parsed_parameter["default"] = parameter_type(parsed_parameter["default"])
            all_parsed_parameters[name] = parsed_parameter

        for key in all_parsed_parameters.keys():
            if key not in inspect.signature(function).parameters.keys():
                raise ValueError(f"Function {function} does not take parameter {key}.")

        parsed_descriptor["parameters"] = all_parsed_parameters

        if self._functions is None:
            self._functions = []

        self._functions.append(parsed_descriptor)

        if menu not in self.function_submenu_names:
            if menu == MenuFunctions.DEFAULT_MENU_NAME:
                self._function_submenu_names += [menu]
            else:
                self._function_submenu_names.insert(-1, menu)

    @property
    def function_submenu_names(self):
        return self._function_submenu_names
        # return set([value["menu"] for value in self.functions])

    @property
    def function_key_mappings(self):
        key_mappings = dict()
        if self.functions is not None:
            for function_descriptor in self.functions:
                if (hotkey := function_descriptor["hotkey"]) is not None:
                    key_mappings[hotkey] = function_descriptor
                # key = ord(str((n + 1) % 10))
                # key_mappings[key] = f
        return key_mappings

    # def create_function_submenus(self, name, functions):
    def create_function_submenus(self):
        submenu_names = self.function_submenu_names

        if len(submenu_names) >= AppWindow.NUMBER_SUBFUNCTIONS:
            raise RuntimeError(
                f"Only {AppWindow.NUMBER_SUBFUNCTIONS} function submenus allowed."
            )

        self.function_submenus = {}

        for name in submenu_names:
            function_descriptors = [
                desc for desc in self.functions if desc["menu"] == name
            ]

            self.function_submenus[name] = MenuFunctions(
                self, function_descriptors, name=name
            )

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

    def _add_geometry_to_scene(self, elem):
        from open3d.geometry import LineSet, AxisAlignedBoundingBox, OrientedBoundingBox

        if isinstance(elem, (LineSet, AxisAlignedBoundingBox, OrientedBoundingBox)):
            mat = self.material_line
        else:
            mat = self.material_regular
        self._scene.scene.add_geometry(str(id(elem)), elem, mat)

    def _remove_geometry_from_scene(self, elem):
        self._scene.scene.remove_geometry(str(id(elem)))

    def _pop_elements(self, indices, from_vis=False):
        # update_old = self.i in indices
        # idx_new = self.i
        elements_popped = []
        for n, i in enumerate(indices):
            elem = self._element_dicts.pop(i - n)
            if from_vis:
                self._remove_geometry_from_scene(elem["drawable"])
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

    def _insert_elements(self, elems_raw, indices=None, selected=False, to_vis=False):
        if indices is None:
            indices = range(len(self.elements), len(self.elements) + len(elems_raw))

        if isinstance(selected, bool):
            selected = [selected] * len(indices)

        number_points_distance = self._settings.number_points_distance
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
                "distance_checker": get_distance_checker(
                    elems_raw[i], number_points_distance
                ),
            }

            # save original colors
            elem["color"] = extract_element_colors(elem["drawable"])

            if self._settings.paint_random:
                self.print_debug(
                    f"[_insert_elements] Randomly painting element at index {i}.",
                    require_verbose=True,
                )
                elem["drawable"] = get_painted_element(
                    elem["drawable"], "random", self._settings.random_color_brightness
                )

            elif self._settings.paint_selected and selected[i]:
                self.print_debug(
                    f"[_insert_elements] Painting and inserting element at index {i}.",
                    require_verbose=True,
                )
                is_current = self._started and (self.i == idx)
                color = self._settings.get_element_color(True, is_current)
                elem["drawable"] = get_painted_element(elem["drawable"], color)

            self.print_debug(
                f"Added {elem['raw']} at index {idx}.", require_verbose=True
            )
            self._element_dicts.insert(idx, elem)

        for idx in indices:
            # Updating vis explicitly in order not to remove it
            self._update_elements(idx, update_vis=False)
            if to_vis:
                self._add_geometry_to_scene(self.elements[idx]["drawable"])

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

        while len(self._past_states) > self._settings.number_undo_states:
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

        if len(self._elements_input) == 0:
            raise ValueError("Elements cannot be an empty list.")

        if len(self.selected) != len(self.elements):
            raise ValueError("Pre-select and input elements must have same length.")

    def _get_open3d(self, elem):
        from pyShapeDetector.geometry import TriangleMesh
        from open3d.geometry import Geometry as Open3D_Geometry

        if isinstance(elem, Open3D_Geometry):
            elem_new = copy.deepcopy(elem)

        else:
            try:
                elem_new = copy.deepcopy(elem.as_open3d)
                mesh_show_back_face = self._settings.mesh_show_back_face
                if mesh_show_back_face and TriangleMesh.is_instance_or_open3d(elem_new):
                    mesh = TriangleMesh(elem_new)
                    mesh.add_reverse_triangles()
                    elem_new = mesh.as_open3d

            except Exception as e:
                warnings.warn(f"Could not convert element: {elem}, got: {str(e)}")
                elem_new = elem

        if self._settings.paint_random:
            elem_new = get_painted_element(
                elem_new, "random", self._settings.random_color_brightness
            )

        return elem_new

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        pref = self._info.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self._info.frame = gui.Rect(
            r.x, r.get_bottom() - pref.height, pref.width, pref.height
        )

        # self._menu_functions._on_layout(r, layout_context)
        # self._settings._on_layout(r, layout_context)
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._main_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._main_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_mouse(self, event):
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        if not event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            return gui.Widget.EventCallbackResult.IGNORED

        if not event.is_modifier_down(gui.KeyModifier.CTRL):
            return gui.Widget.EventCallbackResult.IGNORED

        def depth_callback(depth_image):
            # Coordinates are expressed in absolute coordinates of the
            # window, but to dereference the image correctly we need them
            # relative to the origin of the widget. Note that even if the
            # scene widget is the only thing in the window, if a menubar
            # exists it also takes up space in the window (except on macOS).
            x = event.x - self._scene.frame.x
            y = event.y - self._scene.frame.y
            # Note that np.asarray() reverses the axes.
            depth = np.asarray(depth_image)[y, x]

            if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                return

            click_position = self._scene.scene.camera.unproject(
                x,
                y,
                depth,
                self._scene.frame.width,
                self._scene.frame.height,
            )

            # self.widget3d.scene.get_
            view_matrix = self._scene.scene.camera.get_view_matrix()
            camera_direction = -view_matrix[:3, 2]

            distances = self._get_distances_to_point(click_position, camera_direction)

            i_min_distance = np.argmin(distances)
            if distances[i_min_distance] is np.inf:
                return gui.Widget.EventCallbackResult.IGNORED

            self.i_old = self.i
            self.i = i_min_distance
            if event.is_modifier_down(gui.KeyModifier.SHIFT):
                self._hotkeys._cb_toggle()
            self._update_current_idx()

        self._scene.scene.scene.render_to_depth_image(depth_callback)

        return gui.Widget.EventCallbackResult.HANDLED

    def _get_distances_to_point(self, screen_point, screen_vector):
        """Called by Mouse callback to get distances to point when clicked."""
        from pyShapeDetector.geometry import PointCloud
        from pyShapeDetector.primitives import Primitive, Plane, Sphere

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

    def _setup_window_and_scene(self):
        # Set up the application
        self.app = gui.Application.instance
        self.app.initialize()

        # Create a window
        self.window = self.app.create_window(self.window_name, 1024, 768)
        self.window.set_on_layout(self._on_layout)

        # em = self.window.theme.font_size
        # separation_height = int(round(0.5 * em))

        # Set up a scene as a 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 0)
        )

        self.material_regular = rendering.MaterialRecord()
        self.material_regular.base_color = [1.0, 1.0, 1.0, 1.0]  # White color
        self.material_regular.shader = "defaultUnlit"
        self.material_regular.point_size = 3 * self.window.scaling

        self.material_line = rendering.MaterialRecord()
        self.material_line.shader = "unlitLine"
        self.material_line.line_width = 1.5 * self.window.scaling

        self._info = gui.Label("")
        self._info.visible = False

        self.window.add_child(self._scene)
        self.window.add_child(self._info)

        self._menubar = gui.Application.instance.menubar = gui.Menu()
        em = self.window.theme.font_size
        self._main_panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self.window.add_child(self._main_panel)
        self._main_panel.visible = True

        # 1) create hotkeys
        self._hotkeys = Hotkeys(self, self.function_key_mappings)
        self._scene.set_on_key(self._hotkeys._on_key)
        self._scene.set_on_mouse(self._on_mouse)

        # 2) create function menus
        self.create_function_submenus()
        for i, submenu in enumerate(self.function_submenus.values()):
            submenu._create_menu(AppWindow.MENU_SUBFUNCTIONS[i])

        self._settings._create_menu(AppWindow.MENU_SHOW_SETTINGS)

        # # 2) create hotkeys
        # self._hotkeys = Hotkeys(self, self.function_key_mappings)
        # self._scene.set_on_key(self._hotkeys._on_key)
        # self._scene.set_on_mouse(self._on_mouse)

        # 3) create instructions
        self._instructions = self._hotkeys.get_instructions()

        # 4) create help menu with instructions
        self._menu_help = MenuHelp(self)
        self._menu_help._create_menu(AppWindow.MENU_HELP)

    # def _signal_handler(self, sig, frame):
    #     self._vis.destroy_window()
    #     self._vis.close()
    #     sys.exit(0)

    def _update_plane_boundaries(self):
        for plane in self._plane_boundaries:
            self._remove_geometry_from_scene(plane)

        plane_boundaries = []

        if self._settings.draw_boundary_lines:
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
            self._add_geometry_to_scene(boundary)

    def _update_current_bounding_box(self):
        """Remove bounding box and get new one for current element."""
        from pyShapeDetector.geometry import (
            LineSet,
            OrientedBoundingBox,
            AxisAlignedBoundingBox,
        )

        bbox_expand = self._settings.bbox_expand
        self.print_debug("Updating bounding box...", require_verbose=True)

        # vis = self._vis
        # if vis is None:
        #     return

        if self._current_bbox is not None:
            self._remove_geometry_from_scene(self._current_bbox)

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
            bbox.color = self._settings.color_bbox_selected
        else:
            bbox.color = self._settings.color_bbox_unselected

        self.print_debug(f"New bounding box: {bbox}", require_verbose=True)
        self._current_bbox = bbox.as_open3d

        if self._current_bbox is not None:
            self._add_geometry_to_scene(self._current_bbox)

    def _update_elements(self, indices, update_vis=True):
        num_elems = len(self.elements)

        if indices is None:
            indices = range(len(self.elements))

        if not isinstance(indices, (list, range)):
            indices = [indices]

        if len(indices) == 0:
            return

        if num_elems == 0 or max(indices) >= num_elems:
            warnings.warn(
                "Tried to update index {indices}, but {num_elems} elements present."
            )
            return

        for idx in indices:
            elem = self.elements[idx]
            is_selected = elem["selected"] and self.select_filter(elem)
            is_current = self._started and (idx == self.i)

            if update_vis:
                self._remove_geometry_from_scene(elem["drawable"])

            self.print_debug(
                "[_update_element] "
                f"Updating element at index: {idx}.\n"
                f"Selected = {is_selected}, current = {is_current}.",
                require_verbose=True,
            )

            if self._settings.paint_selected and is_selected:
                color = self._settings.get_element_color(True, is_current)
                self.print_debug(
                    f"[_update_element] Painting drawable to color: {color}.",
                    require_verbose=True,
                )

            else:
                highlight_offset = self._settings.highlight_color_brightness * (
                    int(is_selected) + int(is_current)
                )
                color = elem["color"] + highlight_offset

            set_element_colors(elem["drawable"], color)

            if update_vis:
                self.print_debug(
                    "[_update_element] Updating geometry on gui.",
                    require_verbose=True,
                )
                self._add_geometry_to_scene(elem["drawable"])

        # This is not called on the main thread, so we need to
        # post to the main thread to safely access UI items.
        def update_label():
            self._info.text = (
                f"Current: {self.i + 1} / {len(self._element_dicts)} | "
                f"selected: {'YES' if self.is_current_selected else 'NO'}"
            )

            if n := len(self._hidden_elements):
                self._info.text += f"\n{n} hidden elements"
            self._info.visible = self._info.text != ""
            # We are sizing the info label to be exactly the right size,
            # so since the text likely changed width, we need to
            # re-layout to set the new frame.
            self.window.set_needs_layout()

        gui.Application.instance.post_to_main_thread(self.window, update_label)

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

        self._update_elements(self.i)
        if update_old:
            self._update_elements(self.i_old)
        self._update_current_bounding_box()

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

    def _toggle_indices(self, indices_or_slice):
        indices = self._get_range(indices_or_slice)

        for idx in indices:
            elem = self.elements[idx]
            is_selectable = self.select_filter(elem)
            selected = (not self._hotkeys._is_modifier_pressed) and is_selectable
            elem["selected"] = selected

        self._update_elements(indices)

    def _apply_function_to_elements(self, function):
        """Apply function to selected elements."""

        indices = self.selected_indices
        self.print_debug(
            f"Applying {function.__name__} function to {len(indices)} elements, indices: "
        )
        self.print_debug(indices)
        input_elements = [self.elements[i]["raw"] for i in indices]

        try:
            output_elements = function(input_elements)
        except KeyboardInterrupt:
            return
        except Exception as e:
            warnings.warn(
                f"Failed to apply {function.__name__} function to "
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

        self._last_used_function = function
        self._future_states = []
        self._update_plane_boundaries()

    def _reset_elements_in_gui(self, startup=False, reset_fixed=False):
        """Prepare elements for visualization"""

        if not startup:
            self.i = min(self.i, len(self.elements) - 1)

        if startup or reset_fixed:
            for elem in self._fixed_elements_drawable:
                self._remove_geometry_from_scene(elem)

            self._fixed_elements_drawable = [
                self._get_open3d(elem) for elem in self._fixed_elements
            ]

            for elem in self._fixed_elements_drawable:
                self._add_geometry_to_scene(elem)

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

        self._update_plane_boundaries()
        self._update_current_bounding_box()

        self._update_current_idx(current_idx)
        self._started = True

    def run(self):
        self.print_debug("Starting ElementSelector instance.")

        if len(self._elements_input) == 0:
            raise RuntimeError("No elements added!")

        # Ensure proper format of inputs, check elements and raise warnings
        self._check_and_initialize_inputs()

        # Set up the gui
        self._setup_window_and_scene()
        self._reset_elements_in_gui(startup=True)

        bounds = self._scene.scene.bounding_box
        center = bounds.get_center()
        self._scene.setup_camera(60, bounds, center)
        self._scene.look_at(center, center - [0, 0, 3], [0, -1, 0])

        self.app.run()

        # try:
        #     self._vis.run()
        #     # add hidden elements back to elements list
        #     # self._insert_elements(self._hidden_elements, to_vis=False)
        # except Exception as e:
        #     raise e
        # finally:
        #     self._vis.close()
        #     self._vis.destroy_window()
