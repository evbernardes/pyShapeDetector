#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tur Oct 17 13:17:55 2024

@author: ebernardes
"""
import time
import copy
import inspect
import traceback
import signal
import sys
import warnings
import inspect
import itertools
import numpy as np
from pathlib import Path
from open3d.visualization import gui, rendering
# from .element import Element, ElementGeometry

from .element_container import ElementContainer


class Editor:
    """
    Editor Graphical Interface to manually select elements and apply functions to
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
        Name of window. If empty, just gives the number of elements.
        Default: "Shape Detector".

    Preferences
    -----------
        See: settings.Settings



    return_finish_flag : boolean, optional
        Should be deprecated.
    """

    def __init__(
        self,
        window_name="Shape Detector",
        return_finish_flag=False,
        **kwargs,
    ):
        self._copied_elements = []
        self._past_states = []
        self._future_states = []
        self._elements = ElementContainer(self)
        self._plane_boundaries = ElementContainer(self, is_color_fixed=True)
        self._hidden_elements = ElementContainer(self)
        self._elements_fixed = ElementContainer(self, is_color_fixed=True)
        self._pre_selected = []
        self._current_bbox = None
        self._current_bbox_axes = None
        self._extensions = []
        self._last_used_extension = None
        self._select_filter = lambda x: True
        self.window_name = window_name
        self.return_finish_flag = return_finish_flag
        self._submenu_id_generator = itertools.count(1, 1)
        self._submenus = {}
        self._closing_app = False
        self._temp_windows = []

        # self._elements_distance = []
        self.i_old = 0
        self.i = 0
        self.finish = False
        self._started = False

        from .settings import Settings

        self._settings = Settings(self, **kwargs)

    def _get_submenu_from_path(self, path):
        if path in self._submenus:
            return self._submenus[path]

        fullpath = Path()

        if not hasattr(self, "_menubar"):
            self._menubar = self.app.menubar = gui.Menu()
            self.print_debug("Created menubar.")

        if not hasattr(self, "_submenus"):
            self._submenus = {}
            self.print_debug("Initialized submenus dict.")

        upper_menu = self._menubar

        for part in path.parts:
            fullpath /= part
            if fullpath in self._submenus:
                upper_menu = self._submenus[fullpath]
            menu = gui.Menu()
            upper_menu.add_menu(part, menu)
            self._submenus[path] = menu

        self.print_debug(f"Submenu '{path.as_posix()}' created.")
        return self._submenus[path]

    def print_debug(self, text, require_verbose=False):
        is_debug_activated = self._get_preference("debug")
        is_verbose_activated = self._get_preference("verbose")

        if not is_debug_activated or (require_verbose and not is_verbose_activated):
            return

        text = str(text)
        print("[DEBUG] " + text)

    @property
    def extensions(self):
        return self._extensions

    def add_extension(self, function_or_descriptor):
        from .extension import Extension

        try:
            extension = Extension(function_or_descriptor, self._settings)
            extension.add_to_application(self)

        except Exception:
            warnings.warn(f"Could not create extension {function_or_descriptor}, got:")
            traceback.print_exc()

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
        self._select_filter = lambda elem: new_function(elem.raw)

    def get_elements(self, add_hidden: bool = True, add_fixed: bool = False):
        elements = self.elements
        if add_hidden:
            elements += self._hidden_elements
        if add_fixed:
            elements += self._elements_fixed
        return elements.raw

    @property
    def elements(self):
        return self._elements

    @property
    def elements_fixed(self):
        return self._elements_fixed

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
        return self.current_element.selected

    @is_current_selected.setter
    def is_current_selected(self, boolean_value: bool):
        if not isinstance(boolean_value, bool):
            raise RuntimeError(f"Expected boolean, got {type(boolean_value)}.")
        if self.current_element is None:
            raise RuntimeError(
                "Error setting selected, current element does not exist."
            )
        self.current_element.selected = boolean_value

    def _reset_on_key(self):
        """Reset keys to original hotkeys."""
        self._scene.set_on_key(self._hotkeys._on_key)

    def _close_dialog(self):
        """Closes dialog, if any, then resets keys to original hotkeys."""
        self._window.set_on_key(None)
        self._window.close_dialog()
        self._reset_on_key()

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

        while len(self._past_states) > self._get_preference("number_undo_states"):
            self._past_states.pop(0)

    def _on_layout(self, layout_context):
        r = self._window.content_rect

        self._scene.frame = r

        pref = self._info.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self._info.frame = gui.Rect(
            r.x, r.get_bottom() - pref.height, pref.width, pref.height
        )

        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._right_side_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._right_side_panel.frame = gui.Rect(
            r.get_right() - width, r.y, width, height
        )

        # Hide right side panel if subpanels are also hidden
        self._right_side_panel.visible = np.any(
            [widget.visible for widget in self._right_side_panel.get_children()]
        )

    def _on_close(self):
        if not self._closing_app:
            self._internal_functions._dict["Quit"].callback()
            return False

        # self.elements.insert_multiple(self._hidden_elements.raw, to_gui=False)
        for window in self._temp_windows:
            window.close()
        return True

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

            distances = self.elements.get_distances_to_point(
                click_position, camera_direction
            )

            i_min_distance = np.argmin(distances)
            if distances[i_min_distance] is np.inf:
                return gui.Widget.EventCallbackResult.IGNORED

            self.i_old = self.i
            self.i = i_min_distance
            if event.is_modifier_down(gui.KeyModifier.SHIFT):
                self._internal_functions._cb_toggle()
            self._update_current_idx()

        self._scene.scene.scene.render_to_depth_image(depth_callback)

        return gui.Widget.EventCallbackResult.HANDLED

    def _get_preference(self, key):
        if key not in self._settings._dict:
            warnings.warn(f"Tried getting non existing preference {key}")
            return None
        return self._settings._dict[key].value

    def _setup_window_and_scene(self):
        from .menu_help import MenuHelp
        from .hotkeys import Hotkeys
        from .internal_functions import InternalFunctions

        # Set up the application
        self.app = gui.Application.instance
        self.app.initialize()

        # Create a window
        self._window = self.app.create_window(self.window_name, 1024, 768)
        self._window.set_on_layout(self._on_layout)
        self._window.set_on_close(self._on_close)

        # em = self.window.theme.font_size
        # separation_height = int(round(0.5 * em))

        # Set up a scene as a 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self._window.renderer)
        self._scene.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 0)
        )
        self._scene.scene.show_axes(False)

        self.material_regular = rendering.MaterialRecord()
        self.material_regular.base_color = [1.0, 1.0, 1.0, 1.0]  # White color
        self.material_regular.shader = "defaultUnlit"
        self.material_regular.point_size = (
            self._get_preference("PointCloud_point_size") * self._window.scaling
        )

        self.material_line = rendering.MaterialRecord()
        self.material_line.shader = "unlitLine"
        self.material_line.line_width = (
            self._get_preference("line_width") * self._window.scaling
        )

        self._info = gui.Label("")
        self._info.visible = False

        self._window.add_child(self._scene)
        self._window.add_child(self._info)

        em = self._window.theme.font_size
        self._right_side_panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self._window.add_child(self._right_side_panel)
        self._right_side_panel.visible = True

        # 1) Set internal functions and add menu items
        self._internal_functions = InternalFunctions(self)

        # 2) Add hotkeys for both internal functions and extensions
        self._hotkeys = Hotkeys(self)
        self._reset_on_key()
        self._scene.set_on_mouse(self._on_mouse)

        # 3) Add extension menu items
        self.print_debug(
            f"{len(self._internal_functions.bindings)} internal functions."
        )
        for binding in self._internal_functions.bindings:
            binding.add_to_menu(self)

        self.print_debug(
            f"{len(self.extensions) if self.extensions else 0} extensions loaded."
        )
        for extension in self.extensions:
            extension.binding.add_to_menu(self)

        # 4) Finally, other menus (so that they are at the end)
        self._settings._create_menu()
        self._menu_help = MenuHelp(self)
        self._menu_help._create_menu()

    # def _signal_handler(self, sig, frame):
    #     self._vis.destroy_window()
    #     self._vis.close()
    #     sys.exit(0)

    def _update_plane_boundaries(self):
        from .element import ElementGeometry

        for plane_boundary in self._plane_boundaries:
            plane_boundary.remove_from_scene()
            # self._remove_geometry_from_scene(plane_boundary)

        plane_boundaries = []

        if self._get_preference("draw_boundary_lines"):
            for elem in self.elements:
                try:
                    lineset = elem.raw.vertices_LineSet.as_open3d
                    if hasattr(elem.raw, "holes"):
                        for hole in elem.raw.holes:
                            lineset += hole.vertices_LineSet.as_open3d

                    plane_boundary = ElementGeometry(self, lineset)
                    plane_boundary.add_to_scene()
                    plane_boundaries.append(plane_boundary)

                except AttributeError:
                    continue

        self._plane_boundaries = plane_boundaries

    def _update_BBOX_and_axes(self):
        """Remove bounding box and get new one for current element."""
        from pyShapeDetector.geometry import TriangleMesh

        self.print_debug("Updating bounding box...", require_verbose=True)

        if self._current_bbox is not None:
            self._scene.scene.remove_geometry("CurrentBoundingBox")
            self._current_bbox = None

        if self._current_bbox_axes is not None:
            self._scene.scene.remove_geometry("BBOXAxisX")
            self._scene.scene.remove_geometry("BBOXAxisY")
            self._scene.scene.remove_geometry("BBOXAxisZ")
            self._current_bbox_axes

        if self.current_element is None or not self._get_preference("show_BBOX"):
            self._current_bbox = None
            return

        self._current_bbox = self.current_element._get_bbox().as_open3d

        if self._current_bbox is not None:
            self._scene.scene.add_geometry(
                "CurrentBoundingBox", self._current_bbox, self.material_line
            )

        if self._current_bbox is not None and self._get_preference("show_BBOX_axes"):
            center = self._current_bbox.center
            extent = self._current_bbox.extent
            R = self._current_bbox.R
            radius = self._get_preference("BBOX_axes_width") * self._window.scaling

            min_bound = center - extent.dot(R.T) / 2
            vx = TriangleMesh.create_arrow_from_points(
                min_bound, min_bound + extent[0] * R.dot([1, 0, 0]), radius=radius
            )
            vx.paint_uniform_color([1, 0, 0])
            vy = TriangleMesh.create_arrow_from_points(
                min_bound, min_bound + extent[1] * R.dot([0, 1, 0]), radius=radius
            )
            vy.paint_uniform_color([0, 1, 0])
            vz = TriangleMesh.create_arrow_from_points(
                min_bound, min_bound + extent[2] * R.dot([0, 0, 1]), radius=radius
            )
            vz.paint_uniform_color([0, 0, 1])
            self._scene.scene.add_geometry(
                "BBOXAxisX", vx.as_open3d, self.material_regular
            )
            self._scene.scene.add_geometry(
                "BBOXAxisY", vy.as_open3d, self.material_regular
            )
            self._scene.scene.add_geometry(
                "BBOXAxisZ", vz.as_open3d, self.material_regular
            )
            self._current_bbox_axes = (vx, vy, vz)

    def _update_elements(self, indices, update_gui=True):
        num_elems = len(self.elements)

        if indices is None:
            indices = range(num_elems)

        if not isinstance(indices, (list, range)):
            indices = [indices]

        if len(indices) == 0:
            return

        if num_elems == 0 or max(indices) >= num_elems:
            warnings.warn(
                f"Tried to update index {indices}, but {num_elems} elements present."
            )
            return

        for idx in indices:
            elem = self.elements[idx]
            elem._selected = elem.selected and self.select_filter(elem)
            is_current = self._started and (idx == self.i)

            elem.update(is_current, update_gui)

        self._update_info()

    def _update_info(self):
        def update_label():
            # This is not called on the main thread, so we need to
            # post to the main thread to safely access UI items.
            self._info.text = (
                f"Current: {self.i + 1} / {len(self._elements)} | "
                f"selected: {'YES' if self.is_current_selected else 'NO'}"
            )

            if n := len(self._hidden_elements):
                self._info.text += f" | {n} hidden elements"

            if (ext := self._last_used_extension) is not None:
                name = ext.name
                params = ext.parameters_kwargs

                self._info.text += f"\nLast used function: {name}"
                if len(params) > 0:
                    self._info.text += f", with :{params}"

                repeat_binding = self._internal_functions._dict["Repeat last extension"]
                if repeat_binding is not None:
                    self._info.text += f"\n{repeat_binding.key_instruction} to repeat"

            # self._info.visible = self._info.text != ""
            # We are sizing the info label to be exactly the right size,
            # so since the text likely changed width, we need to
            # re-layout to set the new frame.
            self._window.set_needs_layout()

        self.app.post_to_main_thread(self._window, update_label)

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
        self._update_BBOX_and_axes()

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

    def _toggle_indices(self, indices_or_slice, to_value=None):
        indices = self._get_range(indices_or_slice)

        for idx in indices:
            elem = self.elements[idx]
            is_selectable = self.select_filter(elem)
            if to_value is None:
                selected = (not self._hotkeys._is_lshift_pressed) and is_selectable
            else:
                selected = to_value and is_selectable
            elem.selected = selected

        self._update_elements(indices)

    # def _reset_elements_in_gui(self, reset_fixed=False):
    #     """Prepare elements for visualization"""

    #     current_idx = copy.copy(min(self.i, len(self.elements) - 1))
    #     if reset_fixed:
    #         self.elements_fixed.update_all()
    #     else:
    #         self.elements.update_all()

    #     self._update_plane_boundaries()
    #     self._update_current_bounding_box()
    #     self._update_current_idx(current_idx)

    def run(self):
        self.print_debug(f"Starting {type(self).__name__}.")

        # Set up the gui
        self._setup_window_and_scene()

        for elem in self.elements + self.elements_fixed:
            elem.add_to_scene()

        self._update_BBOX_and_axes()
        self._update_info()
        bounds = self._scene.scene.bounding_box
        center = bounds.get_center()
        self._scene.setup_camera(60, bounds, center)
        self._scene.look_at(center, center - [0, 0, 3], [0, 1, 0])

        self._started = True

        self.app.run()

        # try:
        #     self.app.run()
        # except Exception as e:
        #     # raise e
        #     traceback.print_exc()
        # finally:
        # self._insert_elements(self._hidden_elements, to_gui=False)
        # self.elements.insert_multiple(self._hidden_elements, to_gui=False)
