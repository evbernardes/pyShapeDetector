#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tur Oct 17 13:17:55 2024

@author: ebernardes
"""
# import time
import copy
import inspect
import traceback

# import signal
# import sys
import warnings
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
        self._elements_hidden = ElementContainer(self)
        self._elements_fixed = ElementContainer(self, is_color_fixed=True)
        self._pre_selected = []
        self._current_bbox = None
        self._current_bbox_axes = None
        self._extensions = []
        self._last_used_extension = None
        self.window_name = window_name
        self.return_finish_flag = return_finish_flag
        self._submenu_id_generator = itertools.count(1, 1)
        self._submenus = {}
        self._closing_app = False
        self._temp_windows = []
        self._scene_file_path = None

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

    def get_elements(self, add_hidden: bool = True, add_fixed: bool = False):
        elements = self.elements
        if add_hidden:
            elements += self._elements_hidden
        if add_fixed:
            elements += self._elements_fixed
        return elements.raw

    @property
    def elements(self):
        return self._elements

    @property
    def elements_fixed(self):
        return self._elements_fixed

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
            "current_index": self.elements.current_index,
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

            self.elements._previous_index = self.elements.current_index
            self.elements.current_index = i_min_distance
            if event.is_modifier_down(gui.KeyModifier.SHIFT):
                self._internal_functions._cb_toggle()
            self.elements.update_current_index()

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

        if self.elements.current_element is None or not self._get_preference(
            "show_BBOX"
        ):
            self._current_bbox = None
            return

        self._current_bbox = self.elements.current_element._get_bbox().as_open3d

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

    def _update_info(self):
        def update_label():
            # This is not called on the main thread, so we need to
            # post to the main thread to safely access UI items.
            if self.elements.current_index is None:
                return

            self._info.text = (
                f"Current: {self.elements.current_index + 1} / {len(self.elements)} | "
                f"selected: {'YES' if self.elements.is_current_selected else 'NO'}"
            )

            if n := len(self._elements_hidden):
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
