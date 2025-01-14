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
from typing import Callable, Union, Literal
from pathlib import Path
from open3d.visualization import gui, rendering
from pyShapeDetector.primitives import Primitive
# from .element import Element, ElementGeometry

from .element_container import ElementContainer
from .settings import Settings
from .extensions import default_extensions


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
    width : int, optional
        Default width of window. Default: 1024.
    height : int, optional
        Default height of window. Default: 768.
    window_name : str, optional
        Name of window. If empty, just gives the number of elements.
        Default: "Shape Detector".
    load_default_extensions : bool, optional.
        Load default extensions. Default: True.

    Preferences
    -----------
        See: settings.Settings

    return_finish_flag : boolean, optional
        Should be deprecated.
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        window_name: str = "Shape Detector",
        load_default_extensions: bool = True,
        return_finish_flag: bool = False,
        **kwargs,
    ):
        from .internal_functions import InternalFunctions

        self._extensions = []
        self._settings = Settings(self, **kwargs)

        self._init_window_size = (width, height)
        self._copied_elements = []
        self._past_states = []
        self._future_states = []
        self._elements = ElementContainer(self._settings)
        self._plane_boundaries = ElementContainer(self._settings, is_color_fixed=True)
        self._elements_fixed = ElementContainer(self._settings, is_color_fixed=True)
        self._pre_selected = []
        self._current_bbox = None
        self._current_bbox_axes = None
        self._last_used_extension = None
        self.window_name = window_name
        self.return_finish_flag = return_finish_flag
        self._submenu_id_generator = itertools.count(1, 1)
        self._submenus = {}
        self._closing_app = False
        self._temp_windows = []
        self._scene_file_path = None
        self._window = None
        self._extensions_panels = {}
        # self._extension_tabs = gui.TabControl()
        # self._gray_overlay = gui.Widget()

        self.finish = False
        self._started = False

        self._internal_functions = InternalFunctions(self)

        if load_default_extensions:
            for extension in default_extensions:
                self.add_extension(extension)

    def _create_simple_dialog(
        self,
        text,
        create_button: bool = True,
        button_text: str = "Close",
        button_callback: Union[Callable, None] = None,
    ):
        window = self._window
        em = window.theme.font_size
        dlg = gui.Dialog(text)

        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

        title = gui.Horiz()
        title.add_stretch()
        title.add_child(gui.Label(text))
        title.add_stretch()

        dlg_layout.add_child(title)

        if create_button:
            button = gui.Button(button_text)
            if button_callback is None:

                def _on_ok():
                    self._close_dialog()

                button.set_on_clicked(_on_ok)
            else:
                button.set_on_clicked(button_callback)

            button_stretch = gui.Horiz()
            button_stretch.add_stretch()
            button_stretch.add_child(button)
            button_stretch.add_stretch()
            dlg_layout.add_child(button_stretch)

        dlg.add_child(dlg_layout)
        window.show_dialog(dlg)

    def _get_submenu_from_path(self, path):
        if path in self._submenus:
            if "Create Shapes" in path.as_posix():
                pass
            return self._submenus[path]

        if not hasattr(self, "_menubar"):
            self._menubar = self.app.menubar = gui.Menu()
            self._settings.print_debug("Created menubar.")

        if not hasattr(self, "_submenus"):
            self._submenus = {}
            self._settings.print_debug("Initialized submenus dict.")

        fullpath = Path()
        upper_menu = self._menubar

        # for part in path.parts:
        #     fullpath /= part
        #     if fullpath in self._submenus:
        #         upper_menu = self._submenus[fullpath]
        #     menu = gui.Menu()
        #     upper_menu.add_menu(part, menu)
        #     self._submenus[path] = menu

        for part in path.parts:
            fullpath /= part
            if fullpath not in self._submenus:
                menu = gui.Menu()
                upper_menu.add_menu(part, menu)
                self._submenus[fullpath] = menu
                self._settings.print_debug(f"Submenu '{fullpath.as_posix()}' created.")

            upper_menu = self._submenus[fullpath]

        self._submenus[path] = menu

        return self._submenus[path]

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

    def get_elements(self, add_fixed: bool = False):
        elements = self.elements
        if add_fixed:
            elements += self._elements_fixed
        return elements.raw

    @property
    def elements(self):
        return self._elements

    @property
    def elements_fixed(self):
        return self._elements_fixed

    def _add_extension_panel(self, name, panel, callbacks):
        if name in self._extensions_panels:
            self._extensions_panels.pop(name).visible = False
            self._add_extension_panel(name, panel, callbacks)
            return

        N = len(self._extensions_panels)

        em = self._window.theme.font_size

        # self._extensions_panels[name] = panel
        panel_collapsable = gui.CollapsableVert(name, em, gui.Margins(0, 0, 0, 0))
        panel_collapsable.add_child(panel)

        # Small buttons
        # ok = gui.Button("Ok")
        # ok.set_on_clicked(callbacks["close"])
        # ok.vertical_padding_em = 0
        x = gui.Button("X")
        x.set_on_clicked(callbacks["close"])
        x.vertical_padding_em = 0

        extension_line = gui.VGrid(2, 0)
        # extension_line.add_child(ok)
        extension_line.add_child(x)
        extension_line.add_child(panel_collapsable)

        self._extension_tabs_panel.add_child(extension_line)
        self._extensions_panels[name] = extension_line
        self._window.set_needs_layout()

    def _set_extension_panel_open(self, name, is_open) -> bool:
        if name not in self._extensions_panels:
            return False

        self._extensions_panels[name].visible = is_open
        self._window.set_needs_layout()
        return True

    # def _set_gray_overlay(self, value: bool):
    #     if value is False:
    #         self._gray_overlay.visible = False
    #         return

    #     width = self._window.content_rect.width
    #     height = self._window.content_rect.height
    #     self._gray_overlay.frame = gui.Rect(0, 0, int(width), int(height))
    #     self._gray_overlay.background_color = gui.Color(0.5, 0.5, 0.5, 0.1)
    #     self._gray_overlay.visible = True

    def _reset_on_key(self):
        """Reset keys to original hotkeys."""
        self._scene.set_on_key(self._hotkeys._on_key)

    def _close_dialog(self):
        """Closes dialog, if any, then resets keys to original hotkeys."""
        self._window.set_on_key(None)
        self._window.close_dialog()
        self._reset_on_key()

    def _save_state(
        self, current_state: dict, to_future: bool = False, delete_future: bool = True
    ):
        """Save state for undoing."""

        state_printable = copy.copy(current_state)
        if "indices" in state_printable:
            state_printable["num_inputs"] = len(state_printable["indices"])
            state_printable.pop("indices", None)
            state_printable.pop("elements", None)

        if to_future:
            self._future_states.append(current_state)
            self._settings.print_debug(f"Saving future state {state_printable}.")

            max_states = self._settings.get_setting("number_redo_states")
            while len(self._future_states) > max_states:
                state = self._future_states.pop(0)
                del state

        else:
            self._past_states.append(current_state)
            self._settings.print_debug(f"Saving past state {state_printable}.")

            max_states = self._settings.get_setting("number_undo_states")
            while len(self._past_states) > max_states:
                state = self._past_states.pop(0)
                del state

            if delete_future:
                self._future_states = []

        self._settings.print_debug(f"{len(self._past_states)} states for undoing.")
        self._settings.print_debug(f"{len(self._future_states)} states for redoing.")

    def _on_layout(self, layout_context):
        r = self._window.content_rect

        self._scene.frame = r

        info_preferred_size = self._info.calc_preferred_size(
            layout_context, gui.Widget.Constraints()
        )

        self._info.frame = gui.Rect(
            r.x,
            r.get_bottom() - info_preferred_size.height,
            info_preferred_size.width,
            info_preferred_size.height,
        )

        def _adjust_panel(panel, min_width=17):
            preferred_size = panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            )

            width = min(
                min_width * layout_context.theme.font_size, preferred_size.width
            )
            height = min(r.height, preferred_size.height)

            panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

            panel.visible = np.any(
                [False] + [widget.visible for widget in panel.get_children()]
            )

        _adjust_panel(self._extension_tabs_panel, 17)
        _adjust_panel(self._right_side_panel, 25)

    def _on_close(self):
        if not self._closing_app:
            self._internal_functions._dict["Quit"].callback()
            return False

        # for window in self._temp_windows:
        #     window.close()
        self.app.quit()
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
            self._update_extra_elements(planes_boundaries=False)

        self._scene.scene.scene.render_to_depth_image(depth_callback)

        return gui.Widget.EventCallbackResult.HANDLED

    def _get_setting(self, key):
        return self._settings.get_setting(key)

    def _setup_window_and_scene(self):
        from .menu_show import MenuShow
        from .hotkeys import Hotkeys
        # from .internal_functions import InternalFunctions

        # Set up the application
        self.app = gui.Application.instance
        self.app.initialize()

        # Create a window
        self._window = self.app.create_window(self.window_name, *self._init_window_size)
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
        self._global_axes_visible = False
        self._scene.scene.show_axes(self._global_axes_visible)

        self._ground_plane_visible = True
        self._ground_plane = rendering.Scene.GroundPlane.XY
        self._scene.scene.show_ground_plane(
            self._ground_plane_visible, self._ground_plane
        )

        self._info = gui.Label("")
        self._info.visible = True

        self._window.add_child(self._scene)
        self._window.add_child(self._info)

        em = self._window.theme.font_size
        self._right_side_panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self._window.add_child(self._right_side_panel)
        self._right_side_panel.visible = True

        # 1) Set internal functions and add menu items
        # self._internal_functions = InternalFunctions(self)

        # 2) Add hotkeys for both internal functions and extensions
        self._hotkeys = Hotkeys(self)
        self._reset_on_key()
        self._scene.set_on_mouse(self._on_mouse)

        # 3) Add extension menu items
        self._settings.print_debug(
            f"{len(self._internal_functions.bindings)} internal functions.",
        )
        for binding in self._internal_functions.bindings:
            binding.add_to_menu(self)

        self._settings.print_debug(
            f"{len(self.extensions) if self.extensions else 0} extensions loaded.",
        )
        for extension in self.extensions:
            extension.binding.add_to_menu(self)

        # 4) Finally, other menus (so that they are at the end)
        self._settings._create_menu()
        self._menu_show = MenuShow(self)
        self._menu_show._create_menu()

        self._extension_tabs_panel = gui.CollapsableVert(
            "Extensions", em, gui.Margins(em, em, em, em)
        )
        self._right_side_panel.add_child(self._extension_tabs_panel)

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

        if self._settings.get_setting("draw_boundary_lines"):
            for elem in self.elements:
                if elem.is_hidden:
                    continue

                try:
                    lineset = elem.raw.vertices_LineSet.as_open3d
                    if hasattr(elem.raw, "holes"):
                        for hole in elem.raw.holes:
                            lineset += hole.vertices_LineSet.as_open3d

                    plane_boundary = ElementGeometry(self, lineset)
                    plane_boundary.add_to_scene(self._scene)
                    plane_boundaries.append(plane_boundary)

                except AttributeError:
                    continue

        self._plane_boundaries = plane_boundaries

    def _update_BBOX_and_axes(self):
        """Remove bounding box and get new one for current element."""
        from pyShapeDetector.geometry import TriangleMesh

        self._settings.print_debug("Updating bounding box...", require_verbose=True)

        if self._current_bbox is not None:
            self._scene.scene.remove_geometry("CurrentBoundingBox")
            self._current_bbox = None

        if self._current_bbox_axes is not None:
            self._scene.scene.remove_geometry("BBOXAxisX")
            self._scene.scene.remove_geometry("BBOXAxisY")
            self._scene.scene.remove_geometry("BBOXAxisZ")
            self._current_bbox_axes

        if self.elements.current_element is None or not self._settings.get_setting(
            "show_bbox"
        ):
            self._current_bbox = None
            return

        self._current_bbox = self.elements.current_element._get_bbox().as_open3d

        if self._current_bbox is not None:
            self._scene.scene.add_geometry(
                "CurrentBoundingBox",
                self._current_bbox,
                self._settings.get_material("line"),
            )

        if self._current_bbox is not None and self._settings.get_setting(
            "show_bbox_axes"
        ):
            center = self._current_bbox.center
            extent = self._current_bbox.extent
            R = self._current_bbox.R
            radius = (
                self._settings.get_setting("bbox_axes_width") * self._window.scaling
            )

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

            material = self._settings.get_material("regular")

            vz.paint_uniform_color([0, 0, 1])
            self._scene.scene.add_geometry("BBOXAxisX", vx.as_open3d, material)
            self._scene.scene.add_geometry("BBOXAxisY", vy.as_open3d, material)
            self._scene.scene.add_geometry("BBOXAxisZ", vz.as_open3d, material)
            self._current_bbox_axes = (vx, vy, vz)

    def _update_info(self):
        if self._window is None:
            """ Not initialized yet"""
            return

        debug = self._get_setting("debug")

        def update_label():
            # This is not called on the main thread, so we need to
            # post to the main thread to safely access UI items.
            if self.elements.current_index is None or len(self.elements) == 0:
                self._info.text = ""
            else:
                self._info.text = f"Current: {self.elements.current_index + 1} / {len(self.elements)} | "

                if self._get_setting("show_current"):
                    element_raw = self.elements.current_element.raw
                    self._info.text += f"{element_raw}"

                    if isinstance(element_raw, Primitive):
                        self._info.text += f", {len(element_raw.inliers.points)} inliers, area = {element_raw.surface_area}"

                    try:
                        extent = element_raw.get_oriented_bounding_box().extent
                        extent = np.trunc(10e7 * extent) / 10e7
                        self._info.text += f" | BBOX extent: {extent.tolist()}"

                    except Exception:
                        pass  # not important

                    self._info.text += " | "

                self._info.text += (
                    f"selected: {'YES' if self.elements.is_current_selected else 'NO'}"
                )

                if debug:
                    self._info.text += (
                        f" | {len(self.elements.selected_indices)} selected elements"
                    )

                if n := len(self.elements.hidden_indices):
                    self._info.text += f" | {n} hidden elements"

                if (ext := self._last_used_extension) is not None:
                    name = ext.name
                    # params = ext.parameters_kwargs

                    self._info.text += f"\nLast used function: {name}"
                    # if len(params) > 0:
                    #     self._info.text += f", with :{params}"

                    repeat_binding = self._internal_functions._dict[
                        "Repeat last extension"
                    ]
                    if repeat_binding is not None:
                        self._info.text += (
                            f"\n{repeat_binding.key_instruction} to repeat"
                        )

            self._info.visible = self._info.text != ""
            # We are sizing the info label to be exactly the right size,
            # so since the text likely changed width, we need to
            # re-layout to set the new frame.
            self._window.set_needs_layout()

        self.app.post_to_main_thread(self._window, update_label)

    def _update_extra_elements(self, planes_boundaries: bool = True):
        if planes_boundaries:
            self._update_plane_boundaries()
        self._update_BBOX_and_axes()
        self._update_info()

    def _reset_camera(self):
        bounds = self._scene.scene.bounding_box
        center = bounds.get_center()
        self._scene.setup_camera(60, bounds, center)
        self._scene.look_at(center, center - [0, 0, 3], [0, 1, 0])

    def _startup(self):
        """Runs all necessary startups, might be useful for testing."""
        self._settings.print_debug(f"Starting {type(self).__name__}.")

        # Set up the scene
        self._setup_window_and_scene()
        self._settings._update_materials()

        # Add initial elements
        self.elements.add_to_scene(self._scene.scene, startup=True)
        # self._plane_boundaries.add_to_scene(self._scene.scene)
        self.elements_fixed.add_to_scene(self._scene.scene)

        self._update_extra_elements()
        self._reset_camera()

        self._started = True

    def run(self):
        # This calls of of the necessary startups:
        self._startup()

        # And this runs the main gui loop:
        self.app.run()

        # try:
        #     self.app.run()
        # except Exception as e:
        #     # raise e
        #     traceback.print_exc()
        # finally:
        # pass


# if __name__ == "__main__":
#     editor = Editor()
#     editor.run()
