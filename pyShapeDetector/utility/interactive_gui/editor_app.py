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


from .helpers import (
    extract_element_colors,
    set_element_colors,
    get_painted_element,
    get_distance_checker,
)


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

    Extra Parameters (these can also be set in real time)
    -----------------------------------------------------
    draw_boundary_lines : boolean, optional
        If True, draws the boundaries of planes as LineSets. Default: False.
    mesh_show_back_face : optional, boolean
        If True, shows back faces of surfaces and meshes. Default: True.
    BBOX_expand : float, optional
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
        self._extensions = None
        self._last_used_extension = None
        self._select_filter = lambda x: True
        self.window_name = window_name
        self.return_finish_flag = return_finish_flag
        self._submenu_id_generator = itertools.count(1, 1)
        self._submenus = {}
        self._closing_app = False

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
            extension = Extension(function_or_descriptor)
            extension.add_to_application(self)

        except Exception:
            warnings.warn(f"Could not create extension {function_or_descriptor}, got:")
            traceback.print_exc()

    @property
    def extension_key_mappings(self):
        key_mappings = dict()
        if self.extensions is not None:
            for extension in self.extensions:
                if extension.hotkey is not None:
                    key_mappings[extension.hotkey] = extension
        return key_mappings

    def _create_extension_menu_items(self):
        """Add menu items for each extension"""
        if self.extensions is not None:
            for extension in self.extensions:
                try:
                    extension.add_menu_item()

                except Exception:
                    warnings.warn(
                        f"Could not add extension {extension} to application, got:"
                    )
                    traceback.print_exc()
                    self._extensions.pop(extension)

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

    def get_elements(self):
        return [elem.raw for elem in self.elements]

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
    def is_current_selected(self, boolean_value):
        if not isinstance(boolean_value, bool):
            raise RuntimeError(f"Expected boolean, got {type(boolean_value)}.")
        if self.current_element is None:
            raise RuntimeError(
                f"Error setting selected, current element does not exist."
            )
        self.current_element.selected = boolean_value

    def _reset_on_key(self):
        self._scene.set_on_key(self._hotkeys._on_key)

    def _close_dialog(self):
        """Closes dialog, if any, then resets keys to original hotkeys."""
        self._window.set_on_key(None)
        self._window.close_dialog()
        self._reset_on_key()

    def _add_geometry_to_scene(self, elem):
        from open3d.geometry import LineSet, AxisAlignedBoundingBox, OrientedBoundingBox

        if isinstance(elem, (LineSet, AxisAlignedBoundingBox, OrientedBoundingBox)):
            mat = self.material_line
        else:
            mat = self.material_regular
        self._scene.scene.add_geometry(str(id(elem)), elem, mat)

    def _remove_geometry_from_scene(self, elem):
        self._scene.scene.remove_geometry(str(id(elem)))

    # def _pop_elements(self, indices, from_gui=False):
    #     # update_old = self.i in indices
    #     # idx_new = self.i
    #     elements_popped = []
    #     for n, i in enumerate(indices):
    #         elem = self._elements.pop(i - n)
    #         if from_gui:
    #             elem.remove_from_scene()
    #         elements_popped.append(elem.raw)

    #     idx_new = self.i - sum([idx < self.i for idx in indices])
    #     self.print_debug(
    #         f"popped: {indices}",
    #     )
    #     self.print_debug(f"old index: {self.i}, new index: {idx_new}")

    #     if len(self.elements) == 0:
    #         warnings.warn("No elements left after popping, not updating.")
    #         self.i = 0
    #         self.i_old = 0
    #     else:
    #         idx_new = max(min(idx_new, len(self.elements) - 1), 0)
    #         self._update_current_idx(idx_new, update_old=False)
    #         self.i_old = self.i
    #     return elements_popped

    def _insert_elements(self, elems_raw, indices=None, selected=False, to_gui=False):
        from .element import Element

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

            is_current = self._started and (self.i == idx)

            elem = Element.get_from_type(self, elems_raw[i], selected[i], is_current)

            self.print_debug(f"Added {elem.raw} at index {idx}.", require_verbose=True)
            self._elements.insert_multiple(idx, elem)

        for idx in indices:
            # Updating vis explicitly in order not to remove it
            self._update_elements(idx, update_gui=False)
            if to_gui:
                self.elements[idx].add_to_scene()

        self.print_debug(f"{len(self.elements)} now.", require_verbose=True)

        idx_new = max(min(idx_new, len(self.elements) - 1), 0)
        self._update_current_idx(idx_new, update_old=self._started)
        self.i_old = self.i

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

    def _on_close(self):
        if not self._closing_app:
            self._internal_functions._dict["Quit"].callback()
            return False

        self.elements.insert_multiple(self._hidden_elements.raw, to_gui=False)
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

        self.material_regular = rendering.MaterialRecord()
        self.material_regular.base_color = [1.0, 1.0, 1.0, 1.0]  # White color
        self.material_regular.shader = "defaultUnlit"
        self.material_regular.point_size = 3 * self._window.scaling

        self.material_line = rendering.MaterialRecord()
        self.material_line.shader = "unlitLine"
        self.material_line.line_width = 1.5 * self._window.scaling

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
        # self._create_extension_menu_items()

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

    def _update_current_bounding_box(self):
        """Remove bounding box and get new one for current element."""

        self.print_debug("Updating bounding box...", require_verbose=True)

        name = "CurrentBoundingBox"

        if self._current_bbox is not None:
            self._scene.scene.remove_geometry(name)

        if self.current_element is None:
            self._current_bbox = None
            return

        self._current_bbox = self.current_element._get_bbox().as_open3d

        if self._current_bbox is not None:
            self._scene.scene.add_geometry(name, self._current_bbox, self.material_line)

    def _update_elements(self, indices, update_gui=True):
        num_elems = len(self.elements)

        if indices is None:
            indices = range(len(self.elements))

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

        # This is not called on the main thread, so we need to
        # post to the main thread to safely access UI items.
        def update_label():
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

    def _reset_elements_in_gui(self, reset_fixed=False):
        """Prepare elements for visualization"""

        current_idx = copy.copy(min(self.i, len(self.elements) - 1))
        if reset_fixed:
            self.elements_fixed.update_all()
        else:
            self.elements.update_all()

        # if startup:
        #     current_idx = 0
        #     elems_raw = self._elements_input
        #     pre_selected = self._pre_selected

        # else:
        #     # pre_selected = [False] * len(elems_raw)
        #     current_idx = copy.copy(self.i)
        #     pre_selected = self.elements.selected
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings("ignore", message="No elements left")
        #         elems_raw = self.elements.pop_multiple(
        #             range(len(self.elements)), from_gui=True
        #         )
        #     assert len(self.elements) == 0

        # self.print_debug(
        #     f"\ninserting elements at startup, there are {len(elems_raw)}.",
        #     require_verbose=True,
        # )
        # # self._insert_elements(elems_raw, selected=pre_selected, to_gui=True)
        # self.elements.insert_multiple(elems_raw, selected=pre_selected, to_gui=True)
        # self.print_debug(f"Finished inserting {len(self.elements)} elements.")

        self._update_plane_boundaries()
        self._update_current_bounding_box()

        self._update_current_idx(current_idx)
        # self._started = True

    def run(self):
        self.print_debug(f"Starting {type(self).__name__}.")

        # Set up the gui
        self._setup_window_and_scene()

        for elem in self.elements + self.elements_fixed:
            elem.add_to_scene()

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
