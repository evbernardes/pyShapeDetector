import warnings
import numpy as np
from open3d.visualization import gui
from .editor_app import Editor
from .parameter import (
    ParameterBool,
    ParameterInt,
    ParameterFloat,
    # ParameterNDArray,
    # ParameterOptions,
    ParameterColor,
)

COLOR_BBOX_SELECTED_DEFAULT = np.array([0, 204.8, 0.0]) / 255
COLOR_BBOX_UNSELECTED_DEFAULT = np.array([255.0, 0.0, 0.0]) / 255
COLOR_SELECTED_DEFAULT = np.array([178.5, 163.8, 0.0]) / 255
COLOR_UNSELECTED_DEFAULT = np.array([76.5, 76.5, 76.5]) / 255

# DEFAULT VALUES
_draw_boundary_lines = True
_mesh_show_back_face = True
_paint_selected = True
_color_selected = gui.Color(*COLOR_SELECTED_DEFAULT)
# _color_selected_current = gui.Color(*COLOR_SELECTED_CURRENT)
# _color_unselected = gui.Color(*COLOR_UNSELECTED_DEFAULT)
# _color_unselected_current = gui.Color(*COLOR_UNSELECTED_CURRENT)
_paint_random = False
_debug = False
_verbose = False
_bbox_expand = 0.0
_color_bbox_selected = gui.Color(*COLOR_BBOX_SELECTED_DEFAULT)
_color_bbox_unselected = gui.Color(*COLOR_BBOX_UNSELECTED_DEFAULT)
_number_points_distance = 30
_random_color_brightness = 0.3
_highlight_color_brightness = 0.3
_number_undo_states = 10
_number_redo_states = 5


class Settings:
    @property
    def dict(self):
        self._dict

    def __init__(self, editor_instance: Editor, menu="Preferences", **kwargs):
        options = [
            ParameterBool(
                "draw_boundary_lines",
                {
                    "default": _draw_boundary_lines,
                    "on_update": self._cb_draw_boundary_lines,
                },
            ),
            ParameterBool(
                "mesh_show_back_face",
                {
                    "default": _mesh_show_back_face,
                    "on_update": self._cb_mesh_show_back_face,
                },
            ),
            ParameterBool(
                "paint_selected",
                {
                    "default": _paint_selected,
                    "on_update": self._cb_paint_selected,
                },
            ),
            ParameterColor(
                "color_selected",
                {
                    "default": _color_selected,
                    "on_update": self._cb_color_selected,
                },
            ),
            # ParameterColor(
            #     "_color_unselected",
            #     {
            #         "default": _color_unselected,
            #         "on_update": self._cb_color_unselected,
            #     },
            # ),
            # ParameterColor(
            #     "color_selected_current",
            #     {
            #         "default": _color_selected_current,
            #         "on_update": self._cb_color_selected_current,
            #     },
            # ),
            # ParameterColor(
            #     "color_unselected_current",
            #     {
            #         "default": _color_unselected_current,
            #         "on_update": self._cb_color_unselected_current,
            #     },
            # ),
            ParameterBool(
                "paint_random",
                {
                    "default": _paint_random,
                    "on_update": self._cb_paint_random,
                },
            ),
            ParameterBool(
                "debug",
                {
                    "default": _debug,
                    "on_update": self._cb_debug,
                },
            ),
            ParameterBool(
                "verbose",
                {
                    "default": _verbose,
                    "on_update": self._cb_verbose,
                },
            ),
            ParameterFloat(
                "bbox_expand",
                {
                    "default": _bbox_expand,
                    "limits": (0, 2),
                    "on_update": self._cb_bbox_expand,
                },
            ),
            ParameterColor(
                "color_bbox_selected",
                {
                    "default": _color_bbox_selected,
                    "on_update": self._cb_color_bbox_selected,
                },
            ),
            ParameterColor(
                "color_bbox_unselected",
                {
                    "default": _color_bbox_unselected,
                    "on_update": self._cb_color_bbox_unselected,
                },
            ),
            ParameterInt(
                "number_points_distance",
                {
                    "default": _number_points_distance,
                    "on_update": self._cb_number_points_distance,
                    "limits": (5, 50),
                },
            ),
            ParameterFloat(
                "random_color_brightness",
                {
                    "default": _random_color_brightness,
                    "on_update": self._cb_random_color_brightness,
                    "limits": (0.001, 1),
                },
            ),
            ParameterFloat(
                "highlight_color_brightness",
                {
                    "default": _highlight_color_brightness,
                    "on_update": self._cb_highlight_color_brightness,
                    "limits": (0.01, 1),
                },
            ),
            ParameterInt(
                "number_undo_states",
                {
                    "default": _number_undo_states,
                    "on_update": self._cb_number_undo_states,
                    "limits": (1, 10),
                },
            ),
            ParameterInt(
                "number_redo_states",
                {
                    "default": _number_redo_states,
                    "on_update": self._cb_number_redo_states,
                    "limits": (1, 10),
                },
            ),
        ]

        self._dict = {param.name: param for param in options}
        self._editor_instance = editor_instance
        self._menu = menu

        for key, value in kwargs.items():
            if key in self._dict:
                try:
                    self._dict[key].value = value
                except Exception:
                    warnings.warn(
                        f"Could not initialize value of preference {key} to {value}."
                    )
                    setattr(self, key, value)

    def _cb_draw_boundary_lines(self):
        self._editor_instance._update_plane_boundaries()

    def _cb_mesh_show_back_face(self):
        self._editor_instance._reset_elements_in_gui()

    def _cb_paint_selected(self):
        self._editor_instance._update_elements(None)

    def _cb_paint_random(self):
        self._editor_instance._reset_elements_in_gui()

    def _cb_debug(self):
        pass

    def _cb_verbose(self):
        pass

    def _cb_bbox_expand(self):
        self._editor_instance._update_current_bounding_box()

    def _cb_color_bbox_selected(self):
        if self._editor_instance.is_current_selected:
            self._editor_instance._update_current_bounding_box()

    def _cb_color_bbox_unselected(self):
        if not self._editor_instance.is_current_selected:
            self._editor_instance._update_current_bounding_box()

    def _cb_number_points_distance(self):
        for elem in self._editor_instance.elements:
            dist_checker = self._editor_instance._get_element_distances([elem["raw"]])[
                0
            ]
            elem["distance_checker"] = dist_checker

    def _cb_random_color_brightness(self):
        if self._dict["paint_random"]:
            self._editor_instance._reset_elements_in_gui()

    def _cb_highlight_color_brightness(self):
        if not self._dict["paint_selected"]:
            indices = np.where(self._editor_instance.selected)[0].tolist()
            self._editor_instance._update_elements(indices)

    def _cb_number_undo_states(self):
        value = self._dict["number_undo_states"]
        self._editor_instance._past_states = self._editor_instance._past_states[-value:]

    def _cb_number_redo_states(self):
        value = self._dict["number_redo_states"]
        self._editor_instance._future_states = self._editor_instance._future_states[
            -value:
        ]

    def _cb_color_selected(self):
        indices = np.where(self._editor_instance.selected)[0].tolist()
        self._editor_instance._update_elements(indices)

    def _cb_color_unselected(self):
        unselected = ~np.array(self._editor_instance.selected)
        indices = np.where(unselected)[0].tolist()
        self._editor_instance._update_elements(indices)

    def get_element_color(self, is_selected, is_current, is_bbox=False):
        highlight = self._dict["highlight_color_brightness"].value
        if not is_selected and not is_current:
            return self._dict["color_unselected"].value

        if not is_selected and is_current:
            return self._dict["color_unselected"].value / highlight

        if is_selected and not is_current:
            return self._dict["color_selected"].value

        if is_selected and is_current:
            return self._dict["color_selected"].value / highlight

    def _create_panel(self):
        window = self._editor_instance._window
        em = window.theme.font_size
        separation_height = int(round(0.5 * em))

        _panel_collapsable = gui.CollapsableVert(
            "Preferences", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )

        for preference in self._dict.values():
            _panel_collapsable.add_child(preference.get_gui_element(window))
            _panel_collapsable.add_fixed(separation_height)

        _panel_collapsable.visible = False
        self._editor_instance._right_side_panel.add_child(_panel_collapsable)
        self._panel = _panel_collapsable

    def _create_menu(self):
        editor_instance = self._editor_instance

        self._create_panel()

        self._preferences_binding = editor_instance._internal_functions._dict[
            "Show Preferences"
        ]
        self._preferences_binding._menu = self._menu
        self._preferences_binding.add_to_menu(editor_instance)

    def _on_menu_toggle(self):
        window = self._editor_instance._window
        menubar = self._editor_instance._menubar
        self._panel.visible = not self._panel.visible
        menubar.set_checked(self._preferences_binding._menu_id, self._panel.visible)
        window.set_needs_layout()
