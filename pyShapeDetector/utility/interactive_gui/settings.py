import numpy as np
from open3d.visualization import gui
from .editor_app import Editor

COLOR_BBOX_SELECTED_DEFAULT = np.array([0, 204.8, 0.0]) / 255
COLOR_BBOX_UNSELECTED_DEFAULT = np.array([255.0, 0.0, 0.0]) / 255
COLOR_SELECTED_DEFAULT = np.array([178.5, 163.8, 0.0]) / 255
COLOR_UNSELECTED_DEFAULT = np.array([76.5, 76.5, 76.5]) / 255


class Settings:
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
    _random_color_brightness = 2 / 3
    _highlight_color_brightness = 0.3
    _number_undo_states = 10
    _number_redo_states = 5

    _options = [
        # (name, type, limits)
        ("draw_boundary_lines", bool, None),
        ("mesh_show_back_face", bool, None),
        ("paint_selected", bool, None),
        ("color_selected", gui.Color, None),
        # ("color_selected_current", gui.Color, None),
        # ("color_unselected", gui.Color, None),
        # ("color_unselected_current", gui.Color, None),
        ("paint_random", bool, None),
        ("debug", bool, None),
        ("verbose", bool, None),
        ("bbox_expand", float, (0, 2)),
        ("color_bbox_selected", gui.Color, None),
        ("color_bbox_unselected", gui.Color, None),
        ("number_points_distance", int, (5, 50)),
        ("random_color_brightness", float, (0.01, 1)),
        ("highlight_color_brightness", float, (0.01, 1)),
        ("number_undo_states", int, (1, 10)),
        ("number_redo_states", int, (1, 10)),
    ]

    def __init__(self, editor_instance: Editor, name="Preferences", **kwargs):
        self._editor_instance = editor_instance
        self._name = name

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def dict(self):
        return {
            name: self.__getattribute__(name)
            for name, _, _ in self._options
            if name[0] != "_"
        }

    def __repr__(self):
        lines = "\n".join("{!r}: {!r},".format(k, v) for k, v in self.dict.items())
        dict_str = "{\n" + lines + "}"
        return type(self).__name__ + "(" + dict_str + ")"

    @property
    def draw_boundary_lines(self):
        return self._draw_boundary_lines

    @draw_boundary_lines.setter
    def draw_boundary_lines(self, value):
        if not isinstance(value, bool):
            raise TypeError("draw_boundary_lines must be a boolean.")
        self._draw_boundary_lines = value

    def _cb_draw_boundary_lines(self, value):
        self.draw_boundary_lines = value
        self._editor_instance._update_plane_boundaries()

    @property
    def mesh_show_back_face(self):
        return self._mesh_show_back_face

    @mesh_show_back_face.setter
    def mesh_show_back_face(self, value):
        if not isinstance(value, bool):
            raise TypeError("mesh_show_back_face must be a boolean.")
        self._mesh_show_back_face = value

    def _cb_mesh_show_back_face(self, value):
        if self.mesh_show_back_face != value:
            self._editor_instance.print_debug(
                f"mesh_show_back_face set from {self.mesh_show_back_face} "
                f"to {value}, resetting..."
            )
            self._editor_instance._reset_elements_in_gui()
        self.mesh_show_back_face = value

    @property
    def paint_selected(self):
        return self._paint_selected

    @paint_selected.setter
    def paint_selected(self, value):
        if not isinstance(value, bool):
            raise TypeError("paint_selected must be a boolean.")
        self._paint_selected = value

    def _cb_paint_selected(self, value):
        self.paint_selected = value
        self._editor_instance._update_elements(None)

    @property
    def paint_random(self):
        return self._paint_random

    @paint_random.setter
    def paint_random(self, value):
        if not isinstance(value, bool):
            raise TypeError("paint_random must be a boolean.")
        self._paint_random = value

    def _cb_paint_random(self, value):
        if self.paint_random == value:
            return

        self._editor_instance.print_debug(
            f"paint_random set from {self.paint_random} to {value}, resetting..."
        )
        self.paint_random = value
        self._editor_instance._reset_elements_in_gui()

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        if not isinstance(value, bool):
            raise TypeError("debug must be a boolean.")
        self._debug = value

    def _cb_debug(self, value):
        self.debug = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, bool):
            raise TypeError("verbose must be a boolean.")
        self._verbose = value

    def _cb_verbose(self, value):
        self.verbose = value

    @property
    def bbox_expand(self):
        return self._bbox_expand

    @bbox_expand.setter
    def bbox_expand(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("bbox_expand must be a float or an int.")
        self._bbox_expand = float(value)

    def _cb_bbox_expand(self, value):
        self.bbox_expand = value
        self._editor_instance._update_current_bounding_box()

    @property
    def color_bbox_selected(self):
        color = self._color_bbox_selected
        return (color.red, color.green, color.blue)

    @color_bbox_selected.setter
    def color_bbox_selected(self, values):
        if isinstance(values, gui.Color):
            self._color_bbox_selected = values
        elif isinstance(values, (tuple, list, np.ndarray)) and len(values) == 3:
            self._color_bbox_selected = gui.Color(*np.clip(values, 0, 1))
        else:
            raise TypeError("color_selected should be a list or tuple of 3 values.")

    def _cb_color_bbox_selected(self, values):
        self.color_bbox_selected = values
        if self._editor_instance.is_current_selected:
            self._editor_instance._update_current_bounding_box()

    @property
    def color_bbox_unselected(self):
        color = self._color_bbox_unselected
        return (color.red, color.green, color.blue)

    @color_bbox_unselected.setter
    def color_bbox_unselected(self, values):
        if isinstance(values, gui.Color):
            self._color_bbox_unselected = values
        elif isinstance(values, (tuple, list, np.ndarray)) and len(values) == 3:
            self._color_bbox_unselected = gui.Color(*np.clip(values, 0, 1))
        else:
            raise TypeError("color_selected should be a list or tuple of 3 values.")

    def _cb_color_bbox_unselected(self, values):
        self.color_bbox_unselected = values
        if not self._editor_instance.is_current_selected:
            self._editor_instance._update_current_bounding_box()

    @property
    def number_points_distance(self):
        return self._number_points_distance

    @number_points_distance.setter
    def number_points_distance(self, value):
        if not isinstance(value, int):
            raise TypeError("number_points_distance must be an integer.")
        self._number_points_distance = value

    def _cb_number_points_distance(self, value):
        self.number_points_distance = value
        for elem in self._editor_instance.elements:
            dist_checker = self._editor_instance._get_element_distances([elem["raw"]])[
                0
            ]
            elem["distance_checker"] = dist_checker

    @property
    def random_color_brightness(self):
        return self._random_color_brightness

    @random_color_brightness.setter
    def random_color_brightness(self, value):
        if not isinstance(value, (float, int)) or not (0 <= value <= 1):
            raise ValueError(
                "random_color_brightness must be a float in the range [0, 1]."
            )
        self._random_color_brightness = float(value)

    def _cb_random_color_brightness(self, value):
        if abs(self.random_color_brightness - value) < 1e-5:
            return

        if self.paint_random:
            self._editor_instance.print_debug(
                f"random_color_brightness set from {self.random_color_brightness} "
                f"to {value}, resetting..."
            )
            self._editor_instance._reset_elements_in_gui()

        self.random_color_brightness = value

    @property
    def highlight_color_brightness(self):
        return self._highlight_color_brightness

    @highlight_color_brightness.setter
    def highlight_color_brightness(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("highlight_color_brightness must be a float or an int.")
        self._highlight_color_brightness = max(float(value), 0)

    def _cb_highlight_color_brightness(self, value):
        if abs(self.highlight_color_brightness - value) < 1e-5:
            return

        self.highlight_color_brightness = value
        if not self.paint_selected:
            indices = np.where(self._editor_instance.selected)[0].tolist()
            self._editor_instance._update_elements(indices)

    @property
    def number_undo_states(self):
        return self._number_undo_states

    @number_undo_states.setter
    def number_undo_states(self, value):
        if not isinstance(value, int):
            raise TypeError("number_undo_states must be an integer.")
        self._number_undo_states = value

    def _cb_number_undo_states(self, value):
        self.number_undo_states = value
        self._editor_instance._past_states = self._editor_instance._past_states[-value:]

    @property
    def number_redo_states(self):
        return self._number_redo_states

    @number_redo_states.setter
    def number_redo_states(self, value):
        if not isinstance(value, int):
            raise TypeError("number_redo_states must be an integer.")
        self._number_redo_states = value
        self._number_undo_states = value

    def _cb_number_redo_states(self, value):
        self.number_redo_states = value
        self._editor_instance._future_states = self._editor_instance._future_states[
            -value:
        ]

    @property
    def color_selected(self):
        color = self._color_selected
        return (color.red, color.green, color.blue)

    @color_selected.setter
    def color_selected(self, values):
        if isinstance(values, gui.Color):
            self._color_selected = values
        elif isinstance(values, (tuple, list, np.ndarray)) and len(values) == 3:
            self._color_selected = gui.Color(*np.clip(values, 0, 1))
        else:
            raise TypeError("color_selected should be a list or tuple of 3 values.")

    def _cb_color_selected(self, values):
        self.color_selected = values
        indices = np.where(self._editor_instance.selected)[0].tolist()
        self._editor_instance._update_elements(indices)

    # @property
    # def color_selected_current(self):
    #     color = self._color_selected_current
    #     return (color.red, color.green, color.blue)

    # @color_selected_current.setter
    # def color_selected_current(self, values):
    #     if isinstance(values, gui.Color):
    #         self._color_selected_current = values
    #     elif isinstance(values, (tuple, list, np.ndarray)) and len(values) == 3:
    #         self._color_selected_current = gui.Color(*np.clip(values, 0, 1))
    #     else:
    #         raise TypeError(
    #             "color_selected_current should be a list or tuple of 3 values."
    #         )

    # def _cb_color_selected_current(self, values):
    #     self.color_selected_current = values
    #     self._editor_instance._update_elements(self._editor_instance.i)

    @property
    def color_unselected(self):
        color = self._color_unselected
        return (color.red, color.green, color.blue)

    @color_unselected.setter
    def color_unselected(self, values):
        if isinstance(values, gui.Color):
            self._color_unselected = values
        elif isinstance(values, (tuple, list, np.ndarray)) and len(values) == 3:
            self._color_unselected = gui.Color(*np.clip(values, 0, 1))
        else:
            raise TypeError("color_unselected should be a list or tuple of 3 values.")

    def _cb_color_unselected(self, values):
        self.color_unselected = values
        unselected = ~np.array(self._editor_instance.selected)
        indices = np.where(unselected)[0].tolist()
        self._editor_instance._update_elements(indices)

    # @property
    # def color_unselected_current(self):
    #     color = self._color_unselected_current
    #     return (color.red, color.green, color.blue)

    # @color_unselected_current.setter
    # def color_unselected_current(self, values):
    #     if isinstance(values, gui.Color):
    #         self._color_unselected_current = values
    #     elif isinstance(values, (tuple, list, np.ndarray)) and len(values) == 3:
    #         self._color_unselected_current = gui.Color(*np.clip(values, 0, 1))
    #     else:
    #         raise TypeError(
    #             "color_unselected_current should be a list or tuple of 3 values."
    #         )

    # def _cb_color_unselected_current(self, values):
    #     self.color_unselected_current = values
    #     self._editor_instance._update_elements(self._editor_instance.i)

    def get_element_color(self, is_selected, is_current, is_bbox=False):
        if not is_selected and not is_current:
            return np.array(self.color_unselected)

        if not is_selected and is_current:
            # return self.color_unselected_current
            return np.array(self.color_unselected) / self.highlight_color_brightness

        if is_selected and not is_current:
            return np.array(self.color_selected)

        if is_selected and is_current:
            # return self.color_selected_current
            return np.array(self.color_selected) / self.highlight_color_brightness

    def _create_panel(self):
        window = self._editor_instance._window
        em = window.theme.font_size
        separation_height = int(round(0.5 * em))

        # _panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        _panel_collapsable = gui.CollapsableVert(
            "Preferences", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )

        for key, value_type, limits in self._options:
            name_pretty = key.replace("_", " ").capitalize()
            label = gui.Label(name_pretty)
            value = getattr(self, key)
            try:
                cb = getattr(self, "_cb_" + key)  # callback function
            except AttributeError:
                continue

            if value_type is bool:
                element = gui.Checkbox(name_pretty + "?")
                element.checked = value
                element.set_on_checked(cb)
            elif value_type is int:
                slider = gui.Slider(gui.Slider.INT)
                slider.set_limits(*limits)
                slider.int_value = value
                slider.set_on_value_changed(cb)

                element = gui.VGrid(2, 0.25 * em)
                element.add_child(label)
                element.add_child(slider)
            elif value_type is float:
                slider = gui.Slider(gui.Slider.DOUBLE)
                slider.set_limits(*limits)
                slider.double_value = value
                slider.set_on_value_changed(cb)

                element = gui.VGrid(2, 0.25 * em)
                element.add_child(label)
                element.add_child(slider)
            elif value_type is gui.Color:
                color_selector = gui.ColorEdit()
                color_selector.color_value = gui.Color(*value)
                color_selector.set_on_value_changed(cb)

                element = gui.VGrid(2, 0.25 * em)
                element.add_child(label)
                element.add_child(color_selector)
            else:
                continue

            _panel_collapsable.add_child(element)
            _panel_collapsable.add_fixed(separation_height)

        _panel_collapsable.visible = False
        self._editor_instance._right_side_panel.add_child(_panel_collapsable)
        self._panel = _panel_collapsable

    def _create_menu(self):
        editor_instance = self._editor_instance

        self._create_panel()

        preferences_binding = editor_instance._hotkeys.find_binding("Show Preferences")

        self._on_menu_id = editor_instance._add_menu_item(
            self._name,
            preferences_binding.description_and_instruction,
            self._on_menu_toggle,
        )

    def _on_menu_toggle(self):
        window = self._editor_instance._window
        menubar = self._editor_instance._menubar
        self._panel.visible = not self._panel.visible
        menubar.set_checked(self._on_menu_id, self._panel.visible)
        window.set_needs_layout()
