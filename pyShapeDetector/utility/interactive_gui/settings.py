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

color_BBOX_selected_DEFAULT = np.array([0, 204.8, 0.0]) / 255
color_BBOX_unselected_DEFAULT = np.array([255.0, 0.0, 0.0]) / 255
COLOR_SELECTED_DEFAULT = np.array([178.5, 163.8, 0.0]) / 255
COLOR_UNSELECTED_DEFAULT = np.array([76.5, 76.5, 76.5]) / 255

# DEFAULT VALUES
_draw_boundary_lines = True
_pointcloud_size = 3
_mesh_show_back_face = True
_paint_selected = True
_color_selected = gui.Color(*COLOR_SELECTED_DEFAULT)
_paint_random = False
_debug = False
_verbose = False
_BBOX_expand = 0.0
_color_BBOX_selected = gui.Color(*color_BBOX_selected_DEFAULT)
_color_BBOX_unselected = gui.Color(*color_BBOX_unselected_DEFAULT)
_number_points_distance = 30
_random_color_brightness = 0.5
_original_color_brightness = 0.5
_highlight_ratio = 1.0
_number_undo_states = 10
_number_redo_states = 5


class Settings:
    @property
    def dict(self):
        self._dict

    def __init__(self, editor_instance: Editor, menu="Preferences", **kwargs):
        options = [
            ParameterBool(
                name="draw_boundary_lines",
                default=_draw_boundary_lines,
                on_update=self._cb_draw_boundary_lines,
            ),
            ParameterFloat(
                name="pointcloud_size",
                default=_pointcloud_size,
                on_update=self._cb_pointcloud_size,
                limits=(3, 15),
            ),
            ParameterBool(
                name="mesh_show_back_face",
                default=_mesh_show_back_face,
                on_update=self._cb_mesh_show_back_face,
            ),
            ParameterBool(
                name="paint_selected",
                default=_paint_selected,
                on_update=self._cb_paint_selected,
            ),
            ParameterColor(
                name="color_selected",
                default=_color_selected,
                on_update=self._cb_color_selected,
            ),
            ParameterBool(
                name="paint_random",
                default=_paint_random,
                on_update=self._cb_paint_random,
            ),
            ParameterBool(
                name="debug",
                default=_debug,
                on_update=self._cb_debug,
            ),
            ParameterBool(
                name="verbose",
                default=_verbose,
                on_update=self._cb_verbose,
            ),
            ParameterFloat(
                name="BBOX_expand",
                default=_BBOX_expand,
                limits=(0, 2),
                on_update=self._cb_BBOX_expand,
            ),
            ParameterColor(
                name="color_BBOX_selected",
                default=_color_BBOX_selected,
                on_update=self._cb_color_BBOX_selected,
            ),
            ParameterColor(
                name="color_BBOX_unselected",
                default=_color_BBOX_unselected,
                on_update=self._cb_color_BBOX_unselected,
            ),
            ParameterInt(
                name="number_points_distance",
                default=_number_points_distance,
                on_update=self._cb_number_points_distance,
                limits=(5, 50),
            ),
            ParameterFloat(
                name="random_color_brightness",
                default=_random_color_brightness,
                on_update=self._cb_random_color_brightness,
                limits=(0.001, 1),
            ),
            ParameterFloat(
                name="original_color_brightness",
                default=_original_color_brightness,
                on_update=self._cb_original_color_brightness,
                limits=(0.001, 1),
            ),
            ParameterFloat(
                name="highlight_ratio",
                default=_highlight_ratio,
                on_update=self._cb_highlight_ratio,
                limits=(0.01, 1),
            ),
            ParameterInt(
                name="number_undo_states",
                default=_number_undo_states,
                on_update=self._cb_number_undo_states,
                limits=(1, 10),
            ),
            ParameterInt(
                name="number_redo_states",
                default=_number_redo_states,
                on_update=self._cb_number_redo_states,
                limits=(1, 10),
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

    def _cb_pointcloud_size(self):
        from .element import ElementPointCloud

        point_size = self._dict["pointcloud_size"].value
        self._editor_instance.material_regular.point_size = point_size
        indices = np.where(
            [
                isinstance(element, ElementPointCloud)
                for element in self._editor_instance.elements
            ]
        )[0].tolist()
        self._editor_instance._update_elements(indices)

    def _cb_mesh_show_back_face(self):
        # self._editor_instance._reset_elements_in_gui()
        elements = self._editor_instance.elements
        elements.update_all()

    def _cb_paint_selected(self):
        self._editor_instance._update_elements(None)

    def _cb_paint_random(self):
        elements = self._editor_instance.elements
        for elem in elements:
            if self._dict["paint_random"].value:
                # print("printing new random colors")
                elem._color = np.random.random(3)
                elem._brightness = self._dict["random_color_brightness"].value
            else:
                elem._color = elem.color_original
                elem._brightness = self._dict["original_color_brightness"].value

        self._editor_instance._update_elements(None)

    def _cb_debug(self):
        pass

    def _cb_verbose(self):
        pass

    def _cb_BBOX_expand(self):
        self._editor_instance._update_current_bounding_box()

    def _cb_color_BBOX_selected(self):
        if self._editor_instance.is_current_selected:
            self._editor_instance._update_current_bounding_box()

    def _cb_color_BBOX_unselected(self):
        if not self._editor_instance.is_current_selected:
            self._editor_instance._update_current_bounding_box()

    def _cb_number_points_distance(self):
        for elem in self._editor_instance.elements:
            elem._get_distance_checker(0)
            # dist_checker = self._editor_instance._get_element_distances([elem.raw])[0]
            # elem.distance_checker = dist_checker

    def _cb_random_color_brightness(self):
        if self._dict["paint_random"].value:
            elements = self._editor_instance.elements

            for elem in elements:
                elem._brightness = self._dict["random_color_brightness"].value

        self._editor_instance._update_elements(None)

    def _cb_original_color_brightness(self):
        if not self._dict["paint_random"].value:
            elements = self._editor_instance.elements

            for elem in elements:
                elem._brightness = self._dict["original_color_brightness"].value

        self._editor_instance._update_elements(None)
        # elements.update_all()

    def _cb_highlight_ratio(self):
        # if not self._dict["paint_selected"]:
        self._editor_instance._update_elements(None)
        # self._editor_instance.elements.update_all()
        # indices = self._editor_instance.selected_indices
        # self._editor_instance._update_elements(indices)

    def _cb_number_undo_states(self):
        value = self._dict["number_undo_states"].value
        self._editor_instance._past_states = self._editor_instance._past_states[-value:]

    def _cb_number_redo_states(self):
        value = self._dict["number_redo_states"].value
        self._editor_instance._future_states = self._editor_instance._future_states[
            -value:
        ]

    def _cb_color_selected(self):
        indices = np.where(self._editor_instance.elements.selected)[0].tolist()
        self._editor_instance._update_elements(indices)

    def _cb_color_unselected(self):
        unselected = ~np.array(self._editor_instance.elements.selected)
        indices = np.where(unselected)[0].tolist()
        self._editor_instance._update_elements(indices)

    def get_element_color(self, is_selected, is_current, is_bbox=False):
        # highlight = self._dict["highlight_ratio"].value
        if not is_selected and not is_current:
            return self._dict["color_unselected"].value

        if not is_selected and is_current:
            return self._dict["color_unselected"].value  # / highlight

        if is_selected and not is_current:
            return self._dict["color_selected"].value

        if is_selected and is_current:
            return self._dict["color_selected"].value  # / highlight

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
