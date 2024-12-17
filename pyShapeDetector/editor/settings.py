import warnings
import numpy as np
from typing import TYPE_CHECKING
from open3d.visualization import gui
from open3d.visualization.rendering import MaterialRecord

if TYPE_CHECKING:
    from .editor_app import Editor

from .parameter import (
    ParameterBool,
    ParameterInt,
    ParameterFloat,
    ParameterNDArray,
    ParameterOptions,
    ParameterColor,
    ParameterPanel,
)

COLOR_BBOX_SELECTED_DEFAULT = np.array([0, 204.8, 0.0]) / 255
# color_BBOX_unselected_DEFAULT = np.array([255.0, 0.0, 0.0]) / 255
COLOR_BBOX_UNSELECTED_DEFAULT = np.array([146.0, 3.0, 3.0]) / 255
COLOR_SELECTED_DEFAULT = np.array([178.5, 163.8, 0.0]) / 255
COLOR_UNSELECTED_DEFAULT = np.array([76.5, 76.5, 76.5]) / 255

# DEFAULT VALUES
_PointCloud_density = 0.00224
_draw_boundary_lines = True
_line_width = 7
_PointCloud_point_size = 5
_mesh_show_back_face = True
_paint_selected = True
_color_selected = gui.Color(*COLOR_SELECTED_DEFAULT)
_paint_random = False
_debug = False
_verbose = False
_BBOX_expand = 0.01
_color_BBOX_selected = gui.Color(*COLOR_BBOX_SELECTED_DEFAULT)
_color_BBOX_unselected = gui.Color(*COLOR_BBOX_UNSELECTED_DEFAULT)
_show_BBOX = True
_show_BBOX_axes = False
_BBOX_axes_width = 0.01
_number_points_distance = 30
_random_color_brightness = 0.5
_original_color_brightness = 0.5
_highlight_ratio = 1.0
_number_undo_states = 10
_number_redo_states = 5


###################################################
# Test extension for testing reference parameters #
# Uncomment and this and the init when testing    #
###################################################
# def test(
#     test_int_no_limits,
#     test_int_limits,
#     test_float_no_limits,
#     test_float_limits,
#     test_color,
#     test_options,
#     test_array_int_shape_2_3,
#     test_array_float_shape_2_3,
#     test_array_int_ndim1,
#     test_array_shape_7_3,
# ):
#     return []


# test_extension = {
#     "function": test,
#     "name": "Testing reference widgets",
#     "menu": "test",
#     "inputs": None,
#     "parameters": {
#         "test_int_no_limits": {"type": "preference"},
#         "test_int_limits": {"type": "preference"},
#         "test_float_no_limits": {"type": "preference"},
#         "test_float_limits": {"type": "preference"},
#         "test_color": {"type": "preference"},
#         "test_options": {"type": "preference"},
#         "test_array_int_shape_2_3": {"type": "preference"},
#         "test_array_float_shape_2_3": {"type": "preference"},
#         "test_array_int_ndim1": {"type": "preference"},
#         "test_array_shape_7_3": {"type": "preference"},
#     },
# }


class Settings:
    @property
    def dict(self):
        self._dict

    def get_setting(self, key):
        if key not in self._dict:
            warnings.warn(f"Tried getting non existing preference {key}")
            return None
        return self._dict[key].value

    def set_setting(self, key, value):
        if key not in self._dict:
            warnings.warn(f"Tried getting non existing preference {key}")
            return None
        self._dict[key].value = value

        # TODO: This might be unnecessary:
        self._dict[key].on_update(value)

        self._dict[key]._update_references()

    def get_material(self, key):
        if key not in self._materials:
            warnings.warn(f"Material '{key}' not available")
            return None
        return self._materials[key]

    def _update_materials(self):
        scaling = self._editor_instance._window.scaling
        self._materials["regular"].point_size = (
            self.get_setting("PointCloud_point_size") * scaling
        )

        self._materials["line"].line_width = self.get_setting("line_width") * scaling

    def print_debug(self, text: str, require_verbose: bool = False):
        is_debug_activated = self.get_setting("debug")
        is_verbose_activated = self.get_setting("verbose")

        if not is_debug_activated or (require_verbose and not is_verbose_activated):
            return

        text = str(text)
        print("[DEBUG] " + text)

    def __init__(self, editor_instance: "Editor", menu="Preferences", **kwargs):
        options = [
            ###########################################
            # Test Parameters, uncomment when testing #
            ###########################################
            # ParameterInt(
            #     name="test_int_no_limits",
            #     default=2,
            #     on_update=lambda x: print(x),
            #     subpanel="test",
            # ),
            # ParameterInt(
            #     name="test_int_limits",
            #     default=3,
            #     limits=(-10, 10),
            #     on_update=lambda x: print(x),
            #     subpanel="test",
            # ),
            # ParameterFloat(
            #     name="test_float_no_limits",
            #     default=2,
            #     on_update=lambda x: print(x),
            #     subpanel="test",
            # ),
            # ParameterFloat(
            #     name="test_float_limits",
            #     default=3,
            #     limits=(-10, 10),
            #     on_update=lambda x: print(x),
            #     subpanel="test",
            # ),
            # ParameterColor(
            #     name="test_color",
            #     default=(0, 0, 1),
            #     on_update=lambda x: print(x),
            #     subpanel="test",
            # ),
            # ParameterOptions(
            #     name="test_options",
            #     options=["a", "b", "c"],
            #     default="c",
            #     on_update=lambda x: print(x),
            #     subpanel="test",
            # ),
            # ParameterNDArray(
            #     name="test_array_int_shape_2_3",
            #     default=[[0, 0, 1], [0, 20, 0]],
            #     on_update=lambda x: print(x),
            #     subpanel="test",
            # ),
            # ParameterNDArray(
            #     name="test_array_float_shape_2_3",
            #     default=[[0, 0, 1], [0.0, 20, 0]],
            #     on_update=lambda x: print(x),
            #     subpanel="test",
            # ),
            # ParameterNDArray(
            #     name="test_array_int_ndim1",
            #     default=[0, 0, 1],
            #     on_update=lambda x: print(x),
            #     subpanel="test",
            # ),
            # ParameterNDArray(
            #     name="test_array_shape_7_3",
            #     default=np.zeros((7, 3)),
            #     on_update=lambda x: print(x),
            #     subpanel="test",
            # ),
            ##########################
            # End of Test Parameters #
            ##########################
            ParameterFloat(
                name="PointCloud_density",
                default=_PointCloud_density,
                on_update=self._cb_PointCloud_density,
            ),
            ParameterBool(
                name="draw_boundary_lines",
                default=_draw_boundary_lines,
                on_update=self._cb_draw_boundary_lines,
            ),
            ParameterFloat(
                name="line_width",
                default=_line_width,
                on_update=self._cb_line_width,
                limits=(1.5, 15),
            ),
            ParameterFloat(
                name="PointCloud_point_size",
                default=_PointCloud_point_size,
                on_update=self._cb_PointCloud_point_size,
                limits=(3, 15),
            ),
            ParameterBool(
                name="mesh_show_back_face",
                default=_mesh_show_back_face,
                on_update=self._cb_mesh_show_back_face,
            ),
            ParameterBool(
                name="debug",
                default=_debug,
                on_update=self._cb_debug,
                subpanel="Debug",
            ),
            ParameterBool(
                name="verbose",
                default=_verbose,
                on_update=self._cb_verbose,
                subpanel="Debug",
            ),
            ParameterFloat(
                name="BBOX_expand",
                default=_BBOX_expand,
                limits=(0, 2),
                on_update=self._cb_BBOX_expand,
                subpanel="Bounding Box and axes",
            ),
            ParameterColor(
                name="color_BBOX_selected",
                default=_color_BBOX_selected,
                on_update=self._cb_color_BBOX_selected,
                subpanel="Bounding Box and axes",
            ),
            ParameterColor(
                name="color_BBOX_unselected",
                default=_color_BBOX_unselected,
                on_update=self._cb_color_BBOX_unselected,
                subpanel="Bounding Box and axes",
            ),
            ParameterBool(
                name="show_BBOX",
                default=_show_BBOX,
                on_update=self._cb_show_BBOX,
                subpanel="Bounding Box and axes",
            ),
            ParameterBool(
                name="show_BBOX_axes",
                default=_show_BBOX_axes,
                on_update=self._cb_show_BBOX_axes,
                subpanel="Bounding Box and axes",
            ),
            ParameterFloat(
                name="BBOX_axes_width",
                default=_BBOX_axes_width,
                on_update=self._cb_BBOX_axes_width,
                subpanel="Bounding Box and axes",
                limits=(0.01, 0.5),
            ),
            ParameterBool(
                name="paint_selected",
                default=_paint_selected,
                on_update=self._cb_paint_selected,
                subpanel="Color",
            ),
            ParameterColor(
                name="color_selected",
                default=_color_selected,
                on_update=self._cb_color_selected,
                subpanel="Color",
            ),
            ParameterBool(
                name="paint_random",
                default=_paint_random,
                on_update=self._cb_paint_random,
                subpanel="Color",
            ),
            ParameterFloat(
                name="random_color_brightness",
                default=_random_color_brightness,
                on_update=self._cb_random_color_brightness,
                limits=(0.001, 1),
                subpanel="Color",
            ),
            ParameterFloat(
                name="original_color_brightness",
                default=_original_color_brightness,
                on_update=self._cb_original_color_brightness,
                limits=(0.001, 1),
                subpanel="Color",
            ),
            ParameterFloat(
                name="highlight_ratio",
                default=_highlight_ratio,
                on_update=self._cb_highlight_ratio,
                limits=(0.01, 1),
                subpanel="Color",
            ),
            ParameterInt(
                name="number_points_distance",
                default=_number_points_distance,
                on_update=self._cb_number_points_distance,
                limits=(5, 50),
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

        material_regular = MaterialRecord()
        material_regular.base_color = [1.0, 1.0, 1.0, 1.0]  # White color
        material_regular.shader = "defaultUnlit"

        material_line = MaterialRecord()
        material_line.shader = "unlitLine"

        self._materials = {
            "regular": material_regular,
            "line": material_line,
        }

        ###################################################
        # Test extension for testing reference parameters #
        # Uncomment and this and the definition to test   #
        ###################################################
        # from .extension import Extension
        # extension = Extension(test_extension, self)
        # extension.add_to_application(editor_instance)

    def _cb_PointCloud_density(self, value):
        pass

    def _cb_draw_boundary_lines(self, value):
        self._editor_instance._update_plane_boundaries()

    def _cb_line_width(self, value):
        # line_width = self._dict["line_width"].value

        self._update_materials()
        self._editor_instance._update_extra_elements(planes_boundaries=True)

    def _cb_PointCloud_point_size(self, value):
        from .element import ElementPointCloud

        self._update_materials()
        indices = np.where(
            [
                isinstance(element, ElementPointCloud)
                for element in self._editor_instance.elements
            ]
        )[0].tolist()
        self._editor_instance.elements.update_indices(indices)

    def _cb_mesh_show_back_face(self, value):
        # self._editor_instance._reset_elements_in_gui()
        elements = self._editor_instance.elements
        elements.update_all()

    def _cb_debug(self, value):
        pass

    def _cb_verbose(self, value):
        pass

    def _cb_BBOX_expand(self, value):
        self._editor_instance._update_BBOX_and_axes()

    def _cb_color_BBOX_selected(self, value):
        if self._editor_instance.elements.is_current_selected:
            self._editor_instance._update_BBOX_and_axes()

    def _cb_color_BBOX_unselected(self, value):
        if not self._editor_instance.elements.is_current_selected:
            self._editor_instance._update_BBOX_and_axes()

    def _cb_show_BBOX(self, value):
        self._editor_instance._update_BBOX_and_axes()

    def _cb_show_BBOX_axes(self, value):
        self._editor_instance._update_BBOX_and_axes()

    def _cb_BBOX_axes_width(self, value):
        self._editor_instance._update_BBOX_and_axes()

    def _cb_paint_selected(self, value):
        self._editor_instance.elements.update_indices(None)

    def _cb_color_selected(self, value):
        indices = np.where(self._editor_instance.elements.is_selected)[0].tolist()
        self._editor_instance.elements.update_indices(indices)

    def _cb_paint_random(self, value):
        elements = self._editor_instance.elements
        for elem in elements:
            if value:
                # print("printing new random colors")
                elem._color = np.random.random(3)
                elem._brightness = self._dict["random_color_brightness"].value
            else:
                elem._color = elem.color_original
                elem._brightness = self._dict["original_color_brightness"].value

        self._editor_instance.elements.update_indices(None)

    def _cb_random_color_brightness(self, value):
        if self._dict["paint_random"].value:
            elements = self._editor_instance.elements

            for elem in elements:
                elem._brightness = value

        self._editor_instance.elements.update_indices(None)

    def _cb_original_color_brightness(self, value):
        if not self._dict["paint_random"].value:
            elements = self._editor_instance.elements

            for elem in elements:
                elem._brightness = value

        self._editor_instance.elements.update_indices(None)

    def _cb_highlight_ratio(self, value):
        self._editor_instance.elements.update_indices(None)

    def _cb_number_undo_states(self, value):
        self._editor_instance._past_states = self._editor_instance._past_states[-value:]

    def _cb_number_redo_states(self, value):
        self._editor_instance._future_states = self._editor_instance._future_states[
            -value:
        ]

    def _cb_color_unselected(self, value):
        unselected = ~np.array(self._editor_instance.elements.is_selected)
        indices = np.where(unselected)[0].tolist()
        self._editor_instance.elements.update_indices(indices)

    def _cb_number_points_distance(self, value):
        for elem in self._editor_instance.elements:
            elem._get_distance_checker(0)

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

        _panel_collapsable.add_child(
            ParameterPanel(self._dict, 0.25 * em, separation_height).panel
        )

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
