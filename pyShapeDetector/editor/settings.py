import warnings
import copy
import numpy as np
from typing import TYPE_CHECKING
from open3d.visualization import gui
from open3d.visualization.rendering import MaterialRecord

if TYPE_CHECKING:
    from .editor_app import Editor

from .parameter import (
    ParameterBool,
    ParameterNumeric,
    ParameterNDArray,
    ParameterOptions,
    ParameterColor,
    ParameterPanel,
)

# Uncomment this to create extension for testing reference parameters
TEST_PARAMETER_REFERENCES = False

color_bbox_selected_DEFAULT = np.array([0, 204.8, 0.0]) / 255
# color_bbox_unselected_DEFAULT = np.array([255.0, 0.0, 0.0]) / 255
color_bbox_unselected_DEFAULT = np.array([146.0, 3.0, 3.0]) / 255
COLOR_SELECTED_DEFAULT = np.array([178.5, 163.8, 0.0]) / 255
COLOR_UNSELECTED_DEFAULT = np.array([76.5, 76.5, 76.5]) / 255

# DEFAULT VALUES
_extensions_on_panel = False
_empty_extensions_on_panel_window = False
_PointCloud_density = 0.00224
_draw_boundary_lines = True
_line_width = 7
_PointCloud_point_size = 5
_PCD_downsample_when_drawing = True
_PCD_downsample_mode = "Voxel"
_PCD_max_points = 50000
_PCD_voxel_downsample_ratio = 5
_PCD_use_Tensor = True
_mesh_show_back_face = True
_paint_selected = True
_color_selected = gui.Color(*COLOR_SELECTED_DEFAULT)
_paint_random = False
_show_current = True
_debug = True
_verbose = False
_bbox_expand_percentage = 0
_color_bbox_selected = gui.Color(*color_bbox_selected_DEFAULT)
_color_bbox_unselected = gui.Color(*color_bbox_unselected_DEFAULT)
_show_bbox = True
_show_bbox_axes = False
_bbox_axes_width = 0.012
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
        scaling = self._editor_instance._main_window.scaling
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
        self._dict = {
            "extensions_on_window": ParameterBool(
                label="Open extensions on window",
                default=_extensions_on_panel,
                on_update=self._cb_extensions_on_panel,
                subpanel="Extensions",
            ),
            "empty_extensions_on_panel_window": ParameterBool(
                label="Open extensions without parameters on window/panel",
                default=_empty_extensions_on_panel_window,
                on_update=self._cb_empty_extensions_on_panel_window,
                subpanel="Extensions",
            ),
            "PointCloud_density": ParameterNumeric(
                type=float,
                label="Density",
                default=_PointCloud_density,
                on_update=self._cb_PointCloud_density,
                subpanel="PointCloud options",
            ),
            "PCD_downsample_when_drawing": ParameterBool(
                label="Downsample",
                default=_PCD_downsample_when_drawing,
                on_update=self._cb_PCD_downsample_when_drawing,
                subpanel="PointCloud options",
            ),
            "PCD_downsample_mode": ParameterOptions(
                options=["Uniform", "Voxel"],
                label="Downsample mode",
                default=_PCD_downsample_mode,
                on_update=self._cb_PCD_downsample_mode,
                subpanel="PointCloud options",
            ),
            "PCD_max_points": ParameterNumeric(
                type=int,
                label="Downsample Max points",
                default=_PCD_max_points,
                on_update=self._cb_PCD_max_points,
                limits=(50000, 10000000),
                subpanel="PointCloud options",
            ),
            "PCD_downsample_voxel_ratio": ParameterNumeric(
                type=float,
                label="Downsample Voxel Ratio",
                default=_PCD_voxel_downsample_ratio,
                on_update=self._cb_PCD_downsample_voxel_ratio,
                limits=(1, 100),
                subpanel="PointCloud options",
            ),
            "PCD_use_Tensor": ParameterBool(
                label="Use Tensor",
                default=_PCD_use_Tensor,
                on_update=self._cb_PCD_use_Tensor,
                subpanel="PointCloud options",
            ),
            "PointCloud_point_size": ParameterNumeric(
                type=float,
                label="Point Size",
                default=_PointCloud_point_size,
                on_update=self._cb_PointCloud_point_size,
                limits=(3, 15),
            ),
            "draw_boundary_lines": ParameterBool(
                label="Draw Boundary Lines",
                default=_draw_boundary_lines,
                on_update=self._cb_draw_boundary_lines,
            ),
            "line_width": ParameterNumeric(
                type=float,
                label="Line Width",
                default=_line_width,
                on_update=self._cb_line_width,
                limits=(1.5, 15),
            ),
            "mesh_show_back_face": ParameterBool(
                label="Show meshes' back face",
                default=_mesh_show_back_face,
                on_update=self._cb_mesh_show_back_face,
            ),
            "show_current": ParameterBool(
                label="Show current element on info",
                default=_show_current,
                on_update=self._cb_show_current,
                subpanel="Debug",
            ),
            "debug": ParameterBool(
                label="Debug",
                default=_debug,
                on_update=self._cb_debug,
                subpanel="Debug",
            ),
            "verbose": ParameterBool(
                label="Verbose",
                default=_verbose,
                on_update=self._cb_verbose,
                subpanel="Debug",
            ),
            "bbox_expand_percentage": ParameterNumeric(
                type=int,
                label="Expand BBOX %",
                default=_bbox_expand_percentage,
                limits=(-100, 100),
                on_update=self._cb_bbox_expand_percentage,
                subpanel="Bounding Box and axes",
            ),
            "color_bbox_selected": ParameterColor(
                label="Color BBOX selected",
                default=_color_bbox_selected,
                on_update=self._cb_color_bbox_selected,
                subpanel="Bounding Box and axes",
            ),
            "color_bbox_unselected": ParameterColor(
                label="Color BBOX unselected",
                default=_color_bbox_unselected,
                on_update=self._cb_color_bbox_unselected,
                subpanel="Bounding Box and axes",
            ),
            "show_bbox": ParameterBool(
                label="Show BBOX",
                default=_show_bbox,
                on_update=self._cb_show_bbox,
                subpanel="Bounding Box and axes",
            ),
            "show_bbox_axes": ParameterBool(
                label="Show BBOX's axes",
                default=_show_bbox_axes,
                on_update=self._cb_show_bbox_axes,
                subpanel="Bounding Box and axes",
            ),
            "bbox_axes_width": ParameterNumeric(
                type=float,
                label="BBOX's axes width",
                default=_bbox_axes_width,
                on_update=self._cb_bbox_axes_width,
                subpanel="Bounding Box and axes",
                limits=(0.01, 0.5),
            ),
            "paint_selected": ParameterBool(
                label="Paint when Selected",
                default=_paint_selected,
                on_update=self._cb_paint_selected,
                subpanel="Color",
            ),
            "color_selected": ParameterColor(
                label="Selected color",
                default=_color_selected,
                on_update=self._cb_color_selected,
                subpanel="Color",
            ),
            "paint_random": ParameterBool(
                label="Paint with random colors",
                default=_paint_random,
                on_update=self._cb_paint_random,
                subpanel="Color",
            ),
            "random_color_brightness": ParameterNumeric(
                type=float,
                label="Random Color Brightness",
                default=_random_color_brightness,
                on_update=self._cb_random_color_brightness,
                limits=(0.001, 1),
                subpanel="Color",
            ),
            "original_color_brightness": ParameterNumeric(
                type=float,
                label="Original Color Brightness",
                default=_original_color_brightness,
                on_update=self._cb_original_color_brightness,
                limits=(0.001, 1),
                subpanel="Color",
            ),
            "highlight_ratio": ParameterNumeric(
                type=float,
                label="Highlight ratio",
                default=_highlight_ratio,
                on_update=self._cb_highlight_ratio,
                limits=(0.01, 1),
                subpanel="Color",
            ),
            "number_points_distance": ParameterNumeric(
                type=int,
                label="Number of points distances",
                default=_number_points_distance,
                on_update=self._cb_number_points_distance,
                limits=(5, 50),
            ),
            "number_undo_states": ParameterNumeric(
                type=int,
                label="Number of undo states",
                default=_number_undo_states,
                on_update=self._cb_number_undo_states,
                limits=(1, 10),
            ),
            "number_redo_states": ParameterNumeric(
                type=int,
                label="Number of redo states",
                default=_number_redo_states,
                on_update=self._cb_number_redo_states,
                limits=(1, 10),
            ),
        }

        if TEST_PARAMETER_REFERENCES:
            for key, param in PARAMETERS_TEST_DICTIONARY.items():
                self._dict[key] = param

        # self._dict = {param.name: param for param in options}
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
        if TEST_PARAMETER_REFERENCES:
            from .extension import Extension

            extension = Extension(test_extension, self, editor_instance)
            extension.add_to_application()

    def _cb_extensions_on_panel(self, value):
        if value is False:
            for window in self._editor_instance._temp_windows:
                window.close()
        else:
            for name in self._editor_instance._extensions_panels:
                self._editor_instance._set_extension_panel_open(name, False)

    def _cb_empty_extensions_on_panel_window(self, value):
        pass

    def _cb_PointCloud_density(self, value):
        pass

    def _cb_PCD_downsample_when_drawing(self, value):
        from .element import ElementPointCloud

        for elem in self._editor_instance.element_container:
            if not isinstance(elem, ElementPointCloud):
                continue

            elem._get_drawable()
            elem._init_colors()
            elem.update(is_current=elem.current, update_scene=True, reset=True)

    def _cb_PCD_downsample_mode(self, value):
        if self.get_setting("PCD_downsample_when_drawing"):
            self._cb_PCD_downsample_when_drawing(True)

    def _cb_PCD_max_points(self, value):
        if self.get_setting("PCD_downsample_when_drawing"):
            self._cb_PCD_downsample_when_drawing(True)

    def _cb_PCD_downsample_voxel_ratio(self, value):
        if self.get_setting("PCD_downsample_when_drawing"):
            self._cb_PCD_downsample_when_drawing(True)

    def _cb_PCD_use_Tensor(self, value):
        if self.get_setting("PCD_downsample_when_drawing"):
            self._cb_PCD_downsample_when_drawing(True)

    def _cb_PointCloud_point_size(self, value):
        from .element import ElementPointCloud

        elements = self._editor_instance.element_container

        self._update_materials()
        indices = np.where(
            [isinstance(element, ElementPointCloud) for element in elements]
        )[0].tolist()

        for idx in indices:
            elements[idx].update_on_scene(reset=True)

    def _cb_draw_boundary_lines(self, value):
        self._editor_instance._update_plane_boundaries()

    def _cb_line_width(self, value):
        # line_width = self._dict["line_width"].value

        self._update_materials()
        self._editor_instance._update_extra_elements(planes_boundaries=True)

    def _cb_mesh_show_back_face(self, value):
        # self._editor_instance._reset_elements_in_gui()
        elements = self._editor_instance.element_container
        elements.update_all()

    def _cb_show_current(self, value):
        self._editor_instance._update_info()

    def _cb_debug(self, value):
        pass

    def _cb_verbose(self, value):
        pass

    def _cb_bbox_expand_percentage(self, value):
        self._editor_instance._update_BBOX_and_axes()

    def _cb_color_bbox_selected(self, value):
        if self._editor_instance.element_container.is_current_selected:
            self._editor_instance._update_BBOX_and_axes()

    def _cb_color_bbox_unselected(self, value):
        if not self._editor_instance.element_container.is_current_selected:
            self._editor_instance._update_BBOX_and_axes()

    def _cb_show_bbox(self, value):
        self._editor_instance._update_BBOX_and_axes()

    def _cb_show_bbox_axes(self, value):
        self._editor_instance._update_BBOX_and_axes()

    def _cb_bbox_axes_width(self, value):
        self._editor_instance._update_BBOX_and_axes()

    def _cb_paint_selected(self, value):
        self._editor_instance.element_container.update_indices(None)

    def _cb_color_selected(self, value):
        indices = np.where(self._editor_instance.element_container.is_selected)[
            0
        ].tolist()
        self._editor_instance.element_container.update_indices(indices)

    def _cb_paint_random(self, value):
        elements = self._editor_instance.element_container
        for elem in elements:
            if value:
                # print("printing new random colors")
                elem._color = np.random.random(3)
                elem._brightness = self._dict["random_color_brightness"].value
            else:
                elem._color = elem.color_original
                elem._brightness = self._dict["original_color_brightness"].value

        self._editor_instance.element_container.update_indices(None)

    def _cb_random_color_brightness(self, value):
        if self._dict["paint_random"].value:
            elements = self._editor_instance.element_container

            for elem in elements:
                elem._brightness = value

        self._editor_instance.element_container.update_indices(None)

    def _cb_original_color_brightness(self, value):
        if not self._dict["paint_random"].value:
            elements = self._editor_instance.element_container

            for elem in elements:
                elem._brightness = value

        self._editor_instance.element_container.update_indices(None)

    def _cb_highlight_ratio(self, value):
        self._editor_instance.element_container.update_indices(None)

    def _cb_number_undo_states(self, value):
        self._editor_instance._past_states = self._editor_instance._past_states[-value:]

    def _cb_number_redo_states(self, value):
        self._editor_instance._future_states = self._editor_instance._future_states[
            -value:
        ]

    def _cb_color_unselected(self, value):
        unselected = ~np.array(self._editor_instance.element_container.is_selected)
        indices = np.where(unselected)[0].tolist()
        self._editor_instance.element_container.update_indices(indices)

    def _cb_number_points_distance(self, value):
        for elem in self._editor_instance.element_container:
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
        window = self._editor_instance._main_window
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
        window = self._editor_instance._main_window
        menubar = self._editor_instance.app.menubar
        self._panel.visible = not self._panel.visible
        menubar.set_checked(self._preferences_binding._menu_id, self._panel.visible)
        window.set_needs_layout()


###########################################
# Test Parameters, uncomment when testing #
###########################################
PARAMETERS_TEST_DICTIONARY = {
    "test_bool": ParameterBool(
        label="test_bool",
        default=True,
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_int": ParameterNumeric(
        type=int,
        label="test_int",
        default=3,
        limits=(-10, 10),
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_int_no_slider": ParameterNumeric(
        type=int,
        label="test_int_no_slider",
        default=3,
        limits=(-10, 10),
        use_slider=False,
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_int_no_limits": ParameterNumeric(
        type=int,
        label="test_int_no_limits",
        default=2,
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_float": ParameterNumeric(
        type=float,
        label="test_float",
        default=3,
        limits=(-10, 10),
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_float_no_slider": ParameterNumeric(
        type=float,
        label="test_float_no_slider",
        default=3,
        limits=(-10, 10),
        use_slider=False,
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_float_no_limits": ParameterNumeric(
        type=float,
        label="test_float_no_limits",
        default=2,
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_color": ParameterColor(
        label="test_color",
        default=(0.2, 0.2, 0.2),
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_options": ParameterOptions(
        label="test_options",
        options=["a", "b", "c"],
        default="c",
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_int_shape_2_3": ParameterNDArray(
        label="test_int_shape_2_3",
        default=[[0, 0, 1], [0, 20, 0]],
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_float_shape_2_3": ParameterNDArray(
        label="test_float_shape_2_3",
        default=[[0, 0, 1], [0.0, 20, 0]],
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_array_int_ndim1": ParameterNDArray(
        label="test_array_int_ndim1",
        default=[0, 0, 1],
        on_update=lambda x: print(x),
        subpanel="test",
    ),
    "test_array_shape_7_3": ParameterNDArray(
        label="test_array_shape_7_3",
        default=np.zeros((7, 3)),
        on_update=lambda x: print(x),
        subpanel="test",
    ),
}


def test_function(
    test_bool,
    test_int,
    test_int_no_slider,
    test_int_no_limits,
    test_float,
    test_float_no_slider,
    test_float_no_limits,
    test_color,
    test_options,
    test_int_shape_2_3,
    test_float_shape_2_3,
    test_array_int_ndim1,
    test_array_shape_7_3,
):
    print("************")
    print(f"test_bool: {test_bool}")
    print(f"test_int: {test_int}")
    print(f"test_int_no_slider: {test_int_no_slider}")
    print(f"test_int_no_limits: {test_int_no_limits}")
    print(f"test_float: {test_float}")
    print(f"test_float_no_slider: {test_float_no_slider}")
    print(f"test_float_no_limits: {test_float_no_limits}")
    print(f"test_color: {test_color}")
    print(f"test_options: {test_options}")
    print(f"test_int_shape_2_3: {test_int_shape_2_3}")
    print(f"test_float_shape_2_3 {test_float_shape_2_3}")
    print(f"test_array_int_ndim1: {test_array_int_ndim1}")
    print(f"test_array_shape_7_3: {test_array_shape_7_3}")
    print("************")
    return []


test_extension = {
    "function": test_function,
    "name": "Testing reference widgets",
    "menu": "Test",
    "inputs": None,
    "parameters": {key: {"type": "preference"} for key in PARAMETERS_TEST_DICTIONARY},
}
