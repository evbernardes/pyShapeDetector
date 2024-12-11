import traceback
import warnings
from typing import Callable, Union
import numpy as np
from open3d.visualization import gui
from .editor_app import Editor


class Parameter:
    _type = None.__class__

    @property
    def internal_element(self):
        return self._internal_element

    @property
    def type_name(self):
        return self._type.__name__

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise TypeError(
                f"Expected string as name for Parameter, got {type(new_name)}."
            )
        self._name = new_name

    @property
    def subpanel(self):
        return self._subpanel

    @subpanel.setter
    def subpanel(self, new_subpanel):
        if new_subpanel is not None and not isinstance(new_subpanel, str):
            raise TypeError(
                f"Expected string as subpanel for Parameter, got {type(new_subpanel)}."
            )
        self._subpanel = new_subpanel

    @property
    def pretty_name(self):
        words = self.name.replace("_", " ").split()
        result = []
        for word in words:
            if word.isupper():  # Keep existing UPPERCASE values as is
                result.append(word)
            else:  # Capitalize other words
                result.append(word.capitalize())

        return " ".join(result)

    @property
    def on_update(self):
        if self._on_update is None:
            return lambda value: None
        return self._on_update

    @on_update.setter
    def on_update(self, func: Callable):
        if func is None:
            self._on_update = None
        elif callable(func):
            self._on_update = func
        else:
            raise TypeError(
                f"Parameter of type '{self.type_name}' received invalid 'on_update'"
            )

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, new_type):
        if not isinstance(new_type, type):
            raise TypeError("parameter descriptor has invalid type.")

        self._type = new_type

    @property
    def value(self):
        return self._value

    def _warn_unused_parameters(self, other_kwargs: dict):
        for key in other_kwargs:
            warnings.warn(
                f"Ignoring unexpected '{key}' descriptor in parameter "
                f"'{self.name}' of type '{self.type_name}'."
            )

    def _callback(self, value, text_edit=None):
        old_value = self.value
        try:
            self._value = self.type(value)
            if abs(self.value - old_value) > 1e-6:
                self.on_update(self.value)
        except Exception:
            if text_edit is not None:
                text_edit.text_value = str(self.value)
        self._update_references()

    def _reset_values_and_limits(self, editor_instance: Editor):
        pass

    def _update_references(self):
        for reference in self._references:
            reference.value = self.value

    def get_gui_element(self, font_size):
        label = gui.Label(self.pretty_name)

        # Text field for general inputs
        text_edit = gui.TextEdit()
        # text_edit.placeholder_text = str(self.value)
        text_edit.set_on_value_changed(lambda value: self._callback(value, text_edit))

        element = gui.VGrid(2, 0.25 * font_size)
        element.add_child(label)
        element.add_child(text_edit)

        return element

    @staticmethod
    def create_from_dict(key: str, parameter_descriptor: dict):
        parameter_descriptor = parameter_descriptor.copy()
        if "name" not in parameter_descriptor:
            parameter_descriptor["name"] = key
        _type = parameter_descriptor.pop("type", None)
        if _type not in PARAMETER_TYPE_DICTIONARY:
            raise ValueError(f"{_type} does not correspond to valid Parameter type.")

        parameter = PARAMETER_TYPE_DICTIONARY[_type](**parameter_descriptor)

        return parameter

    def __init__(
        self,
        name: str,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
    ):
        self._references = []
        self.name = name
        self.on_update = on_update
        self.subpanel = subpanel

    def create_reference(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Ignoring unexpected.*")
            new_parameter = Parameter.create_from_dict(
                self.name,
                {
                    "type": self.type,
                    "default": self.value,
                    "limits": getattr(self, "limits", None),
                    "limit_setter": getattr(self, "limit_setter", None),
                    "options": getattr(self, "options", None),
                },
            )

        if isinstance(self.internal_element, list):
            for elem in np.array(new_parameter.internal_element).flatten():
                elem.enabled = False
        else:
            new_parameter.internal_element.enabled = False

        self._references.append(new_parameter)
        return new_parameter


class ParameterBool(Parameter):
    _type = bool

    @Parameter.value.setter
    def value(self, new_value):
        self._value = bool(new_value)
        self.internal_element.checked = self.value

    def get_gui_element(self, font_size):
        checkbox = self.internal_element
        checkbox.set_on_checked(self._callback)
        return checkbox

    def __init__(
        self,
        name: str,
        default: bool = False,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(name=name, on_update=on_update, subpanel=subpanel)
        self._internal_element = gui.Checkbox(self.pretty_name + "?")
        self.value = default
        self._warn_unused_parameters(other_kwargs)


class ParameterOptions(Parameter):
    _type = list

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, new_options):
        if not isinstance(new_options, (tuple, list)) or len(new_options) == 0:
            raise ValueError(
                f"Parameter {self.name} requires non-empty of values for "
                f"options, got {new_options}."
            )

        self._options = list(new_options)

    @Parameter.value.setter
    def value(self, new_value):
        if new_value not in self.options:
            raise ValueError(
                f"Value '{new_value}' for parameter '{self.name}' is not in "
                f"options list {self.options}."
            )

        self._value = new_value

        options_strings = [str(option) for option in self.options]
        self.internal_element.selected_index = options_strings.index(str(self.value))

    def _callback(self, text, index):
        self._value = self.options[index]
        self.on_update(self.value)
        self._update_references()

    def get_gui_element(self, font_size):
        label = gui.Label(self.pretty_name)

        combobox = self.internal_element
        combobox.set_on_selection_changed(self._callback)

        element = gui.VGrid(2, 0.25 * font_size)
        element.add_child(label)
        element.add_child(combobox)

        return element

    def __init__(
        self,
        name: str,
        options: list,
        default=None,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(name=name, on_update=on_update, subpanel=subpanel)
        self.options = options

        if default is None:
            default = options[0]

        self._internal_element = gui.Combobox()
        for option in self.options:
            self._internal_element.add_item(str(option))

        self.value = default

        self._warn_unused_parameters(other_kwargs)


class ParameterInt(Parameter):
    _type = int

    @property
    def limit_setter(self):
        return self._limit_setter

    @limit_setter.setter
    def limit_setter(self, new_setter):
        if new_setter is None:
            self._limit_setter = None
            return

        elif callable(new_setter):
            self._limit_setter = new_setter

    @Parameter.value.setter
    def value(self, new_value):
        if new_value is not None:
            try:
                new_value = int(new_value)
            except ValueError:
                new_value = int(float(new_value))

        else:
            if self.limits is None:
                raise TypeError("No limits or default value for parameter of type int.")
            else:
                new_value = self.limits[0]

        if self.limits is not None and (
            not (self.limits[0] <= new_value <= self.limits[1])
        ):
            warnings.warn(
                f"Default value not in limits for parameter {self.name}, resetting it."
            )
            new_value = self.limits[0]

        self._value = new_value

        if isinstance(self.internal_element, gui.Slider):
            self.internal_element.int_value = self.value
        else:
            self.internal_element.text_value = str(self.value)

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, new_limits):
        if new_limits is None:
            self._limits = None
            return

        if not isinstance(new_limits, (list, tuple)) or len(new_limits) != 2:
            raise TypeError(
                f"Limits should be list or tuple of 2 elements, got {new_limits}."
            )

        new_limits = (min(new_limits), max(new_limits))
        self._limits = new_limits
        self.internal_element.set_limits(*self.limits)

    def _reset_values_and_limits(self, editor_instance: Editor):
        if self.limit_setter is None:
            return

        try:
            self.limits = self.limit_setter(
                editor_instance.elements.selected_raw_elements
            )
        except Exception:
            warnings.warn(f"Could not reset limits of parameter {self.name}:")
            traceback.print_exc()
        finally:
            # Recheck changed limits
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            self.value = self.value

    def get_gui_element(self, font_size):
        label = gui.Label(self.pretty_name)

        if isinstance(self.internal_element, gui.Slider):
            slider = self.internal_element
            slider.set_on_value_changed(self._callback)

            element = gui.VGrid(2, 0.25 * font_size)
            element.add_child(label)
            element.add_child(slider)

        elif isinstance(self.internal_element, gui.TextEdit):
            text_edit = self.internal_element
            text_edit.set_on_value_changed(
                lambda value: self._callback(value, text_edit)
            )

            element = gui.VGrid(2, 0.25 * font_size)
            element.add_child(label)
            element.add_child(text_edit)
        else:
            raise RuntimeError(
                "ParameterInt internal element is neither a gui.Glider "
                "nor a gui.TextEdit. This should never happen."
            )

        return element

    def __init__(
        self,
        name: str,
        limit_setter: Callable = None,
        limits: Union[list, tuple] = None,
        default: int = None,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(name=name, on_update=on_update, subpanel=subpanel)
        self.limit_setter = limit_setter

        if limits is not None and self.limit_setter is None:
            self._internal_element = gui.Slider(gui.Slider.INT)
        else:
            # Text field for general inputs
            self._internal_element = gui.TextEdit()

        if self.limit_setter is None:
            self.limits = limits
            self.value = default

        else:
            if limits is not None:
                warnings.warn(
                    f"Limit setter defined for parameter {self.name}, "
                    "ignoring input limits."
                )

            self._limits = None

            if default is not None:
                warnings.warn(
                    f"Limit setter defined for parameter {self.name}, "
                    "ignoring input default value."
                )
            self._value = None

        self._warn_unused_parameters(other_kwargs)


class ParameterFloat(Parameter):
    _type = float

    @property
    def limit_setter(self):
        return self._limit_setter

    @limit_setter.setter
    def limit_setter(self, new_setter):
        if new_setter is None:
            self._limit_setter = None
            return

        elif callable(new_setter):
            self._limit_setter = new_setter

    @Parameter.value.setter
    def value(self, new_value):
        if new_value is not None:
            new_value = float(new_value)

        else:
            if self.limits is None:
                raise TypeError(
                    "No limits or default value for parameter of type float."
                )
            else:
                new_value = self.limits[0]

        if self.limits is not None and (
            not (self.limits[0] <= new_value <= self.limits[1])
        ):
            warnings.warn(
                f"Default value not in limits for parameter {self.name}, resetting it."
            )
            new_value = self.limits[0]

        self._value = new_value

        if isinstance(self.internal_element, gui.Slider):
            self.internal_element.double_value = self.value
        else:
            self.internal_element.text_value = str(self.value)

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, new_limits):
        if new_limits is None:
            self._limits = None
            return

        if not isinstance(new_limits, (list, tuple)) or len(new_limits) != 2:
            raise TypeError(
                f"Limits should be list or tuple of 2 elements, got {new_limits}."
            )

        new_limits = (min(new_limits), max(new_limits))
        self._limits = new_limits
        self.internal_element.set_limits(*self.limits)

    def _reset_values_and_limits(self, editor_instance: Editor):
        if self.limit_setter is None:
            return

        try:
            self.limits = self.limit_setter(
                editor_instance.elements.selected_raw_elements
            )
        except Exception:
            warnings.warn(f"Could not reset limits of parameter {self.name}:")
            traceback.print_exc()
        finally:
            # Recheck changed limits
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            self.value = self.value

    def get_gui_element(self, font_size):
        label = gui.Label(self.pretty_name)

        if isinstance(self.internal_element, gui.Slider):
            slider = self.internal_element
            slider.set_on_value_changed(self._callback)

            element = gui.VGrid(2, 0.25 * font_size)
            element.add_child(label)
            element.add_child(slider)

        elif isinstance(self.internal_element, gui.TextEdit):
            # Text field for general inputs
            text_edit = self.internal_element
            # text_edit.placeholder_text = str(self.value)
            text_edit.set_on_value_changed(
                lambda value: self._callback(value, text_edit)
            )

            element = gui.VGrid(2, 0.25 * font_size)
            element.add_child(label)
            element.add_child(text_edit)

        else:
            raise RuntimeError(
                "ParameterInt internal element is neither a gui.Glider "
                "nor a gui.TextEdit. This should never happen."
            )

        return element

    def __init__(
        self,
        name: str,
        limit_setter: Callable = None,
        limits: Union[list, tuple] = None,
        default: int = None,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(name=name, on_update=on_update, subpanel=subpanel)
        self.limit_setter = limit_setter

        if limits is not None and limit_setter is None:
            self._internal_element = gui.Slider(gui.Slider.DOUBLE)
        else:
            # Text field for general inputs
            self._internal_element = gui.TextEdit()

        if self.limit_setter is None:
            self.limits = limits
            self.value = default

        else:
            if limits is not None:
                warnings.warn(
                    f"Limit setter defined for parameter {self.name}, "
                    "ignoring input limits."
                )

            self._limits = None

            if default is not None:
                warnings.warn(
                    f"Limit setter defined for parameter {self.name}, "
                    "ignoring input default value."
                )
            self._value = None

        self._warn_unused_parameters(other_kwargs)


class ParameterColor(Parameter):
    _type = gui.Color

    @property
    def red(self):
        return self._value.red

    @property
    def green(self):
        return self._value.green

    @property
    def blue(self):
        return self._value.blue

    @property
    def value(self):
        return np.array((self.red, self.green, self.blue))

    @value.setter
    def value(self, values):
        if isinstance(values, gui.Color):
            self._value = values
        elif isinstance(values, (tuple, list, np.ndarray)) and len(values) == 3:
            self._value = gui.Color(*np.clip(values, 0, 1))
        elif values is None:
            self._value = gui.Color((0, 0, 0, 1))
        else:
            raise TypeError(
                f"Value of parameter {self.name} of type {self.type_name} should "
                f"be a gui.Color, a list or tuple of 3 values, got {values}."
            )
        self.internal_element.color_value = self._value

    def _callback(self, value):
        old_value = self.value
        self._value = value
        if np.any(abs(self.value - old_value) > 1e-6):
            self.on_update(self.value)
        self._update_references()

    def get_gui_element(self, font_size):
        label = gui.Label(self.pretty_name)

        color_selector = self.internal_element
        color_selector.set_on_value_changed(self._callback)

        element = gui.VGrid(2, 0.25 * font_size)
        element.add_child(label)
        element.add_child(color_selector)

        return element

    def __init__(
        self,
        name: str,
        default: Union[list, tuple, np.ndarray] = None,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(name=name, on_update=on_update, subpanel=subpanel)
        self._internal_element = gui.ColorEdit()
        self.value = default

        self._warn_unused_parameters(other_kwargs)


class ParameterNDArray(Parameter):
    _type = np.ndarray

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return self._ndim

    @property
    def value(self):
        if self.ndim == 1:
            return self._value[0]
        else:
            return self._value

    @value.setter
    def value(self, new_value):
        if new_value is None:
            raise TypeError(
                f"Parameter {self.name} of type {self.type_name} requires default value."
            )

        new_value = np.asarray(new_value)
        if new_value.dtype.type in (float, np.float_):
            self._dtype = float
        elif new_value.dtype.type in (int, np.int_):
            self._dtype = int
        else:
            raise TypeError("Supported values for dtype are 'int' and 'float'.")

        if new_value.ndim > 2:
            raise ValueError(
                "Only shapes up to 2 dimentions are accepted, got "
                f"{new_value.shape} for parameter {self.name}"
            )
        self._ndim = new_value.ndim
        self._shape = new_value.shape
        self._value = np.atleast_2d(new_value)

        if not hasattr(self, "_internal_element"):
            return

        for i in range(self._value.shape[0]):
            for j in range(self._value.shape[1]):
                text_edit = self.internal_element[i][j]
                text_edit.text_value = str(self._value[i, j])

    def _callback(self, line, col, value, text_edit):
        try:
            self._value[line, col] = self.dtype(value)
        except Exception:
            text_edit.text_value = str(self._value[line, col])
        self.on_update(self.value)
        self._update_references()

    def get_gui_element(self, font_size):
        label = gui.Label(self.pretty_name)

        text_edits = self.internal_element

        elements_array = gui.VGrid(self._value.shape[0], 0.25 * font_size)
        for i in range(self._value.shape[0]):
            elements_line = gui.Horiz(0.25 * font_size)
            for j in range(self._value.shape[1]):
                text_edit = text_edits[i][j]
                text_edit.set_on_value_changed(
                    lambda value, line=i, col=j, t=text_edit: self._callback(
                        line, col, value, t
                    )
                )
                elements_line.add_child(text_edit)

            elements_array.add_child(elements_line)

        element = gui.VGrid(2, 0.25 * font_size)
        element.add_child(label)
        element.add_child(elements_array)

        return element

    def __init__(
        self,
        name: str,
        default: Union[list, tuple, np.ndarray] = None,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(name=name, on_update=on_update, subpanel=subpanel)

        self.value = default
        shape = self._value.shape
        self._internal_element = [
            [gui.TextEdit() for col in range(shape[1])] for line in range(shape[0])
        ]
        # Setting value on internal element
        self.value = self.value

        self._warn_unused_parameters(other_kwargs)


# class ParameterReference(Parameter):
#     @property
#     def referenced_parameter(self):
#         return self._referenced_parameter

#     @property
#     def name(self):
#         return self._referenced_parameter.name

#     @property
#     def type(self):
#         return self._referenced_parameter.type

#     @property
#     def value(self):
#         return self._referenced_parameter.value

#     def __init__(self, referenced_parameter: Parameter):
#         self._referenced_parameter = referenced_parameter


PARAMETER_TYPE_DICTIONARY = {
    None: Parameter,
    bool: ParameterBool,
    int: ParameterInt,
    float: ParameterFloat,
    list: ParameterOptions,
    np.ndarray: ParameterNDArray,
    gui.Color: ParameterColor,
}


class ParameterPanel:
    def _get_panel(
        self,
        separation_width: float,
        separation_height: float,
        title: Union[str, None],
    ):
        panel = gui.Vert(
            separation_width,
            gui.Margins(
                separation_width, separation_width, separation_width, separation_width
            ),
        )

        if title is not None:
            h = gui.Horiz()
            h.add_stretch()
            h.add_child(gui.Label(title))
            h.add_stretch()
            panel.add_child(h)

        subpanels = {}

        for param in self.parameters.values():
            element = param.get_gui_element(separation_width)
            if param.subpanel is None:
                panel.add_child(element)
                panel.add_fixed(separation_height)
                continue

            if param.subpanel not in subpanels:
                subpanel = gui.CollapsableVert(
                    param.subpanel,
                    0.25 * separation_width,
                    gui.Margins(separation_width, 0, 0, 0),
                )
                subpanels[param.subpanel] = subpanel
                panel.add_child(subpanel)
                subpanel.set_is_open(False)
                panel.add_fixed(separation_height)

            subpanels[param.subpanel].add_child(element)
        return panel

    @property
    def panel(self):
        return self._panel

    @property
    def parameters(self):
        return self._parameters

    def __init__(
        self,
        parameters: dict[Parameter],
        separation_width: float,
        separation_height: float,
        title: Union[str, None] = None,
    ):
        self._parameters = parameters
        self._panel = self._get_panel(separation_width, separation_height, title)
