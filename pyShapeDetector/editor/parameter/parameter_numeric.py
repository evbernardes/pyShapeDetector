import warnings
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Union
from open3d.visualization import gui
from .parameter import ParameterBase
from pyShapeDetector.editor.element import ElementContainer


class ParameterNumeric(ParameterBase[type]):
    """Parameter for Int for Float types.

    Creates a Slider if has limits, otherwise a number edit edit.

    Attributes
    ----------
    is_reference
    valid_arguments
    type
    value
    type_name
    label
    subpanel
    on_update

    use_slider
    limits
    limit_setter

    Methods
    -------
    _warn_unused_parameters
    _callback
    _update_internal_element
    _reset_values_and_limits
    _update_references
    _enable_internal_element
    get_gui_widget
    create_reference
    create_from_dict
    """

    # _type = int
    _valid_arguments = ParameterBase._valid_arguments + [
        "limits",
        "limit_setter",
        "use_slider",
        "type",
    ]

    def get_slider_type(self):
        if self.type == int:
            return gui.Slider.INT
        elif self.type == float:
            return gui.Slider.DOUBLE
        else:
            assert False

    def get_number_edit_type(self):
        if self.type == int:
            return gui.NumberEdit.INT
        elif self.type == float:
            return gui.NumberEdit.DOUBLE
        else:
            assert False

    def update_widget(self, value):
        if self.type is int:
            self.internal_element.int_value = value
        elif self.type is float:
            self.internal_element.double_value = value
        else:
            assert False

    @property
    def use_slider(self):
        return self._use_slider

    @use_slider.setter
    def use_slider(self, value):
        if not isinstance(value, bool):
            raise TypeError(f"Expected boolean for 'use_slider', got {value}.")
        self._use_slider = value

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

    @ParameterBase.value.setter
    def value(self, new_value):
        if new_value is not None:
            try:
                new_value = self.type(new_value)
            except ValueError:
                new_value = self.type(float(new_value))

        else:
            if self.limits is None:
                raise TypeError(
                    f"No limits or default value for parameter {self.label} of type {self.type_name}."
                )
            else:
                new_value = self.limits[0]

        if self.limits is not None:
            if (self.limits[0] - new_value > 1e-50) or (
                new_value - self.limits[1] > 1e-5
            ):
                warnings.warn(
                    f"Default value {new_value} not in limits {self.limits} "
                    f"for parameter {self.label} of type {self.type_name}, "
                    "resetting it."
                )
                new_value = self._value

        self._value = new_value
        # if self.is_reference:
        self._update_internal_element()

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

    def _update_internal_element(self):
        if self.internal_element is None:
            return

        self.update_widget(self._value)
        if self.limits is not None:
            self.internal_element.set_limits(*self.limits)

    def _callback(self, value):
        self.value = value
        self.on_update(self.value)
        self._update_references()

    def _reset_values_and_limits(self, elements: ElementContainer):
        if self.limit_setter is None:
            return

        try:
            selected_raw_elements = [elem.raw for elem in elements if elem.is_selected]
            self.limits = self.limit_setter(selected_raw_elements)
        except Exception:
            warnings.warn(f"Could not reset limits of parameter {self.label}:")
            traceback.print_exc()
        finally:
            # Recheck changed limits
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            self.value = self.value

    def get_gui_widget(self, font_size):
        if self.use_slider:
            self._internal_element = gui.Slider(self.get_slider_type())
        else:
            self._internal_element = gui.NumberEdit(self.get_number_edit_type())
        self._update_internal_element()

        label = gui.Label(self.label)

        element = gui.VGrid(1, 0.25 * font_size)
        element.add_child(label)
        element.add_child(self.internal_element)

        self.internal_element.set_on_value_changed(self._callback)
        self._enable_internal_element(not self.is_reference)
        return element

    def __init__(
        self,
        label: str,
        limit_setter: Callable = None,
        limits: Union[list, tuple] = None,
        default: int = None,
        use_slider: Union[bool, None] = None,
        type: Union[type, None] = None,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(label=label, on_update=on_update, subpanel=subpanel)

        if limit_setter is not None and limits is None:
            warnings.warn(
                "When setting a limit setter, default limits should be given. "
                f"Ignoring limit_setter for parameter '{self.label}'."
            )
            limit_setter = None

        if type is None:
            if isinstance(default, int):
                type = int
            elif isinstance(default, float):
                type = float
            else:
                raise ValueError(
                    "Expected int or float values for Numeric parameter, got {value}."
                )
        self._type = type

        self.limit_setter = limit_setter
        self.limits = limits
        self.value = default

        if self.label == "number_undo_states":
            print()

        if use_slider is None:
            self.use_slider = self.limits is not None
        elif use_slider and self.limits is None:
            warnings.warn(
                f"Cannot use Slider for parameter {self.label}, no limits given."
            )
            self.use_slider = False
        else:
            self.use_slider = use_slider

        self._warn_unused_parameters(other_kwargs)
