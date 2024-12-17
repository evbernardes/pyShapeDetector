import warnings
import traceback
from typing import Callable, Union
from open3d.visualization import gui
from pyShapeDetector.utility.editor.editor_app import Editor
from .parameter import ParameterBase


class ParameterFloat(ParameterBase[float]):
    """Parameter for Float types.

    Creates a Slider if has limits, otherwise a normal text edit.

    Attributes
    ----------
    is_reference
    valid_arguments
    type
    value
    type_name
    name
    pretty_name
    subpanel
    on_update

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
    get_gui_element
    create_reference
    create_from_dict
    """

    _type = float
    _valid_arguments = ParameterBase._valid_arguments + ["limits", "limit_setter"]

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
            new_value = float(new_value)

        else:
            if self.limits is None:
                raise TypeError(
                    f"No limits or default value for parameter {self.name} of type {self.type_name}."
                )
            else:
                new_value = self.limits[0]

        if self.limits is not None and (
            not (self.limits[0] <= new_value <= self.limits[1])
        ):
            warnings.warn(
                f"Default value not in limits for parameter {self.name} of type {self.type_name}, resetting it."
            )
            new_value = self.limits[0]

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
        # if self.is_reference:
        # self._update_internal_element()

    def _update_internal_element(self):
        if self.internal_element is None:
            return
        elif isinstance(self.internal_element, gui.Slider):
            self.internal_element.double_value = self.value
            self.internal_element.set_limits(*self.limits)
        else:
            self.internal_element.text_value = str(self.value)

    def _callback(self, value):
        old_value = self.value
        self.value = value
        if abs(self.value - old_value) > 1e-6:
            self.on_update(self.value)
        # if self.is_reference:
        # self._update_internal_element()
        self._update_references()

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
            # slider.set_on_value_changed(self._callback)

            element = gui.VGrid(2, 0.25 * font_size)
            element.add_child(label)
            element.add_child(slider)

        elif isinstance(self.internal_element, gui.TextEdit):
            # Text field for general inputs
            text_edit = self.internal_element
            # text_edit.placeholder_text = str(self.value)
            # text_edit.set_on_value_changed(
            #     lambda value: self._callback(value, text_edit)
            # )

            element = gui.VGrid(2, 0.25 * font_size)
            element.add_child(label)
            element.add_child(text_edit)

        else:
            raise RuntimeError(
                "ParameterInt internal element is neither a gui.Glider "
                "nor a gui.TextEdit. This should never happen."
            )

        self.internal_element.set_on_value_changed(self._callback)

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
