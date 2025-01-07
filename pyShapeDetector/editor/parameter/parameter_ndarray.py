#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 10:03:26

@author: evbernardes
"""


from typing import Callable, Union
import numpy as np
from open3d.visualization import gui
from .parameter import ParameterBase


class ParameterNDArray(ParameterBase):
    """Parameter for NDArray types.

    Creates an array of multiple instances of gui.TextEdit.

    Detects shape and dtype by the given default value.

    Only possible for arrays with 1 or 2 dimensions.

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

    shape
    dtype
    ndim

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
                f"Parameter {self.label} of type {self.type_name} requires default value."
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
                f"{new_value.shape} for parameter {self.label}"
            )
        self._ndim = new_value.ndim
        self._shape = new_value.shape
        self._value = np.atleast_2d(new_value)

        # if self.is_reference:
        self._update_internal_element()

    def _update_internal_element(self):
        if self.internal_element is None:
            return
        for i in range(self._value.shape[0]):
            for j in range(self._value.shape[1]):
                text_edit = self.internal_element[i][j]
                text_edit.text_value = str(self._value[i, j])

    def _enable_internal_element(self, value: bool) -> None:
        """Enables/Disables internal element when creating references"""
        if self.internal_element is None:
            return
        for elem in np.array(self.internal_element).flatten():
            elem.enabled = value

    def _callback(self, line, col, value, text_edit):
        try:
            self._value[line, col] = self.dtype(value)
        except Exception:
            text_edit.text_value = str(self._value[line, col])
        self.on_update(self.value)
        self._update_references()

    def get_gui_widget(self, font_size):
        shape_internal = self._value.shape
        self._internal_element = [
            [gui.TextEdit() for col in range(shape_internal[1])]
            for line in range(shape_internal[0])
        ]
        self._update_internal_element()

        label = gui.Label(self.label)

        text_edits = self.internal_element

        elements_array = gui.VGrid(self._value.shape[1], 0.5 * font_size)
        for i in range(self._value.shape[0]):
            # elements_line = gui.Horiz(0.25 * font_size)
            for j in range(self._value.shape[1]):
                text_edit = text_edits[i][j]
                text_edit.set_on_value_changed(
                    lambda value, line=i, col=j, t=text_edit: self._callback(
                        line, col, value, t
                    )
                )
                elements_array.add_child(text_edit)

        element = gui.VGrid(1, 0.5 * font_size)
        element.add_child(label)
        element.add_child(elements_array)
        self._enable_internal_element(not self.is_reference)

        return element

    def __init__(
        self,
        label: str,
        default: Union[list, tuple, np.ndarray] = None,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(label=label, on_update=on_update, subpanel=subpanel)

        self.value = default
        self._warn_unused_parameters(other_kwargs)
