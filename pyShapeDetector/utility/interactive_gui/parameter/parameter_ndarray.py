#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 10:03:26

@author: evbernardes
"""


from typing import Callable, Union
import numpy as np
from open3d.visualization import gui
from .parameter import Parameter


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
