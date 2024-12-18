#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 09:59:28

@author: evbernardes
"""
from typing import Callable, Union
import numpy as np
from open3d.visualization import gui
from .parameter import ParameterBase


class ParameterColor(ParameterBase):
    """Parameter for Color types.

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

    red
    green
    blue

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
        self._update_internal_element()

    def _update_internal_element(self):
        if self.internal_element is None:
            return
        self.internal_element.color_value = self._value

    def _callback(self, value):
        old_value = self.value
        self._value = value
        if np.any(abs(self.value - old_value) > 1e-6):
            self.on_update(self.value)
        self._update_references()

    def get_gui_widget(self, font_size):
        self._internal_element = gui.ColorEdit()
        self._update_internal_element()
        label = gui.Label(self.pretty_name)

        color_selector = self.internal_element
        color_selector.set_on_value_changed(self._callback)

        element = gui.VGrid(2, 0.25 * font_size)
        element.add_child(label)
        element.add_child(color_selector)
        self._enable_internal_element(not self.is_reference)

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
        self._warn_unused_parameters(other_kwargs)
