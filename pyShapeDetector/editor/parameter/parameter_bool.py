#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 09:54:34

@author: evbernardes
"""
import warnings
from typing import Callable, Union
from open3d.visualization import gui
from .parameter import ParameterBase


class ParameterBool(ParameterBase[bool]):
    """Parameter for boolean types

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

    _type = bool

    @ParameterBase.value.setter
    def value(self, new_value):
        if not isinstance(new_value, bool):
            corrected = bool(new_value)
            warnings.warn(
                f"Value {new_value} is not a boolean, automatically converting "
                f"as {corrected}."
            )
            new_value = corrected
        self._value = new_value
        # if self.is_reference:
        self._update_internal_element()

    def _update_internal_element(self):
        if self.internal_element is None:
            return
        self.internal_element.checked = self.value

    def get_gui_widget(self, font_size):
        checkbox = self._internal_element = gui.Checkbox(self.label + "?")
        self._update_internal_element()
        checkbox.set_on_checked(self._callback)
        self._enable_internal_element(not self.is_reference)
        return checkbox

    def __init__(
        self,
        label: str,
        default: bool = False,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(label=label, on_update=on_update, subpanel=subpanel)
        self.value = default
        self._warn_unused_parameters(other_kwargs)
