#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 09:54:34

@author: evbernardes
"""
from typing import Callable, Union
from open3d.visualization import gui
from .parameter import ParameterBase


class ParameterBool(ParameterBase[bool]):
    """Parameter foor boolean types

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

    Methods
    -------
    _warn_unused_parameters
    _callback
    _update_internal_element
    _reset_values_and_limits
    _update_references
    _enable_internal_element
    _create_gui_widget
    get_gui_widget
    create_reference
    create_from_dict
    """

    _type = bool

    @ParameterBase.value.setter
    def value(self, new_value):
        self._value = bool(new_value)
        # if self.is_reference:
        self._update_internal_element()

    def _update_internal_element(self):
        if self.internal_element is None:
            return
        self.internal_element.checked = self.value

    def _create_gui_widget(self, font_size):
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
