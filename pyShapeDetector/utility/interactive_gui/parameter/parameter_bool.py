#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 09:54:34

@author: evbernardes
"""
from typing import Callable, Union
from open3d.visualization import gui
from .parameter import Parameter


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
