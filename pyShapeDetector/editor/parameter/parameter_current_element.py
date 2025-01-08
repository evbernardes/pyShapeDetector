#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 09:54:34

@author: evbernardes
"""
from typing import Callable, Union
from open3d.visualization import gui
from .parameter import ParameterBase


class ParameterCurrentElement(ParameterBase[bool]):
    """Convenienc Parameter current element

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

    _value = None
    _type = type
    _valid_arguments = ["label", "type", "subpanel"]

    def _update_internal_element(self):
        pass

    def create_reference(self):
        raise TypeError("Cannot reference instance ot ParameterCurrentElement.")

    def get_gui_widget(self, font_size):
        element = gui.VGrid(1, 0.25 * font_size)
        element.add_child(gui.Label(self.label))
        text_edit = gui.TextEdit()
        text_edit.placeholder_text = "CURRENT ELEMENT"
        text_edit.enabled = False
        element.add_child(text_edit)
        return element

    def __init__(
        self,
        label: str,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(label=label, on_update=None, subpanel=subpanel)
        self._warn_unused_parameters(other_kwargs)
