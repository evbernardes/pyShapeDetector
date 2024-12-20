#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 09:54:34

@author: evbernardes
"""
from typing import Callable, Union
from open3d.visualization import gui
from .parameter import ParameterBase


class ParameterOptions(ParameterBase[list]):
    """Parameter for choosing between multiple options.

    Creates a combobox

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

    options

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
    create
    """

    _type = list
    _valid_arguments = ParameterBase._valid_arguments + ["options"]

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, new_options):
        if not isinstance(new_options, (tuple, list)) or len(new_options) == 0:
            raise ValueError(
                f"Parameter {self.label} requires non-empty of values for "
                f"options, got {new_options}."
            )

        self._options = list(new_options)

    @ParameterBase.value.setter
    def value(self, new_value):
        if new_value not in self.options:
            raise ValueError(
                f"Value '{new_value}' for parameter '{self.label}' is not in "
                f"options list {self.options}."
            )

        self._value = new_value

        # if self.is_reference:
        self._update_internal_element()

    def _update_internal_element(self):
        if self.internal_element is None:
            return
        options_strings = [str(option) for option in self.options]
        self.internal_element.selected_index = options_strings.index(str(self.value))

    def _callback(self, text, index):
        self._value = self.options[index]
        self.on_update(self.value)
        self._update_references()

    def get_gui_widget(self, font_size):
        self._internal_element = gui.Combobox()
        for option in self.options:
            self._internal_element.add_item(str(option))
        self._update_internal_element()

        label = gui.Label(self.label)

        combobox = self.internal_element
        combobox.set_on_selection_changed(self._callback)

        element = gui.VGrid(2, 0.25 * font_size)
        element.add_child(label)
        element.add_child(combobox)
        self._enable_internal_element(not self.is_reference)

        return element

    def __init__(
        self,
        label: str,
        options: list,
        default=None,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
        **other_kwargs,
    ):
        super().__init__(label=label, on_update=on_update, subpanel=subpanel)
        self.options = options

        if default is None:
            default = options[0]

        self.value = default

        self._warn_unused_parameters(other_kwargs)
