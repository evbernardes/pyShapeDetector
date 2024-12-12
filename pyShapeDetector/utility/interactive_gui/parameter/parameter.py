#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 09:50:31

@author: evbernardes
"""

import traceback
import warnings
from typing import Callable, Union
import numpy as np
from open3d.visualization import gui
from pyShapeDetector.utility.interactive_gui.editor_app import Editor


class Parameter:
    _type = None.__class__

    @property
    def internal_element(self):
        return self._internal_element

    @property
    def type_name(self):
        return self._type.__name__

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise TypeError(
                f"Expected string as name for Parameter, got {type(new_name)}."
            )
        self._name = new_name

    @property
    def subpanel(self):
        return self._subpanel

    @subpanel.setter
    def subpanel(self, new_subpanel):
        if new_subpanel is not None and not isinstance(new_subpanel, str):
            raise TypeError(
                f"Expected string as subpanel for Parameter, got {type(new_subpanel)}."
            )
        self._subpanel = new_subpanel

    @property
    def pretty_name(self):
        words = self.name.replace("_", " ").split()
        result = []
        for word in words:
            if word.isupper():  # Keep existing UPPERCASE values as is
                result.append(word)
            else:  # Capitalize other words
                result.append(word.capitalize())

        return " ".join(result)

    @property
    def on_update(self):
        if self._on_update is None:
            return lambda value: None
        return self._on_update

    @on_update.setter
    def on_update(self, func: Callable):
        if func is None:
            self._on_update = None
        elif callable(func):
            self._on_update = func
        else:
            raise TypeError(
                f"Parameter of type '{self.type_name}' received invalid 'on_update'"
            )

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, new_type):
        if not isinstance(new_type, type):
            raise TypeError("parameter descriptor has invalid type.")

        self._type = new_type

    @property
    def value(self):
        return self._value

    def _warn_unused_parameters(self, other_kwargs: dict):
        for key in other_kwargs:
            warnings.warn(
                f"Ignoring unexpected '{key}' descriptor in parameter "
                f"'{self.name}' of type '{self.type_name}'."
            )

    def _callback(self, value, text_edit=None):
        old_value = self.value
        try:
            self._value = self.type(value)
            if abs(self.value - old_value) > 1e-6:
                self.on_update(self.value)
        except Exception:
            if text_edit is not None:
                text_edit.text_value = str(self.value)
        self._update_references()

    def _reset_values_and_limits(self, editor_instance: Editor):
        pass

    def _update_references(self):
        for reference in self._references:
            reference.value = self.value

    def get_gui_element(self, font_size):
        label = gui.Label(self.pretty_name)

        # Text field for general inputs
        text_edit = gui.TextEdit()
        # text_edit.placeholder_text = str(self.value)
        text_edit.set_on_value_changed(lambda value: self._callback(value, text_edit))

        element = gui.VGrid(2, 0.25 * font_size)
        element.add_child(label)
        element.add_child(text_edit)

        return element

    @staticmethod
    def create_from_dict(key: str, parameter_descriptor: dict):
        from .__init__ import PARAMETER_TYPE_DICTIONARY

        parameter_descriptor = parameter_descriptor.copy()
        if "name" not in parameter_descriptor:
            parameter_descriptor["name"] = key
        _type = parameter_descriptor.pop("type", None)
        if _type not in PARAMETER_TYPE_DICTIONARY:
            raise ValueError(f"{_type} does not correspond to valid Parameter type.")

        parameter = PARAMETER_TYPE_DICTIONARY[_type](**parameter_descriptor)

        return parameter

    def __init__(
        self,
        name: str,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
    ):
        self._references = []
        self.name = name
        self.on_update = on_update
        self.subpanel = subpanel

    def create_reference(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Ignoring unexpected.*")
            new_parameter = Parameter.create_from_dict(
                self.name,
                {
                    "type": self.type,
                    "default": self.value,
                    "limits": getattr(self, "limits", None),
                    "limit_setter": getattr(self, "limit_setter", None),
                    "options": getattr(self, "options", None),
                },
            )

        if isinstance(self.internal_element, list):
            for elem in np.array(new_parameter.internal_element).flatten():
                elem.enabled = False
        else:
            new_parameter.internal_element.enabled = False

        self._references.append(new_parameter)
        return new_parameter
