#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 09:50:31

@author: evbernardes
"""

import warnings
from typing import Callable, Union, TypeVar, Generic
from abc import ABC, abstractmethod
from open3d.visualization import gui

from pyShapeDetector.editor.element import ElementContainer

T = TypeVar("T")


class ParameterBase(ABC, Generic[T]):
    """Base class to define parameter types that also generate GUI widgets.

    To Inherit, implement the following abstract methods:
        _update_internal_element
        _update_references
        _enable_internal_element
        get_gui_widget

    Attributes
    ----------
    internal_element
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

    _type = None.__class__
    _valid_arguments = ["label", "type", "default", "on_update", "subpanel"]

    @property
    def internal_element(self):
        return self._internal_element

    @property
    def references(self: T) -> list[T]:
        return self._references

    @property
    def is_reference(self) -> bool:
        return self._is_reference

    @property
    def valid_arguments(self) -> list:
        return self._valid_arguments

    @property
    def type(self) -> type:
        return self._type

    @property
    def value(self) -> T:
        return self._value

    @property
    def type_name(self) -> str:
        return self._type.__name__

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, new_label: str):
        if not isinstance(new_label, str):
            raise TypeError(
                f"Expected string as label for Parameter, got {type(new_label)}."
            )
        self._label = new_label

    def __repr__(self):
        return self.__class__.__name__ + f"('{self.label}', {self.value})"

    # @property
    # def label(self) -> str:
    #     words = self.label.replace("_", " ").split()
    #     result = []
    #     for word in words:
    #         if word.isupper():  # Keep existing UPPERCASE values as is
    #             result.append(word)
    #         else:  # Capitalize other words
    #             result.append(word.capitalize())

    #     return " ".join(result)

    @property
    def subpanel(self) -> str:
        return self._subpanel

    @subpanel.setter
    def subpanel(self, new_subpanel: Union[str, None]):
        if new_subpanel is not None and not isinstance(new_subpanel, str):
            raise TypeError(
                f"Expected string as subpanel for Parameter, got {type(new_subpanel)}."
            )
        self._subpanel = new_subpanel

    @property
    def on_update(self) -> Union[Callable, None]:
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

    @abstractmethod
    def _update_internal_element(self):
        pass

    def _callback(self, value):
        # old_value = self.value
        self.value = value
        # if abs(self.value - old_value) > 1e-6:
        self.on_update(self.value)
        self._update_references()

    def _reset_values_and_limits(self, elements: ElementContainer):
        """Resets values and limits if needed"""
        pass

    def _update_references(self) -> None:
        """Updates references to class"""
        for reference in self._references:
            reference.value = self.value

    def _enable_internal_element(self, value: bool) -> None:
        """Enables/Disables internal element when creating references"""
        if self.internal_element is not None:
            self.internal_element.enabled = value

    @abstractmethod
    def get_gui_widget(self, font_size: float) -> gui.Widget:
        pass

    def create_reference(self: T) -> T:
        """Creates a new unusable copy of the parameter that is updated when
        the original is updated.

        Returns
        -------
        T
            _description_
        """
        kwargs = {
            key: getattr(self, key, None)
            for key in self.valid_arguments
            if key != "default" and key != "subpanel"
        }
        kwargs["default"] = self.value

        new_parameter = ParameterBase.create_from_dict(self.label, kwargs)
        new_parameter._enable_internal_element(False)

        new_parameter._is_reference = True
        self._references.append(new_parameter)
        return new_parameter

    def _warn_unused_parameters(self, other_kwargs: dict):
        """Warns user that useless arguments were present in the parameter descriptor"""
        for key in other_kwargs:
            if key in self.valid_arguments:
                continue

            warnings.warn(
                f"Ignoring unexpected '{key}' descriptor in parameter "
                f"'{self.label}' of type '{self.type_name}'."
            )

    @staticmethod
    def create_from_dict(key: str, parameter_descriptor: dict) -> "ParameterBase":
        """Parse a parameter descriptor dictionary and return the correct parameter type

        Parameters
        ----------
        key : str
            Key/variable name of the parameter
        parameter_descriptor : dict
            Dictionary defining parameter.

        Returns
        -------
        ParameterBase
            Create parameter

        Raises
        ------
        ValueError
            It type is invalid
        """

        from .__init__ import PARAMETER_TYPE_DICTIONARY

        parameter_descriptor = parameter_descriptor.copy()

        if "label" not in parameter_descriptor:
            parameter_descriptor["label"] = key
        _type = parameter_descriptor.get("type", None)

        if _type not in PARAMETER_TYPE_DICTIONARY:
            raise ValueError(f"{_type} does not correspond to valid Parameter type.")

        parameter = PARAMETER_TYPE_DICTIONARY[_type](**parameter_descriptor)

        return parameter

    def __init__(
        self,
        label: str,
        on_update: Callable = None,
        subpanel: Union[str, None] = None,
    ):
        self._references = []
        self._is_reference = False
        self.label = label
        self.on_update = on_update
        self.subpanel = subpanel
        self._internal_element = None
        self._gui_widget = None
