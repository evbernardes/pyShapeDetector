#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-23 14:52:02

@author: evbernardes
"""
import pytest
from open3d.visualization import gui
from pyShapeDetector.editor import Editor, Extension
from pyShapeDetector.editor.parameter import (
    ParameterBase,
    ParameterNumeric,
    ParameterBool,
    ParameterColor,
)

editor_instance = Editor(load_default_extensions=False)
default_settings = editor_instance._settings


def test_parameter_numeric():
    for type_ in (int, float):
        # Correct with limits
        descriptor = {"type": type_, "default": 0}
        parameter = ParameterBase.create_from_dict("param_name", descriptor)
        assert isinstance(parameter, ParameterNumeric)
        extension = {
            "function": lambda var: [],
            "inputs": None,
            "parameters": {"var": descriptor},
        }
        Extension(extension, default_settings)

        # Correct with default and limits
        descriptor = {"type": type_, "default": 0, "limits": [0, 2]}
        parameter = ParameterBase.create_from_dict("param_name", descriptor)
        extension = {
            "function": lambda var: [],
            "inputs": None,
            "parameters": {"var": descriptor},
        }
        Extension(extension, default_settings)

        with pytest.raises(TypeError, match="No limits or default value"):
            descriptor = {"type": type_}
            ParameterBase.create_from_dict("radius", descriptor)

        with pytest.raises(TypeError, match="should be list or tuple of 2 elements"):
            descriptor = {"type": type_, "limits": (1, 2, 3)}
            ParameterBase.create_from_dict("var", descriptor)

        with pytest.raises(TypeError, match="Expected boolean for 'use_slider'"):
            descriptor = {"type": type_, "limits": (1, 2), "use_slider": "asd"}
            ParameterBase.create_from_dict("var", descriptor)

        # Testing valid descriptors with useless parameters
        with pytest.warns(UserWarning, match="unexpected 'options' descriptor"):
            descriptor = {"type": type_, "default": 0, "options": [1, 2]}
            ParameterBase.create_from_dict("var", descriptor)


def test_parameter_bool():
    # Correct with limits
    for type_ in (bool, "bool"):
        good_descriptor = {"type": type_, "default": False}
        parameter = ParameterBase.create_from_dict("param_name", good_descriptor)
        assert isinstance(parameter, ParameterBool)
        extension = {
            "function": lambda var: [],
            "inputs": None,
            "parameters": {"var": good_descriptor},
        }
        Extension(extension, default_settings)

        with pytest.warns(UserWarning, match="not a boolean, automatically converting"):
            descriptor = {"type": type_, "default": "false"}
            ParameterBase.create_from_dict("var", descriptor)

        # Testing valid descriptors with useless parameters
        with pytest.warns(UserWarning, match="unexpected 'options' descriptor"):
            descriptor = {"type": type_, "options": [1, 2]}
            ParameterBase.create_from_dict("var", descriptor)


def test_parameter_color():
    for type_ in ("color", gui.Color):
        # Correct with input color
        good_descriptor = {"type": type_, "default": (0, 0, 0)}
        parameter = ParameterBase.create_from_dict("param_name", good_descriptor)
        assert isinstance(parameter, ParameterColor)
        extension = {
            "function": lambda var: [],
            "inputs": None,
            "parameters": {"var": good_descriptor},
        }
        Extension(extension, default_settings)

        # Correct without input color, defaults to (0, 0, 0)
        descriptor = {"type": type_}
        ParameterBase.create_from_dict("var", descriptor)
        assert isinstance(parameter, ParameterColor)
        assert parameter.red == 0
        assert parameter.blue == 0
        assert parameter.green == 0

        with pytest.raises(
            TypeError, match="color should be a gui.Color, list or tuple"
        ):
            descriptor = {"type": type_, "default": "red"}
            ParameterBase.create_from_dict("var", descriptor)

        # Testing valid descriptors with useless parameters
        with pytest.warns(UserWarning, match="unexpected 'options' descriptor"):
            descriptor = {"type": type_, "options": [1, 2]}
            ParameterBase.create_from_dict("var", descriptor)
