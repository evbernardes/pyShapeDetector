#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-23 14:52:02

@author: evbernardes
"""
import pytest
from pyShapeDetector.editor import Editor, Extension
from pyShapeDetector.editor.parameter import (
    ParameterBase,
    ParameterNumeric,
    ParameterBool,
)

editor_instance = Editor(load_default_extensions=False)
default_settings = editor_instance._settings


def test_param_numeric():
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
        descriptor = {"type": type_, "default": 0, "options": [1, 2]}
        with pytest.warns(UserWarning, match="unexpected 'options' descriptor"):
            ParameterBase.create_from_dict("var", descriptor)


def test_param_bool():
    # Correct with limits
    good_descriptor = {"type": bool, "default": False}
    parameter = ParameterBase.create_from_dict("param_name", good_descriptor)
    assert isinstance(parameter, ParameterBool)
    extension = {
        "function": lambda var: [],
        "inputs": None,
        "parameters": {"var": good_descriptor},
    }
    Extension(extension, default_settings)

    descriptor = {"type": bool, "default": "false"}
    with pytest.warns(UserWarning, match="not a boolean, automatically converting"):
        ParameterBase.create_from_dict("var", descriptor)

    # Testing valid descriptors with useless parameters
    descriptor = {"type": bool, "options": [1, 2]}
    with pytest.warns(UserWarning, match="unexpected 'options' descriptor"):
        ParameterBase.create_from_dict("var", descriptor)
