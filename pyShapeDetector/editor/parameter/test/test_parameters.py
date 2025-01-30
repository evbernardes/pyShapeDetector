#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-23 14:52:02

@author: evbernardes
"""
import pytest
import numpy as np
from pathlib import Path
from numpy.testing import assert_array_equal
from open3d.visualization import gui
from pyShapeDetector.editor import Editor
from pyShapeDetector.editor.extension import Extension
from pyShapeDetector.editor.parameter import (
    ParameterBase,
    ParameterNumeric,
    ParameterBool,
    ParameterColor,
    ParameterOptions,
    ParameterNDArray,
    ParameterPath,
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
    for type_ in (bool, "bool"):
        # Correct with default value
        descriptor = {"type": type_, "default": True}
        parameter = ParameterBase.create_from_dict("param_name", descriptor)
        assert isinstance(parameter, ParameterBool)
        assert parameter.value
        extension = {
            "function": lambda var: [],
            "inputs": None,
            "parameters": {"var": descriptor},
        }
        Extension(extension, default_settings)

        # No default value, defaults to False
        descriptor = {"type": type_}
        parameter = ParameterBase.create_from_dict("param_name", descriptor)
        assert isinstance(parameter, ParameterBool)
        assert not parameter.value

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
        descriptor = {"type": type_, "default": (1, 2, 3)}
        parameter = ParameterBase.create_from_dict("param_name", descriptor)
        assert isinstance(parameter, ParameterColor)
        extension = {
            "function": lambda var: [],
            "inputs": None,
            "parameters": {"var": descriptor},
        }
        Extension(extension, default_settings)

        # Correct without input color, defaults to (0, 0, 0)
        descriptor = {"type": type_}
        parameter = ParameterBase.create_from_dict("var", descriptor)
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


def test_parameter_options():
    for type_ in (list, "list", "options"):
        # Correct with options and default value
        descriptor = {"type": type_, "default": "s", "options": [10, "s", 3]}
        parameter = ParameterBase.create_from_dict("param_name", descriptor)
        assert isinstance(parameter, ParameterOptions)
        assert parameter.value == "s"
        extension = {
            "function": lambda var: [],
            "inputs": None,
            "parameters": {"var": descriptor},
        }
        Extension(extension, default_settings)

        # Correct without default value, defaults to first option
        descriptor = {"type": type_, "options": [10, "s", 3]}
        parameter = ParameterBase.create_from_dict("param_name", descriptor)
        assert isinstance(parameter, ParameterOptions)
        assert parameter.value == 10

        with pytest.raises(ValueError, match="requires non-empty list of options"):
            descriptor = {"type": type_, "options": []}
            ParameterBase.create_from_dict("var", descriptor)

        with pytest.raises(ValueError, match="requires non-empty list of options"):
            descriptor = {"type": type_, "options": 5}
            ParameterBase.create_from_dict("var", descriptor)

        with pytest.raises(ValueError, match="is not in options list"):
            descriptor = {"type": type_, "default": 3, "options": [1, 2]}
            ParameterBase.create_from_dict("var", descriptor)

        # Testing valid descriptors with useless parameters
        with pytest.warns(UserWarning, match="unexpected 'invalid_name' descriptor"):
            descriptor = {"type": type_, "invalid_name": 0, "options": [1, 2]}
            ParameterBase.create_from_dict("var", descriptor)


def test_parameter_ndarray():
    for type_ in (np.ndarray, "ndarray", "array"):
        # Correct with tuple
        descriptor = {"type": type_, "default": (0, 0, 0)}
        parameter = ParameterBase.create_from_dict("param_name", descriptor)
        assert isinstance(parameter, ParameterNDArray)
        extension = {
            "function": lambda var: [],
            "inputs": None,
            "parameters": {"var": descriptor},
        }
        Extension(extension, default_settings)

        # Correct with complex tuple
        value = ((0, 0, 0), (2, 3, 1))
        descriptor = {"type": type_, "default": value}
        parameter = ParameterBase.create_from_dict("var", descriptor)
        assert isinstance(parameter, ParameterNDArray)
        assert_array_equal(parameter.value, np.asarray(value))

        # Correct with complex list
        value = [(0, 5, 0), (2, 3, 1)]
        descriptor = {"type": type_, "default": value}
        parameter = ParameterBase.create_from_dict("var", descriptor)
        assert isinstance(parameter, ParameterNDArray)
        assert_array_equal(parameter.value, np.asarray(value))

        # Correct with array
        value = np.array([(0, 5, 0), (2, 3, 1)])
        descriptor = {"type": type_, "default": value}
        parameter = ParameterBase.create_from_dict("var", descriptor)
        assert isinstance(parameter, ParameterNDArray)
        assert_array_equal(parameter.value, value)

        with pytest.raises(TypeError, match="requires default value"):
            descriptor = {"type": type_}
            ParameterBase.create_from_dict("var", descriptor)

        with pytest.raises(
            TypeError, match="Supported values for dtype are 'int' and 'float'"
        ):
            descriptor = {"type": type_, "default": ["a", "b"]}
            ParameterBase.create_from_dict("var", descriptor)

        with pytest.raises(ValueError, match="shapes up to 2 dimensions are accepted"):
            descriptor = {"type": type_, "default": np.random.random((2, 2, 2))}
            ParameterBase.create_from_dict("var", descriptor)

        # Testing valid descriptors with useless parameters
        with pytest.warns(UserWarning, match="unexpected 'options' descriptor"):
            descriptor = {"type": type_, "default": (0, 0, 0), "options": [1]}
            parameter = ParameterBase.create_from_dict("param_name", descriptor)


def test_parameter_path():
    for type_ in (Path, "path"):
        # Correct with default value
        descriptor = {"type": type_, "default": "a/b"}
        parameter = ParameterBase.create_from_dict("param_name", descriptor)
        assert isinstance(parameter, ParameterPath)
        assert parameter.value == Path("a/b")
        assert parameter.path_type == "open"  # default path_type is "open"
        extension = {
            "function": lambda var: [],
            "inputs": None,
            "parameters": {"var": descriptor},
        }
        Extension(extension, default_settings)

        # No default value, defaults to ""
        descriptor = {"type": type_}
        parameter = ParameterBase.create_from_dict("param_name", descriptor)
        assert isinstance(parameter, ParameterPath)
        assert parameter.value == Path()

        descriptor = {"type": type_, "path_type": "open"}
        parameter = ParameterBase.create_from_dict("var", descriptor)
        assert parameter.path_type == "open"

        descriptor = {"type": type_, "path_type": "open_dir"}
        parameter = ParameterBase.create_from_dict("var", descriptor)
        assert parameter.path_type == "open_dir"

        descriptor = {"type": type_, "path_type": "save"}
        parameter = ParameterBase.create_from_dict("var", descriptor)
        assert parameter.path_type == "save"

        with pytest.raises(ValueError, match="Valid path types are"):
            descriptor = {"type": type_, "path_type": "load"}
            ParameterBase.create_from_dict("radius", descriptor)

        with pytest.raises(ValueError, match="suffixes expected to be dict"):
            descriptor = {"type": type_, "suffixes": []}
            ParameterBase.create_from_dict("radius", descriptor)

        with pytest.raises(ValueError, match="expected to be dict mapping str->str"):
            descriptor = {"type": type_, "suffixes": {1: 2}}
            ParameterBase.create_from_dict("radius", descriptor)

        with pytest.raises(ValueError, match="expected to be dict mapping str->str"):
            descriptor = {"type": type_, "suffixes": {"1": 2}}
            ParameterBase.create_from_dict("radius", descriptor)

        with pytest.raises(ValueError, match="expected to be dict mapping str->str"):
            descriptor = {"type": type_, "suffixes": {1: "2"}}
            ParameterBase.create_from_dict("radius", descriptor)

        with pytest.raises(ValueError, match="Invalid suffix"):
            descriptor = {"type": type_, "suffixes": {"1": "2"}}
            ParameterBase.create_from_dict("radius", descriptor)

        descriptor = {"type": type_, "suffixes": {".tar": "tar files"}}
        parameter = ParameterBase.create_from_dict("radius", descriptor)
        assert parameter.suffixes == {".tar": "tar files"}

        # Testing valid descriptors with useless parameters
        with pytest.warns(UserWarning, match="unexpected 'options' descriptor"):
            descriptor = {"type": type_, "options": [1, 2]}
            ParameterBase.create_from_dict("var", descriptor)
