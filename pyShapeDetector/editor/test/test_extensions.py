#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-23 09:44:19

@author: evbernardes
"""
from pyShapeDetector.editor import Editor, Extension
import pytest

param_radius = {"radius": {"type": float, "default": 0}}

editor_instance = Editor(load_default_extensions=False)
default_settings = editor_instance._settings


def test_loading_default_extensions():
    editor_instance_with_extensions = Editor(load_default_extensions=True, testing=True)


def test_extension_current_selected_global():
    for inputs in ("current", "selected", "global"):
        extension = {
            "name": "Correct extension",
            "function": lambda element, radius: [],
            "inputs": inputs,
            "parameters": param_radius,
        }
        Extension(extension, default_settings)

        extension = {
            "name": "Missing in function signature",
            "function": lambda element: [],
            "inputs": inputs,
            "parameters": param_radius,
        }
        with pytest.raises(
            ValueError, match="missing parameters from function signature: {'radius'}"
        ):
            Extension(extension, default_settings)

        extension = {
            "name": "Missing in descriptor",
            "function": lambda element, radius: [],
            "inputs": inputs,
        }
        with pytest.raises(ValueError, match="missing parameters from descriptor"):
            Extension(extension, default_settings)

        extension = {
            "name": "Missing elements",
            "function": lambda radius: [],
            "inputs": inputs,
            "parameters": param_radius,
        }
        with pytest.raises(ValueError, match="Invalid number of arguments"):
            Extension(extension, default_settings)


def test_extension_none():
    for inputs in ("none", None):
        extension = {
            "name": "Missing in function signature",
            "function": lambda: [],
            "inputs": inputs,
            "parameters": param_radius,
        }
        with pytest.raises(
            ValueError, match="missing parameters from function signature: {'radius'}"
        ):
            Extension(extension, default_settings)

        extension = {
            "name": "Missing in descriptor",
            "function": lambda radius: [],
            "inputs": inputs,
        }
        with pytest.raises(
            ValueError, match="missing parameters from descriptor: {'radius'}"
        ):
            Extension(extension, default_settings)

        extension = {
            "name": "Missing elements",
            "function": lambda element, radius: [],
            "inputs": inputs,
            "parameters": param_radius,
        }
        # Can't test this properly yet:
        with pytest.raises(ValueError, match="missing parameters from descriptor"):
            Extension(extension, default_settings)
        # with pytest.raises(ValueError, match="Invalid number of arguments"):
        #     Extension(extension, settings)


def test_param_int_float():
    for type_ in (int, float):
        extension = {
            "name": "Correct with default",
            "function": lambda radius: [],
            "inputs": None,
            "parameters": {"radius": {"type": type_, "default": 0}},
        }
        Extension(extension, default_settings)

        extension = {
            "name": "Correct with limits",
            "function": lambda radius: [],
            "inputs": None,
            "parameters": {"radius": {"type": type_, "limits": [0, 2]}},
        }
        Extension(extension, default_settings)

        extension = {
            "name": "Correct with default and limits",
            "function": lambda radius: [],
            "inputs": None,
            "parameters": {"radius": {"type": type_, "default": 0, "limits": [0, 2]}},
        }
        Extension(extension, default_settings)

        extension = {
            "name": "No default or limits",
            "function": lambda radius: [],
            "inputs": None,
            "parameters": {"radius": {"type": type_}},
        }
        with pytest.raises(TypeError, match="No limits or default value"):
            Extension(extension, default_settings)

        extension = {
            "name": "Invalid limits",
            "function": lambda radius: [],
            "inputs": None,
            "parameters": {"radius": {"type": type_, "limits": (1, 2, 3)}},
        }
        with pytest.raises(
            TypeError, match="Limits should be list or tuple of 2 elements"
        ):
            Extension(extension, default_settings)

        extension = {
            "name": "Invalid slider",
            "function": lambda radius: [],
            "inputs": None,
            "parameters": {
                "radius": {"type": type_, "limits": (1, 2), "use_slider": "asd"}
            },
        }
        with pytest.raises(TypeError, match="Expected boolean for 'use_slider'"):
            Extension(extension, default_settings)


if __name__ == "__main__":
    test_extension_none()
