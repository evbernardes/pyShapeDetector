#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-23 09:44:19

@author: evbernardes
"""
from pyShapeDetector.editor import Editor
import pytest

param_radius = {"radius": {"type": float, "default": 0}}


def test_extension_current_selected_global():
    editor_instance = Editor(load_default_extensions=False)

    for inputs in ("current", "selected", "global"):
        extension = {
            "name": "Correct extension",
            "function": lambda element, radius: [],
            "inputs": inputs,
            "parameters": param_radius,
        }
        editor_instance.add_extension(extension, testing=True)

        extension = {
            "name": "Missing in function signature",
            "function": lambda element: [],
            "inputs": inputs,
            "parameters": param_radius,
        }
        with pytest.raises(
            ValueError, match="missing parameters from function signature: {'radius'}"
        ):
            editor_instance.add_extension(extension, testing=True)

        extension = {
            "name": "Missing in descriptor",
            "function": lambda element, radius: [],
            "inputs": inputs,
        }
        with pytest.raises(ValueError, match="missing parameters from descriptor"):
            editor_instance.add_extension(extension, testing=True)

        extension = {
            "name": "Missing elements",
            "function": lambda radius: [],
            "inputs": inputs,
            "parameters": param_radius,
        }
        with pytest.raises(ValueError, match="Invalid number of arguments"):
            editor_instance.add_extension(extension, testing=True)


def test_extension_none():
    editor_instance = Editor(load_default_extensions=False)

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
            editor_instance.add_extension(extension, testing=True)

        extension = {
            "name": "Missing in descriptor",
            "function": lambda radius: [],
            "inputs": inputs,
        }
        with pytest.raises(
            ValueError, match="missing parameters from descriptor: {'radius'}"
        ):
            editor_instance.add_extension(extension, testing=True)

        extension = {
            "name": "Missing elements",
            "function": lambda element, radius: [],
            "inputs": inputs,
            "parameters": param_radius,
        }
        with pytest.raises(ValueError, match="missing parameters from descriptor"):
            editor_instance.add_extension(extension, testing=True)
        # with pytest.raises(ValueError, match="Invalid number of arguments"):
        #     editor_instance.add_extension(extension, testing=True)


if __name__ == "__main__":
    test_extension_none()
