#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 09:48:24

@author: evbernardes
"""

from numpy import ndarray
from open3d.visualization.gui import Color
from pathlib import Path

from .parameter import ParameterBase
from .parameter_bool import ParameterBool
from .parameter_options import ParameterOptions
from .parameter_numeric import ParameterNumeric
from .parameter_color import ParameterColor
from .parameter_ndarray import ParameterNDArray
from .parameter_current_element import ParameterCurrentElement
from .parameter_path import ParameterPath
from .parameter_panel import ParameterPanel

PARAMETER_TYPE_DICTIONARY = {
    None: ParameterBase,
    bool: ParameterBool,
    "bool": ParameterBool,
    int: ParameterNumeric,
    "int": ParameterNumeric,
    float: ParameterNumeric,
    "float": ParameterNumeric,
    list: ParameterOptions,
    "list": ParameterOptions,
    ndarray: ParameterNDArray,
    "array": ParameterNDArray,
    Color: ParameterColor,
    "color": ParameterColor,
    "current": ParameterCurrentElement,
    Path: ParameterPath,
    "path": ParameterPath,
}
