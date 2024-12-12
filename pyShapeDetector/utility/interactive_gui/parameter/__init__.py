#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-12 09:48:24

@author: evbernardes
"""

from numpy import ndarray
from open3d.visualization.gui import Color

from .parameter import Parameter
from .parameter_panel import ParameterPanel

from .parameter_bool import ParameterBool
from .parameter_options import ParameterOptions
from .parameter_float import ParameterFloat
from .parameter_int import ParameterInt
from .parameter_color import ParameterColor
from .parameter_ndarray import ParameterNDArray

PARAMETER_TYPE_DICTIONARY = {
    None: Parameter,
    bool: ParameterBool,
    "bool": ParameterBool,
    int: ParameterInt,
    "int": ParameterInt,
    float: ParameterFloat,
    "float": ParameterFloat,
    list: ParameterOptions,
    "list": ParameterOptions,
    ndarray: ParameterNDArray,
    "array": ParameterNDArray,
    Color: ParameterColor,
    "color": ParameterColor,
}
