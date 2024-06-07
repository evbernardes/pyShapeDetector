#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:02:34 2023

Collection of useful methods.

Internal


Math-related
-----------------
parallelize
    
Visualization-related
---------------------
paint_random
draw_geometries
draw_two_columns

Input/Output-related
--------------------
check_existance
save_elements
save_ask
ask_and_save

@author: ebernardes
"""

from .multidetector import MultiDetector
from .primitivelimits import PrimitiveLimits
from .detector_options import DetectorOptions

from .helpers_internal import (
    parallelize,
    )

from .helpers_math import (
    get_rotation_from_axis,
    rgb_to_cielab,
    cielab_to_rgb,
    )

from .helpers_visualization import (
    paint_random,
    draw_geometries,
    draw_two_columns,
    draw_and_ask
    )

from .helpers_io import (
    check_existance,
    save_elements,
    save_ask,
    ask_and_save,
    )
