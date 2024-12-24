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
from .input_selector import InputSelector, SingleChoiceSelector

from .helpers_internal import (
    parallelize,
    _set_and_check_3d_array,
    accept_one_or_multiple_elements,
    combine_indices_to_remove,
)

from .helpers_math import (
    midrange,
    get_area_with_shoelace,
    check_vertices_clockwise,
    get_rotation_from_axis,
    rgb_to_cielab,
    cielab_to_rgb,
)

from .helpers_visualization import (
    get_inputs,
    select_function_with_gui,
    get_painted,
    get_open3d_geometries,
    draw_geometries,
    draw_two_columns,
    # draw_and_ask,
    select_manually,
    apply_function_manually,
    select_combinations_manually,
)

from .helpers_io import (
    mesh_to_obj_description,
    write_obj,
    create_unity_package,
    check_existance,
    save_elements,
    save_ask,
    ask_and_save,
    load_pointcloud_from_e57,
)
