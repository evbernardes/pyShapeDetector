#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-17 16:54:48

@author: evbernardes
"""


import extensions_simple
import extensions_create_shapes
import extensions_edit_pointclouds
import extensions_edit_planes

default_extensions = []
default_extensions += extensions_simple.extensions
default_extensions += extensions_create_shapes.extensions
default_extensions += extensions_edit_pointclouds.extensions
default_extensions += extensions_edit_planes.extensions
