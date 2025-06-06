#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-17 16:54:48

@author: evbernardes
"""

from .extensions_internals import extensions as extensions_internals
from .extensions_io import extensions as extensions_io
from .extensions_create_shapes import extensions as extensions_create_shapes
from .extensions_inliers import extensions as extensions_inliers
from .extensions_edit_transform import extensions as extensions_edit_transform
from .extensions_edit_pointclouds import extensions as extensions_edit_pointclouds
from .extensions_edit_planes import extensions as extensions_edit_planes
from .extensions_edit_meshes import extensions as extensions_edit_meshes
from .extensions_misc import extensions as extensions_misc

default_extensions = []
default_extensions += extensions_internals
default_extensions += extensions_io
default_extensions += extensions_create_shapes
default_extensions += extensions_inliers
default_extensions += extensions_edit_transform
default_extensions += extensions_edit_pointclouds
default_extensions += extensions_edit_planes
default_extensions += extensions_edit_meshes
default_extensions += extensions_misc
