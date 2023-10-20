#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:02:34 2023

@author: ebernardes
"""

from .primitivebase import PrimitiveBase
from .plane import Plane
from .sphere import Sphere
from .cylinder import Cylinder

list_primitives=[
    Plane, Sphere, Cylinder,
]
