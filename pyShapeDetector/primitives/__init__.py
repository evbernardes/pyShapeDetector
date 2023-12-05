#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:02:34 2023

@author: ebernardes
"""

from .primitivebase import Primitive
from .plane import Plane, PlaneBounded
from .sphere import Sphere
from .cylinder import Cylinder
from .cone import Cone

from .utilities import group_similar_shapes, fuse_shapes

list_primitives=[
    Plane, Sphere, Cylinder, PlaneBounded, Cone,
]
