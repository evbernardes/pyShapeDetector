#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:02:34 2023

@author: ebernardes
"""

from .primitivebase import Primitive
from .plane import Plane
from .planebounded import PlaneBounded
from .planetriangulated import PlaneTriangulated
from .planerectangular import PlaneRectangular
from .sphere import Sphere
from .cylinder import Cylinder
from .cone import Cone
from .line import Line

list_primitives = [
    Plane,
    Sphere,
    Cylinder,
    PlaneBounded,
    PlaneTriangulated,
    PlaneRectangular,
    Cone,
    Line,
]

dict_primitives = {p._name: p for p in list_primitives}
