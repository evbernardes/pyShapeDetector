#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:51:40 2023

@author: ebernardes
"""

import pytest

import numpy as np
from numpy.testing import assert_equal

from pyShapeDetector.primitives import Plane, Sphere, Cylinder
primitives = [Plane, Sphere, Cylinder]

def test_primitive_init():
    for primitive in primitives:
        model = np.random.rand(primitive._model_args_n)
        shape = primitive(model)
        assert_equal(shape.model, model)

        with pytest.raises(ValueError, match='elements, got'): 
            model = np.random.rand(primitive._model_args_n+1)
            shape = primitive(model)

        with pytest.raises(ValueError, match='elements, got'): 
            model = np.random.rand(primitive._model_args_n-1)
            shape = primitive(model)


def test_fit():
    for primitive in primitives:
        model = np.random.rand(primitive._model_args_n)
        shape = primitive(model)

        mesh = shape.get_mesh()