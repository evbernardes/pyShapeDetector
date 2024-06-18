#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:47:35 2024

@author: ebernardes
"""

import pytest
import numpy as np

from pyShapeDetector.primitives import Plane, PlaneBounded, Sphere, Cylinder, Cone, Line
from pyShapeDetector.utility import PrimitiveLimits


def get_x(vector):
    return vector[0]


def test_init():
    func = [None, get_x]
    attributes = ["radius", "center"]
    bounds = [(0, 1), (0, 1)]

    limits1 = PrimitiveLimits(["radius", bounds[0]])
    limits1_alt = PrimitiveLimits([None, attributes[0], bounds[0]])
    limits2 = PrimitiveLimits([func[1], attributes[1], bounds[1]])
    assert limits1 == limits1_alt
    assert not limits1 == limits2

    limits_sum = limits1 + limits2
    limits3 = PrimitiveLimits(list(zip(func, attributes, bounds)))
    assert limits_sum == limits3

    with pytest.raises(ValueError):
        PrimitiveLimits([1, "radius", bounds[0]])

    limits_inf_neg = PrimitiveLimits(["radius", (None, 1)])
    assert limits_inf_neg.bounds[0] == (-np.inf, 1)
    limits_inf_pos = PrimitiveLimits(["radius", (1, None)])
    assert limits_inf_pos.bounds[0] == (1, np.inf)
    with pytest.raises(ValueError):
        PrimitiveLimits(["radius", (None, None)])


def test_compatibility():
    plane = Plane.random()
    planebounded = PlaneBounded.random()
    sphere = Sphere.random()
    cylinder = Cylinder.random()
    cone = Cone.random()
    line = Line.random()

    limits = PrimitiveLimits(["radius", [0, 1]])

    assert not limits.check_compatibility(plane)
    assert not limits.check_compatibility(planebounded)
    assert not limits.check_compatibility(line)

    assert limits.check_compatibility(sphere)
    assert limits.check_compatibility(cylinder)
    assert limits.check_compatibility(cone)


def test_check():
    limits_radius = PrimitiveLimits(["radius", [0, 1]])
    limits_x = PrimitiveLimits([get_x, "center", [0, 1]])

    spheres = [Sphere.from_center_radius([x, 1, 1], x) for x in range(3)]

    assert limits_radius.check(spheres[0])
    assert limits_radius.check(spheres[1])
    assert not limits_radius.check(spheres[2])

    assert limits_x.check(spheres[0])
    assert limits_x.check(spheres[1])
    assert not limits_x.check(spheres[2])


def test_check_two_conditions():
    limits_radius = PrimitiveLimits(["radius", (0, 1)])
    limits_position = PrimitiveLimits([get_x, "center", (0, 1)])
    limits_sum = limits_radius + limits_position

    sphere_too_big = Sphere.from_center_radius([0.5, 0, 0], 3)
    assert not limits_radius.check(sphere_too_big)
    assert limits_position.check(sphere_too_big)
    assert not limits_sum.check(sphere_too_big)

    sphere_outside = Sphere.from_center_radius([3, 0, 0], 0.5)
    assert limits_radius.check(sphere_outside)
    assert not limits_position.check(sphere_outside)
    assert not limits_sum.check(sphere_outside)

    sphere_ok = Sphere.from_center_radius([0.5, 0, 0], 0.5)
    assert limits_radius.check(sphere_ok)
    assert limits_position.check(sphere_ok)
    assert limits_sum.check(sphere_ok)

    sphere_problematic = Sphere.from_center_radius([3, 0, 0], 3)
    assert not limits_radius.check(sphere_problematic)
    assert not limits_position.check(sphere_problematic)
    assert not limits_sum.check(sphere_problematic)
