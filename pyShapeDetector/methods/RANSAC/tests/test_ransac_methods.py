#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr Nov 21 13:03:40 2024

@author: ebernardes
"""

import pytest
import copy
import numpy as np
from numpy.testing import (
    assert_allclose,
)
import warnings

from pyShapeDetector.methods import RANSAC_Classic, BDSAC, LDSAC
from pyShapeDetector.primitives import (
    Plane,
    # PlaneBounded,
    # PlaneTriangulated,
    # PlaneRectangular,
    Sphere,
    Cylinder,
    # Cone,
)


def get_shape(primitive, num_points, canonical=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        shape = primitive.random()
        pcd = shape.sample_PointCloud_uniformly(num_points)
        shape.set_inliers(pcd)

        if canonical:
            shape = shape.canonical

        shape.inliers.estimate_normals()

        return shape


def get_angle_error(axis1, axis2):
    norm1 = np.linalg.norm(axis1)
    norm2 = np.linalg.norm(axis2)
    cos_angle = np.dot(axis1, axis2) / (norm1 * norm2)
    cos_angle = np.clip(np.abs(cos_angle), 0, 1)
    return np.arccos(cos_angle)


def test_fit():
    for method in [RANSAC_Classic, BDSAC, LDSAC]:
        detector = method()

        # testing Cylinder separately
        for i in range(5):
            detector.remove_all()
            detector.add(Cylinder)
            shape = get_shape(Cylinder, 1000, canonical=True)
            shape_fit = detector.fit(shape.inliers)[0].canonical
            # test axis instead of vector for direct fit
            assert get_angle_error(shape.axis, shape_fit.axis) < 1e-2
            assert_allclose(shape.radius, shape_fit.radius, rtol=1e-2, atol=1e-2)

        # # testing Cone separately
        # for i in range(5):
        #     detector.remove_all()
        #     detector.add(Cone)
        #     shape = get_shape(Cone, 1000, canonical=True)
        #     shape_fit = detector.fit(shape.inliers)[0].canonical
        #     assert_allclose(shape.appex, shape_fit.appex, rtol=1e-1, atol=1e-1)
        #     assert_allclose(
        #         shape.half_angle, shape_fit.half_angle, rtol=1e-1, atol=1e-1
        #     )
        #     # assert_allclose(shape.vector, shape_fit.vector, rtol=1e-2, atol=1e-2)
        #     # assert_allclose(shape.radius, shape_fit.radius, rtol=1e-2, atol=1e-2)

        for primitive in [Plane, Sphere]:
            # for primitive in [Sphere]:
            detector.remove_all()
            detector.add(primitive)
            for i in range(5):
                shape = get_shape(primitive, 100, canonical=True)
                shape_fit = detector.fit(shape.inliers)[0].canonical
                assert_allclose(shape.model, shape_fit.model, rtol=1e-2, atol=1e-2)
