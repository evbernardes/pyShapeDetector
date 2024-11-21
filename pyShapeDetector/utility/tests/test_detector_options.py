#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr Nov 21 12:15:10 2024

@author: ebernardes
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from pyShapeDetector.utility import DetectorOptions

options = DetectorOptions()


def test_invalid_parameters():
    # This works
    options.threshold_angle_degrees = 10

    with pytest.raises(AttributeError, match="Cannot set undefined parameter"):
        # This doesn't, typo in parameter
        options.threshold_angle_degree = 10


def test_threshold_distance():
    options.threshold_distance = 1
    options.threshold_distance = 0
    options.threshold_distance = None

    with pytest.raises(ValueError, match="must be None or non-negative"):
        options.threshold_distance = -1


def test_threshold_angle():
    options.threshold_angle = 1
    options.threshold_angle = 0

    with pytest.raises(ValueError, match="must be non-negative"):
        options.threshold_angle = -1

    with pytest.raises(ValueError, match="must be non-negative"):
        options.threshold_angle_degrees = -10

    options.threshold_angle_degrees = 10
    assert_almost_equal(options.threshold_angle, np.deg2rad(10))

    options.threshold_angle = np.pi / 2
    assert_almost_equal(options.threshold_angle_degrees, 90)


def test_threshold_ratios():
    options.threshold_ratios = (0, 0.1)
    options.threshold_ratios = (0, 1)

    with pytest.raises(ValueError, match="must be a tuple/list of two values"):
        options.threshold_ratios = (0, 0.2, 3)

    with pytest.raises(ValueError, match="values must be in range"):
        options.threshold_ratios = (0, 2)

    with pytest.raises(ValueError, match="values must be in range"):
        options.threshold_ratios = (-0.1, 0.5)

    with pytest.warns(UserWarning, match="values are almost the same"):
        options.threshold_ratios = (0.2, 0.2)

    options.threshold_ratios = (0.4, 0.2)
    assert options.threshold_ratios == (0.2, 0.4)


def test_refit_ratio():
    options.threshold_refit_ratio = 1
    options.threshold_refit_ratio = 1.5
    with pytest.raises(ValueError, match="should be higher or equal to 1"):
        options.threshold_refit_ratio = 0.9


def test_probability():
    options.probability = 0.1
    options.probability = 0.9
    options.probability = 0

    with pytest.raises(ValueError, match="must be in range"):
        options.probability = -0.1

    with pytest.raises(ValueError, match="must be in range"):
        options.probability = 1.1

    with pytest.raises(ValueError, match="must be in range"):
        options.probability = 1


def test_fitness_min():
    options.fitness_min = 0.1
    options.fitness_min = 0.9
    options.fitness_min = 0
    options.fitness_min = 1
    options.fitness_min = None

    with pytest.raises(ValueError, match="must be None or in range"):
        options.fitness_min = -0.1

    with pytest.raises(ValueError, match="must be None or in range"):
        options.fitness_min = 1.1


def test_connected_components_eps():
    options.connected_components_eps = 1
    options.connected_components_eps = 0
    options.connected_components_eps = None

    with pytest.raises(ValueError, match="must be None or non-negative"):
        options.connected_components_eps = -1


def test_adaptative_threshold_k():
    options.adaptative_threshold_k = 10
    options.adaptative_threshold_k = 100

    with pytest.raises(ValueError, match="must be a positive integer"):
        options.adaptative_threshold_k = 3.9

    with pytest.raises(ValueError, match="must be a positive integer"):
        options.adaptative_threshold_k = 0

    with pytest.raises(ValueError, match="must be a positive integer"):
        options.adaptative_threshold_k = -1


def test_num_samples():
    options.num_samples = 10
    options.num_samples = 100
    options.num_samples = None

    with pytest.raises(ValueError, match="must be None or a positive integer"):
        options.num_samples = 3.9

    with pytest.raises(ValueError, match="must be None or a positive integer"):
        options.num_samples = 0

    with pytest.raises(ValueError, match="must be None or a positive integer"):
        options.num_samples = -1


def test_num_iterations():
    options.num_iterations = 10
    options.num_iterations = 100

    with pytest.raises(ValueError, match="must be a positive integer"):
        options.num_iterations = 3.9

    with pytest.raises(ValueError, match="must be a positive integer"):
        options.num_iterations = 0

    with pytest.raises(ValueError, match="must be a positive integer"):
        options.num_iterations = -1


def test_inliers_min():
    options.inliers_min = 10
    options.inliers_min = 100
    options.num_samples = None

    with pytest.raises(ValueError, match="must be None or a positive integer"):
        options.inliers_min = 3.9

    with pytest.raises(ValueError, match="must be None or a positive integer"):
        options.inliers_min = 0

    with pytest.raises(ValueError, match="must be None or a positive integer"):
        options.inliers_min = -1


def test_downsample():
    options.downsample = 10
    options.downsample = 100

    with pytest.raises(ValueError, match="must be a positive integer"):
        options.downsample = 3.9

    with pytest.raises(ValueError, match="must be a positive integer"):
        options.downsample = 0

    with pytest.raises(ValueError, match="must be a positive integer"):
        options.downsample = -1
