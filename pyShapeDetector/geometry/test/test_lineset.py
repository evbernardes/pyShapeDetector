#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  11 13:52:35 2024

@author: ebernardes
"""

import pytest
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal

from pyShapeDetector.primitives import Line
from pyShapeDetector.geometry import LineSet


def test_line_lineset_conversion():
    N_points = 30
    N_lines = 15

    lineset = LineSet()

    lineset.points = np.random.random((30, 3))
    lineset.lines = np.random.randint(0, N_points, [N_lines, 2])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*Line instance has very small length.*"
        )
        lines = lineset.as_list_of_lines

    assert len(lines) == N_lines

    lineset_new = Line.get_LineSet_from_list(lines)
    assert_array_almost_equal(
        lineset.points[lineset.lines], lineset_new.points[lineset_new.lines]
    )

    lineset_new = LineSet.from_lines(lines)
    assert_array_almost_equal(
        lineset.points[lineset.lines], lineset_new.points[lineset_new.lines]
    )
