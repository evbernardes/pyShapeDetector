#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:53:00 2024

@author: ebernardes
"""
import numpy as np
from open3d.geometry import LineSet as open3d_LineSet

from .numpy_geometry import link_to_open3d_geometry, Numpy_Geometry


@link_to_open3d_geometry(open3d_LineSet)
class LineSet(Numpy_Geometry):
    """
    LineSet class that uses Open3D.geometry.LineSet internally.

    Attributes
    ----------
    as_list_of_lines

    """

    @property
    def as_list_of_lines(self):
        from pyShapeDetector.primitives import Line

        return [Line.from_two_points(p1, p2) for (p1, p2) in self.points[self.lines]]

    @staticmethod
    def from_lines(lines):
        from pyShapeDetector.primitives import Line

        if isinstance(lines, Line):
            lines = [lines]
        elif isinstance(lines, list):
            pass
        else:
            raise ValueError(
                "Expected Line or list of Line instances, got {type(lines)}."
            )

        points = []
        lines_indices = []
        N = 0
        for line in lines:
            points.append(line.beginning)
            points.append(line.ending)
            lines_indices.append([N, N + 1])
            N += 2

        lineset = LineSet()
        lineset.points = points
        lineset.lines = lines_indices
        lineset.colors = [line.color for line in lines]
        return lineset
