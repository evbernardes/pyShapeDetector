#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:27:03 2024

@author: ebernardes
"""
import copy
import numpy as np
from open3d.geometry import OrientedBoundingBox as open3d_OrientedBoundingBox

from pyShapeDetector.utility import _set_and_check_3d_array
from .open3d_geometry import link_to_open3d_geometry, Open3D_Geometry
from .axis_aligned_bounding_box import AxisAlignedBoundingBox


@link_to_open3d_geometry(open3d_OrientedBoundingBox)
class OrientedBoundingBox(Open3D_Geometry):
    """
    OrientedBoundingBox class that uses Open3D.geometry.OrientedBoundingBox internally.

    Almost every method and property are automatically copied and decorated.

    Methods
    -------
    contains_points
    expanded

    """

    def contains_points(self, points, inclusive=True, eps=1e-5):
        """
        Check which points are inside of the bounding box.

        Parameters
        ----------

        points : N x 3 array
            N input points

        Returns
        -------
        Numpy array
            Boolean values
        """
        points = _set_and_check_3d_array(points, name="points")
        points = (points - self.center) @ self.R
        extent = self.extent / 2 + eps
        aabb = AxisAlignedBoundingBox(-extent, +extent)

        return aabb.contains_points(points, inclusive=inclusive)

    def expanded(self, slack=0):
        """Return expanded version with bounds expanded in all directions.

        Parameters
        ----------
        slack : float, optional
            Expand bounding box in all directions, useful for testing purposes.
            Default: 0.

        Returns
        -------
        AxisAlignedBoundingBox
        """
        obb = OrientedBoundingBox(
            center=self.center, R=self.R, extent=self.extent + slack
        )
        obb.color = self.color
        return obb
