#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:27:03 2024

@author: ebernardes
"""
from open3d.geometry import OrientedBoundingBox as open3d_OrientedBoundingBox

from .open3d_geometry import link_to_open3d_geometry, Open3D_Geometry


@link_to_open3d_geometry(open3d_OrientedBoundingBox)
class OrientedBoundingBox(Open3D_Geometry):
    """
    OrientedBoundingBox class that uses Open3D.geometry.OrientedBoundingBox internally.

    Almost every method and property are automatically copied and decorated.

    Methods
    -------
    expanded

    """

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
        slack = abs(slack)
        return OrientedBoundingBox(
            center=self.center, R=self.R, extent=self.extent + slack
        )
