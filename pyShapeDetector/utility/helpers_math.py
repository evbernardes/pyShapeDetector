#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions that act on primitives.

Created on Tue Dec  5 15:48:31 2023

@author: ebernardes
"""
import numpy as np
from scipy.spatial.transform import Rotation
# from open3d.geometry import LineSet, TriangleMesh, PointCloud
# from pyShapeDetector.primitives import Plane, PlaneBounded, Line
# from pyShapeDetector.primitives import Primitive, Line

def get_rotation_from_axis(axis_origin, axis):
    """ Rotation matrix that transforms `axis_origin` in `axis`.
    
    Parameters
    ----------
    axis_origin : 3 x 1 array
        Initial axis.
    axis : 3 x 1 array
        Goal axis.
    
    Returns
    -------
    rotation
        3x3 rotation matrix
    """
    axis = np.array(axis) / np.linalg.norm(axis)
    axis_origin = np.array(axis_origin) / np.linalg.norm(axis_origin)
    if abs(axis.dot(axis_origin) + 1) > 1E-6:
        # axis_origin = -axis_origin
        halfway_axis = (axis_origin + axis)[..., np.newaxis]
        halfway_axis /= np.linalg.norm(halfway_axis)
        return 2 * halfway_axis * halfway_axis.T - np.eye(3)
    else:
        orthogonal_axis = np.cross(np.random.random(3), axis)
        orthogonal_axis /= np.linalg.norm(orthogonal_axis)
        return Rotation.from_quat(list(orthogonal_axis)+[0]).as_matrix()
