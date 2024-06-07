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

REF_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])

RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]])

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
    
def rgb_to_cielab(rgb, already_normalized=True):

    if not already_normalized:
        # Normalize RGB values to [0, 1]
        rgb = rgb / 255.0

    # Apply a gamma correction to linearize sRGB values
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    # Convert to XYZ using the sRGB color space conversion matrix    
    xyz = np.dot(rgb, RGB_TO_XYZ.T) / REF_WHITE_D65

    # Apply the Lab conversion formula
    mask = xyz > 0.008856
    xyz[mask] = xyz[mask] ** (1/3)
    xyz[~mask] = (7.787 * xyz[~mask]) + (16 / 116)

    l = (116 * xyz[:, 1]) - 16
    a = 500 * (xyz[:, 0] - xyz[:, 1])
    b = 200 * (xyz[:, 1] - xyz[:, 2])

    lab = np.stack([l, a, b], axis=1)
    
    return lab

def cielab_to_rgb(lab):

    l, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    fy = (l + 16) / 116
    fx = fy + (a / 500)
    fz = fy - (b / 200)

    # Apply the inverse Lab conversion formula
    fx3 = fx ** 3
    fy3 = fy ** 3
    fz3 = fz ** 3

    mask_fx = fx3 > 0.008856
    mask_fy = fy3 > 0.008856
    mask_fz = fz3 > 0.008856

    fx[mask_fx] = fx3[mask_fx]
    fx[~mask_fx] = (fx[~mask_fx] - 16 / 116) / 7.787

    fy[mask_fy] = fy3[mask_fy]
    fy[~mask_fy] = (fy[~mask_fy] - 16 / 116) / 7.787

    fz[mask_fz] = fz3[mask_fz]
    fz[~mask_fz] = (fz[~mask_fz] - 16 / 116) / 7.787

    x = fx * REF_WHITE_D65[0]
    y = fy * REF_WHITE_D65[1]
    z = fz * REF_WHITE_D65[2]

    xyz = np.stack([x, y, z], axis=1)

    # Convert to RGB using the inverse sRGB color space conversion matrix
    matrix = np.linalg.inv(RGB_TO_XYZ)
    rgb = np.dot(xyz, matrix.T)

    # Apply the inverse gamma correction
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1 / 2.4)) - 0.055
    rgb[~mask] = 12.92 * rgb[~mask]

    # Clip values to [0, 1]
    rgb = np.clip(rgb, 0, 1)

    # Convert to [0, 255]
    # rgb = (rgb * 255).astype(np.uint8)
    return rgb

