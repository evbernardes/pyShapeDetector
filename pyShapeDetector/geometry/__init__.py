#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:32:03 2024

This sub-module contains heavily decorated and modified versions of the classes
in Open3D.geometry, so that they can work directly with Numpy arrays.

@author: ebernardes
"""
from .numpy_geometry import Numpy_Geometry
from .pointcloud import PointCloud
from .trianglemesh import TriangleMesh
from .axis_aligned_bounding_box import AxisAlignedBoundingBox
from .oriented_bounding_box import OrientedBoundingBox
from .lineset import LineSet
from .equivalent_classes import equivalent_classes_dict
