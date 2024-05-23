#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:26:20 2024

@author: ebernardes
"""
from open3d.geometry import PointCloud as open3d_PointCloud

from .open3d_geometry import (
    link_to_open3d_geometry,
    Open3D_Geometry)

@link_to_open3d_geometry(open3d_PointCloud)
class PointCloud(Open3D_Geometry):
    pass