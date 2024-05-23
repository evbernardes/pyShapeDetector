#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:32:38 2024

@author: ebernardes
"""
from open3d.geometry import TriangleMesh as open3d_TriangleMesh

from .open3d_geometry import (
    link_to_open3d_geometry,
    Open3D_Geometry)

@link_to_open3d_geometry(open3d_TriangleMesh)
class TriangleMesh(Open3D_Geometry):
    pass
    # @to_open3d_and_back
    # def __init__(self, *args):
    #     self._open3d = open3d_TriangleMesh(*args)