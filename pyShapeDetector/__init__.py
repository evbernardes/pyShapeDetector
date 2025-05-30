#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:02:34 2023

@author: ebernardes
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyShapeDetector")
except PackageNotFoundError:
    __version__ = "unknown"
