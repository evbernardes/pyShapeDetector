#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:02:34 2023

@author: ebernardes
"""

from .RANSAC import (RANSAC_Classic, MSAC, LDSAC, BDSAC
                     )

list_methods_RANSAC=[
    RANSAC_Classic, MSAC, LDSAC, BDSAC
    ]