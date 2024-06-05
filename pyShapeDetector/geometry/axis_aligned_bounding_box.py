#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:27:03 2024

@author: ebernardes
"""
import copy
import numpy as np
from open3d.geometry import AxisAlignedBoundingBox as open3d_AxisAlignedBoundingBox

from .open3d_geometry import (
    link_to_open3d_geometry,
    Open3D_Geometry)

@link_to_open3d_geometry(open3d_AxisAlignedBoundingBox)
class AxisAlignedBoundingBox(Open3D_Geometry):
    """
    AxisAlignedBoundingBox class that uses Open3D.geometry.AxisAlignedBoundingBox internally.
    
    Almost every method and property are automatically copied and decorated.
    
    Methods
    -------
    split
    """
    
    def split(self, num_boxes, dim=None):
        """ Separates bounding boxes into multiple sub-boxes.

        Parameters
        ----------
        num_boxes : int
            Number of sub-boxes.
        dim : int, optional
            Dimension that should be divided. If not given, will be chosen as the
            largest dimension. Default: None.
            
        Returns
        -------
        list
            Divided boxes
        """
        if not isinstance(num_boxes, int) or num_boxes < 1:
            raise ValueError(f'Number of divisions should be a positive integer, got {num_boxes}.')
            
        if num_boxes == 1:
            return [copy.copy(self)]
            
        if dim is not None and dim not in [0, 1, 2]:
            raise ValueError(f"Dim must be 0, 1 or 2, got {dim}.")
        
        min_bound = self.min_bound
        max_bound = self.max_bound
        delta = (max_bound - min_bound)
        
        if dim is None:
            dim = np.where(delta == max(delta))[0][0]
         
        parallel = np.zeros(3)
        parallel[dim] = delta[dim]
        orthogonal = delta - parallel

        bboxes = []
        for i in range(num_boxes):
            subbox = AxisAlignedBoundingBox(
                min_bound + i * parallel/num_boxes, 
                min_bound + orthogonal + (i+1) * parallel/num_boxes)
            subbox.color = self.color
            bboxes.append(subbox)

        return bboxes
        