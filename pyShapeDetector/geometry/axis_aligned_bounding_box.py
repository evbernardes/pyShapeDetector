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
    check_bbox_intersection
    intersects
    expanded
    split
    """
    
    def intersects(self, other_bbox, distance=0):
        """ Check if minimal distance of the inlier points bounding box
        is below a given distance.
        
        Parameters
        ----------
        other_bbox : Primitive
            A shape with inlier points
        distance : float
            Max distance between the bounding boxes.
            
        Returns
        -------
        bool
            True if the calculated distance is smaller than the input distance.
        """
        if not isinstance(other_bbox, AxisAlignedBoundingBox):
            raise ValueError('Input should be another instance of AxisAlignedBoundingBox.')
        
        if distance is None:
            return True
        
        if distance < 0:
            raise ValueError("Distance must be non-negative.")
            
        bb1 = self
        bb2 = other_bbox
        if distance == 0:
            bb1 = bb1.expanded(distance/2)
            bb2 = bb2.expanded(distance/2)
        
        test_order = bb2.max_bound - bb1.min_bound >= 0
        if test_order.all():
            pass
        elif (~test_order).all():
            bb1, bb2 = bb2, bb1
        else:
            return False
        
        # test_intersect = (bb1.max_bound + atol) - (bb2.min_bound - atol) >= 0
        test_intersect = bb1.max_bound - bb2.min_bound >= 0
        return test_intersect.all()
    
    def expanded(self, slack=0):
        """ Return expanded version with bounds expanded in all directions.
        
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
        return AxisAlignedBoundingBox(self.min_bound - slack, 
                                      self.max_bound + slack)
    
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
        