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

from pyShapeDetector.utility import _set_and_check_3d_array
from .pointcloud import PointCloud

@link_to_open3d_geometry(open3d_AxisAlignedBoundingBox)
class AxisAlignedBoundingBox(Open3D_Geometry):
    """
    AxisAlignedBoundingBox class that uses Open3D.geometry.AxisAlignedBoundingBox internally.
    
    Almost every method and property are automatically copied and decorated.
    
    Methods
    -------
    contains_points
    intersects
    expanded
    split
    sample_points_uniformly
    sample_points_density
    sample_PointCloud_uniformly
    sample_PointCloud_density
    """

    def contains_points(self, points, inclusive=True):
        """
        Check which points are inside of the bounding box.
    
        Parameters
        ----------
        
        points : N x 3 array
            N input points 
    
        Returns
        -------
        Numpy array
            Boolean values
        """
        points = _set_and_check_3d_array(points, name='points')
        
        if inclusive:
            min_check = np.all(self.min_bound <= points, axis=1)
            max_check = np.all(points <= self.max_bound, axis=1)

        else:
            min_check = np.all(self.min_bound <= points, axis=1)
            max_check = np.all(points <= self.max_bound, axis=1)
            
        return np.logical_and(min_check, max_check)#.tolist()
    
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
    
    def sample_points_uniformly(self, number_of_points=100):
        """ Sample points inside bounding box.
        
        Parameters
        ----------
        number_of_points : int, optional
            Number of points that should be uniformly sampled. Default = 100.
        
        Returns
        -------
        Numpy array with shape (number_of_points, 3)
            Sampled pointcloud from shape.
        """
        if number_of_points <= 0:
            raise ValueError("Number of points must be a non-negative number.")
        return np.random.uniform(size=(number_of_points, 3), low=self.min_bound, high=self.max_bound)
    
    def sample_points_density(self, density=1):
        """ Sample points inside bounding box.
        
        Parameters
        ----------
        density: float, optional
            Ratio between points and surface area. Default: 1.
        
        Returns
        -------
        Numpy array with shape (number_of_points, 3)
        """
        if density <= 0:
            raise ValueError("Density must be a non-negative number.")
            
        # mesh = self.get_mesh()
        number_of_points = int(density * self.volume())
        return self.sample_points_uniformly(number_of_points)
    
    def sample_PointCloud_uniformly(self, number_of_points=100):
        """ Sample points inside bounding box and return PointCloud.
        
        Parameters
        ----------
        number_of_points : int, optional
            Number of points that should be uniformly sampled. Default = 100.
        use_triangle_normal : bool, optional
            If True assigns the triangle normals instead of the interpolated 
            vertex normals to the returned points. The triangle normals will 
            be computed and added to the mesh if necessary. Default = False.
        
        Returns
        -------
        open3d.geometry.PointCloud
            Sampled pointcloud from shape.
        """
        return PointCloud(self.sample_points_uniformly(number_of_points))
    
    def sample_PointCloud_density(self, density=1):
        """ Sample points inside bounding box and return PointCloud.
        
        See: sample_points_uniformly, sample_points_density, 
        sample_PointCloud_uniformly
        
        Parameters
        ----------
        density: float, optional
            Ratio between points and surface area. Default: 1.
        use_triangle_normal : bool, optional
            If True assigns the triangle normals instead of the interpolated 
            vertex normals to the returned points. The triangle normals will 
            be computed and added to the mesh if necessary. Default = False.
        
        Returns
        -------
        open3d.geometry.PointCloud
            Sampled pointcloud from shape.
        """
        return PointCloud(self.sample_points_density(density))
        