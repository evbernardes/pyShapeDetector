#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:59:55 2023

@author: ebernardes
"""

from abc import ABC, abstractmethod
import time
import random
import numpy as np

random.seed(951)

from .RANSAC_Base import RANSAC_Base

class RANSAC_WeightedBase(RANSAC_Base):
    """
    Base class used to define weighted RANSAC-based methods.
    
    To define a primitive, inherit from this class and define at least the 
    following method:
        `compare_metrics`
        `weight_distances`

    The method `weight_angles` can also be implemented. Otherwise, considers
    equal to `weight_distances`.
    
    Attributes
    ----------
    _type : str
        Name of method.
    
    Methods
    -------
    compare_metrics(metrics, metrics_best):
        Gives the absolute value of cosines of the angles between the input 
        normal vectors and the calculated normal vectors from the input points.

    termination_criterion(metrics):
        Gives number of max needed iterations, depending on current metrics.
        
    get_metrics(num_points=None, num_inliers=None, distances=None, 
    angles=None):
        Gives a dictionary with metrics that can be used to compared fits.
        
    get_model(points, normals, samples):
        Fits shape, then test if its model parameters respect input
        max and min values. If it does, return shape, otherwise, return None.
        
    get_samples(points, num_samples, tries_max=5000):
        Sample points and return indices of sampled points.

    get_inliers_from_residuals(distances, angles):
        Return indices of inliers: points whose distance to shape and
        angle with normal vector are below the given thresholds.
        
    fit(points, normals=None, debug=False):#, filter_model=True):
        Main loop implementing RANSAC algorithm.
               
    get_residuals(points, normals):
        Convenience function returning both distances and angles.

    weight_distances(distances, threshold_distances):
        Gives weights for each point based on the distances.

    weight_angles(angles, threshold_angles):
        Gives weights for each point based on the distances.

    get_total_weight(distances, angles):
        Total weight of shape.
    """
    
    @abstractmethod
    def weight_distances(self, distances, threshold_distances):
        """ Gives weights for each point based on the distances.

        Actual implementation depends on method.
        
        Parameters
        ----------
        distances : array
            Distances of each point to shape
        threshold_distances : float
            Max distance accepted between points and shape
        
        Returns
        -------
        array
            Weights of each point
        """
        pass

    def weight_angles(self, angles, threshold_angles):
        """ Gives weights for each point based on the distances.

        Actual implementation depends on method, but usually considered
        the same as `weight_distances`.
        
        Parameters
        ----------
        angles : array or None
            Angles between each input and theoretical normal vector
        threshold_angles : float
            Max angle accepted between point normals and shape
        
        Returns
        -------
        array
            Weights of each normal vector
        """
        return self.weight_distances(angles, threshold_angles)
    
    def get_total_weight(self, distances, angles):
        """ Total weight of shape.
        
        Parameters
        ----------
        distances : array
            Distances of each point to shape
        angles : array or None
            Angles between each input and theoretical normal vector
        
        Returns
        -------
        float
            Total weight
        """
        if distances is None:
            weight_distance = 1
        else:
            weight_distance = sum(
                self.weight_distances(distances, self.threshold_distance))
            
        if angles is None:
            weight_angle = 1
        else:
            weight_angle = sum(
                self.weight_angles(angles, self.threshold_angle))
            
        return weight_distance * weight_angle
    
    def compare_metrics(self, metrics, metrics_best):
        """ Compare metrics to decide if new fit is better than current
        best fit.

        For weighted RANSAC methods, choose the shape with higher weight.
        
        Parameters
        ----------
        metrics : dict
            Metrics analyzing current fit
        metrics_best : dict
            Metrics analyzing current best fit
        
        Returns
        -------
        bool
            True if `metrics` is considered better than `metrics_best`
        """
        return metrics['weight'] > metrics_best['weight']
    
    def get_metrics(self, 
                 num_points=None, num_inliers=None, 
                 distances=None, angles=None):
        """ Gives a dictionary with metrics that can be used to compared fits.

        For weighted methods, also includes total weight.
        
        Parameters
        ----------
        num_points : int or None
            Total number of points
        num_inliers : dict
            Number of inliers
        distances : array
            Distances of each point to shape
        angles : array or None
            Angles between each input and theoretical normal vector
        
        Returns
        -------
        dict
            Metrics analyzing current fit
        """
        metrics = super().get_metrics(num_points, num_inliers, distances, angles)
        if num_points is None or num_inliers is None or distances is None:
            metrics['weight'] = 0
        else:
            metrics['weight'] = self.get_total_weight(distances, angles)
        return metrics
