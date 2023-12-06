#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:59:55 2023

@author: ebernardes
"""
import warnings
import time
import random
# import numpy as np

# from pyShapeDetector.utility import PrimitiveLimits

# random.seed(time.time())

class RANSAC_Options():
    """
    Base class used to define RANSAC-based methods.
    
    To define a primitive, inherit from this class and define at least the 
    following attribute:
        `_type`
    And the following method:
        `compare_metrics`

    Attributes
    ----------
    _type : str
        Name of method.
    
    Methods
    -------

    """
    _reduction_rate=1.0
    _threshold_distance=0.1
    _threshold_angle=3.141592653589793  # ~ 180 degrees
    _threshold_ratios=[0.2, 0.7] # for LDSAC
    _threshold_refit_ratio=1
    _num_samples=None
    _num_iterations=100
    _probability=0.99999
    _max_point_distance=None
    _max_normal_angle_degrees=10
    _inliers_min=None
    _fitness_min=None
    _eps=None
    
    # @property
    # def properties(self):
    # # class_items = self.__class__.__dict__.iteritems()  # Python 2
    #     class_items = self.__class__.__dict__.items()
    #     return dict((k, getattr(self, k)) 
    #                 for k, v in class_items 
    #                 if isinstance(v, property))
    
    @property
    def dict(self):
        return {
            'reduction_rate': self.reduction_rate,
            'threshold_distance': self.threshold_distance,
            'threshold_angle': self.threshold_angle,
            'threshold_ratios': self.threshold_ratios,
            'threshold_refit_ratio': self.threshold_refit_ratio,
            'num_samples': self.num_samples,
            'num_iterations': self.num_iterations,
            'probability': self.probability,
            'max_point_distance': self.max_point_distance,
            'max_normal_angle_degrees': self.max_normal_angle_degrees,
            'inliers_min': self.inliers_min,
            'fitness_min': self.fitness_min,
            'connected_components_density': self.connected_components_density
            }
    
    def __repr__(self):
        lines = "\n".join("{!r}: {!r},".format(k, v) for k, v in self.dict.items())
        dict_str = "{\n" + lines + "}"
        return type(self).__name__+'('+dict_str+')'
    
    def __init__(self, dict_parameters={}):
        # self = RANSAC_Options()
        for key in dict_parameters.keys():
            if not hasattr(self, key):
                warnings.warn(f"Ignoring unknown attribute '{key}'")
            else:
                setattr(self, key, dict_parameters[key])
    
    @property
    def reduction_rate(self):
        return self._reduction_rate
    
    @reduction_rate.setter
    def reduction_rate(self, value):
        self._reduction_rate = value
        
    @property
    def threshold_distance(self):
        return self._threshold_distance
        
    @threshold_distance.setter
    def threshold_distance(self, value):
        if value < 0:
            raise ValueError('threshold_distance must be positive')
        self._threshold_distance = value
        
    @property
    def threshold_angle(self):
        return self._threshold_angle
    
    @threshold_angle.setter
    def threshold_angle(self, value):
        if value < 0:
            raise ValueError('threshold_angle must be positive')
        self._threshold_angle = value
        
    @property
    def threshold_ratios(self):
        return self._threshold_ratios
    
    @threshold_ratios.setter
    def threshold_ratios(self, values):
        if len(values) != 2:
            raise ValueError('threshold_ratios must be a tuple/list of two '
                             'values.')
        if not (0 <= values[0] <= 1) or not (0 <= values[1] <= 1):
            raise ValueError('threshold_ratios must be a tuple of two '
                             'values between 0 and 1, got {values}')
        values.sort()
        self._threshold_ratios = values
        
    @property
    def threshold_refit_ratio(self):
        return self._threshold_refit_ratio
    
    @threshold_refit_ratio.setter
    def threshold_refit_ratio(self, value):
        if value < 1:
            raise ValueError('threshold_refit_ratio should be higher or equal '
                             'to 1.')
        self._threshold_refit_ratio = value
        
    @property
    def num_samples(self):
        return self._num_samples
    
    @num_samples.setter
    def num_samples(self, value):
        self._num_samples = value
        
    @property
    def num_iterations(self):
        return self._num_iterations
    
    @num_iterations.setter
    def num_iterations(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError('num_interations must be a positive integer')
        self._num_iterations = value
        
    @property
    def probability(self):
        return self._probability
    
    @probability.setter
    def probability(self, value):
        if value <= 0 or value > 1:
            raise ValueError('probability must be > 0 and <= 1.0')
        self._probability = value
        
    @property
    def max_point_distance(self):
        return self._max_point_distance
        
    @max_point_distance.setter
    def max_point_distance(self, value):
        self._max_point_distance = value
        
    @property
    def max_normal_angle_degrees(self):
        return self._max_normal_angle_degrees
        
    @max_normal_angle_degrees.setter
    def max_normal_angle_degrees(self, value):
        self._max_normal_angle_degrees = value
        
    @property
    def inliers_min(self):
        return self._inliers_min
        
    @inliers_min.setter
    def inliers_min(self, value):
        self._inliers_min = value
        
    @property
    def fitness_min(self):
        return self._fitness_min
    
    @fitness_min.setter
    def fitness_min(self, value):
        if value and (value < 0 or value > 1):
            raise ValueError('fitness_min must be number between 0 and 1, '
                             f'got {value}')
        self._fitness_min = value
        
    @property
    def connected_components_density(self):
        return self._eps
        
    @connected_components_density.setter
    def connected_components_density(self, value):
        self._eps = value
