#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:59:55 2023

@author: ebernardes
"""
import warnings
import numpy as np


class DetectorOptions:
    """
    Base class used to define options for detector methods.

    """

    _reduction_rate = 1.0
    _threshold_distance = None
    # _max_point_distance = None
    _threshold_angle = 3.141592653589793  # ~ 180 degrees
    # _max_normal_angle_degrees = 10
    _threshold_ratios = [0.2, 0.7]  # for LDSAC
    _threshold_refit_ratio = 1
    _probability = 0.99999
    _fitness_min = None
    _connected_components_eps = None
    _adaptative_threshold_k = 15
    _num_samples = None
    _num_iterations = 100
    _inliers_min = None
    _downsample = 1

    _valid_parameters = [
        "_reduction_rate",
        "reduction_rate",
        "_threshold_distance",
        "threshold_distance",
        # "_max_point_distance",
        # "max_point_distance",
        "_threshold_angle",
        "threshold_angle",
        # no "_threshold_angle_degrees", it's an internal call to "_threshold_angle"
        "threshold_angle_degrees",
        # "_max_normal_angle_degrees",
        # "max_normal_angle_degrees",
        "_threshold_ratios",
        "threshold_ratios",
        "_threshold_refit_ratio",
        "threshold_refit_ratio",
        "_probability",
        "probability",
        "_fitness_min",
        "fitness_min",
        "_connected_components_eps",
        "connected_components_eps",
        "_adaptative_threshold_k",
        "adaptative_threshold_k",
        "_num_samples",
        "num_samples",
        "_num_iterations",
        "num_iterations",
        "_inliers_min",
        "inliers_min",
        "_downsample",
        "downsample",
    ]

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
            name: self.__getattribute__(name)
            for name in self._valid_parameters
            if name[0] != "_"
        }

    def __repr__(self):
        lines = "\n".join("{!r}: {!r},".format(k, v) for k, v in self.dict.items())
        dict_str = "{\n" + lines + "}"
        return type(self).__name__ + "(" + dict_str + ")"

    def __init__(self, dict_parameters={}):
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
        if value is not None and value < 0:
            raise ValueError("threshold_distance must be None or non-negative.")
        self._threshold_distance = value

    # @property
    # def max_point_distance(self):
    #     return self._max_point_distance

    # @max_point_distance.setter
    # def max_point_distance(self, value):
    #     self._max_point_distance = value

    @property
    def threshold_angle(self):
        return self._threshold_angle

    @threshold_angle.setter
    def threshold_angle(self, value):
        if value < 0:
            raise ValueError("threshold_angle must be non-negative.")
        self._threshold_angle = value

    # @property
    # def max_normal_angle_degrees(self):
    #     return self._max_normal_angle_degrees

    # @max_normal_angle_degrees.setter
    # def max_normal_angle_degrees(self, value):
    #     self._max_normal_angle_degrees = value

    @property
    def threshold_angle_degrees(self):
        return np.rad2deg(self.threshold_angle)

    @threshold_angle_degrees.setter
    def threshold_angle_degrees(self, value):
        self.threshold_angle = np.deg2rad(value)

    @property
    def threshold_ratios(self):
        return self._threshold_ratios

    @threshold_ratios.setter
    def threshold_ratios(self, values):
        if len(values) != 2:
            raise ValueError(
                f"threshold_ratios must be a tuple/list of two values, got {values}."
            )

        if not (0 <= values[0] <= 1) or not (0 <= values[1] <= 1):
            raise ValueError(
                "threshold_ratios values must be in range [0, 1], got {values}."
            )

        if abs(values[1] - values[0]) < 1e-7:
            warnings.warn(
                "Input threshold_ratios values are almost the same, "
                "should be a range to work properly."
            )

        self._threshold_ratios = (min(values), max(values))

    @property
    def threshold_refit_ratio(self):
        return self._threshold_refit_ratio

    @threshold_refit_ratio.setter
    def threshold_refit_ratio(self, value):
        if value < 1:
            raise ValueError("threshold_refit_ratio should be higher or equal to 1.")
        self._threshold_refit_ratio = value

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, value):
        if value < 0 or value >= 1:
            raise ValueError(f"probability must be in range [0, 1), got {value}")
        self._probability = value

    @property
    def fitness_min(self):
        return self._fitness_min

    @fitness_min.setter
    def fitness_min(self, value):
        if value is not None and (value < 0 or value > 1):
            raise ValueError(
                f"fitness_min must be None or in range [0, 1], got {value}"
            )
        self._fitness_min = value

    @property
    def connected_components_eps(self):
        return self._connected_components_eps

    @connected_components_eps.setter
    def connected_components_eps(self, value):
        if value is not None and value < 0:
            raise ValueError("connected_components_eps must be None or non-negative.")
        self._connected_components_eps = value

    @property
    def adaptative_threshold_k(self):
        return self._adaptative_threshold_k

    @adaptative_threshold_k.setter
    def adaptative_threshold_k(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                "adaptative_threshold_k must be a positive integer, got {value}."
            )
        self._adaptative_threshold_k = value

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value):
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError(
                f"num_samples must be None or a positive integer, got {value}."
            )
        self._num_samples = value

    @property
    def num_iterations(self):
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError("num_interations must be a positive integer, got {value}.")
        self._num_iterations = value

    @property
    def inliers_min(self):
        return self._inliers_min

    @inliers_min.setter
    def inliers_min(self, value):
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError(
                f"inliers_min must be None or a positive integer, got {value}."
            )
        self._inliers_min = value

    @property
    def downsample(self):
        return self._downsample

    @downsample.setter
    def downsample(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError("downsample must be a positive integer, got {value}.")
        self._downsample = value

    # Only allows setting valid arguments
    def __setattr__(self, name, value):
        if name in self._valid_parameters:
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot set undefined parameter '{name}'.")
