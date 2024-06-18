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

# random.seed(951)

from .RANSAC_Base import RANSAC_Base


class MSAC(RANSAC_Base):
    """
    MSAC weighted RANSAC.

    Attributes
    ----------
    _type
    num_samples
    max_weight
    primitives
    limits
    num_primitives
    options

    Methods
    -------
    __init__
    __repr__
    add
    remove
    remove_all
    copy_options
    weight_distances
    weight_angles
    get_total_weight
    compare_metrics
    termination_criterion
    get_metrics
    get_model
    get_samples
    get_biggest_connected_component
    get_inliers
    fit
    """

    _type = "MSAC"

    def weight_distances(self, distances, distance_threshold):
        """Gives weights for each point based on the distances.

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
        threshold = self._opt.reduction_rate * distance_threshold

        weight = np.zeros(len(distances))
        idx = distances > threshold
        weight[idx] = 1 - (distances[idx] / threshold) ** 2

        return weight
