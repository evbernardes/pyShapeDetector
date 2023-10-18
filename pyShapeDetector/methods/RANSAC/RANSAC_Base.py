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

class RANSAC_Base(ABC):
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
    """

    def __init__(self,
                 primitive,
                 reduction_rate=1.0,
                 threshold_distance=0.1,
                 threshold_angle=0.174,  # ~ 10 degrees
                 ransac_n=None,
                 num_iterations=100,
                 probability=0.99999,
                 max_point_distance=None,
                 model_max=None,
                 model_min=None,
                 max_normal_angle_degrees=10,
                 inliers_min=None):

        if threshold_angle < 0:
            raise ValueError('threshold_angle must be positive')

        if probability <= 0 or probability > 1:
            raise ValueError('Probability must be > 0 and <= 1.0')

        if ransac_n is None:
            ransac_n = primitive._fit_n_min
        elif ransac_n < primitive._fit_n_min:
            raise ValueError(f'for {primitive._name}s, ransac_n should be at '
                             f'least {primitive._fit_n_min}.')
        
        if model_max is None:
            self.idx_model_max = []
            self.model_max = None
        elif len(model_max) == primitive._model_args_n:
            self.model_max = np.array(model_max)
            self.idx_model_max = np.where(self.model_max != None)[0]
        else:
            raise ValueError(f'for {self._type}s, model_max is either None or a'
                             f' list of size {primitive._model_args_n}, got '
                             f'{model_max}')
        
        if model_min is None:
            self.idx_model_min = []
            self.model_min = None
        elif len(model_min) == primitive._model_args_n:
            self.model_min = np.array(model_min)
            self.idx_model_min = np.where(self.model_min != None)[0]
        else:
            raise ValueError(f'for {self._type}s, model_min is either None or a'
                             f' list of size {primitive._model_args_n}, got '
                             f'{model_min}')

        self.primitive = primitive
        self.reduction_rate = reduction_rate
        self.threshold_distance = threshold_distance
        self.threshold_angle = threshold_angle
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.probability = probability
        self.max_point_distance = max_point_distance
        self.inliers_min = inliers_min

    @property
    @abstractmethod
    def _type(self):
        """ Name of method."""
        pass
    
    @abstractmethod
    def compare_metrics(self, metrics, metrics_best):
        """ Compare metrics to decide if new fit is better than current
        best fit.

        Actual implementation depends on exact type of RANSAC-based method.
        
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
        pass
    
    def termination_criterion(self, metrics):
        """ Gives number of max needed iterations, depending on current metrics.
        
        Parameters
        ----------
        metrics : dict
            Metrics analyzing current fit
        
        Returns
        -------
        int
            Number of max needed iterations
        """
        if metrics['fitness'] > 1.0:
            return 0
        
        den = np.log(1 - metrics['fitness'] ** self.ransac_n)
        if den != 0:
            return min(self.num_iterations, np.log(1 - self.probability) / den)
        else:
            return self.num_iterations
    
    def get_metrics(self, 
                 num_points=None, num_inliers=None, 
                 distances=None, angles=None):
        """ Gives a dictionary with metrics that can be used to compared fits.
        
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
        if num_points is None or num_inliers is None or distances is None:
            metrics = {'num_inliers': 0, 'fitness': 0, 
                    'rmse_distances': 0, 'rmse_angles': 0}
            
        else:
            rmse_distances = np.sqrt(distances.dot(distances)) / len(distances)
            if angles is not None:
                rmse_angles = np.sqrt(angles.dot(angles)) / len(angles)
            else:
                rmse_angles = None
            metrics = {'num_inliers': num_inliers,
                    'fitness': num_inliers / num_points,
                    'rmse_distances': rmse_distances,
                    'rmse_angles': rmse_angles}
            
        metrics['break_iteration'] = self.termination_criterion(metrics)
        return metrics

    def get_model(self, points, normals, samples):
        """ Fits shape, then test if its model parameters respect input
        max and min values. If it does, return shape, otherwise, return None.
        
        Parameters
        ----------
        points : array
            All input points
        normals : array or None
            All input normal vectors 
        samples : array
            Indices of supposed inliers
        
        Returns
        -------
        shape or None
            Fitted shape, if it respects limits
        """

        n = None if normals is None else normals[samples]
        shape = self.primitive.fit(points[samples], n)

        model = np.array(shape.model)
        if self.model_max is not None and np.any(
                self.model_max[self.idx_model_max] < model[self.idx_model_max]):
            return None
        
        if self.model_min is not None and np.any(
                self.model_min[self.idx_model_min] > model[self.idx_model_min]):
            return None
                
        return shape

    # def get_probabilities(self, distances, angles=None):

    #     if angles is not None and len(distances) != len(angles):
    #         raise ValueError('distances and angles must have the same size, '
    #                          f'got {distances.shape} and {angles.shape}.')

    #     probabilities = np.zeros(len(distances))

    #     mask = distances < self.threshold_distance
    #     if angles is not None:
    #         mask *= angles < self.threshold_angle

    #     probabilities[mask] = 1 - distances[mask] / self.threshold_distance

    #     if angles is not None:
    #         probabilities[mask] *= 1 - angles[mask] / self.threshold_angle

    #     return probabilities

    def get_samples(self, points, num_samples):
        """ Sample points and return indices of sampled points.

        If the method's `max_point_distance` attribute is set, then only
        iteratively take samples only accepting at each time one that is 
        close to the others.
        
        Parameters
        ----------
        points : N x 3 array
            All input points
        num_samples : int
            Number of samples
        
        Returns
        -------
        list
            Indices of samples
        """
        if self.max_point_distance is None:
            return random.sample(range(num_samples), self.ransac_n)

        samples = set([random.randrange(num_samples)])
        
        while len(samples) < self.ransac_n:
            
            # Array of shape (num_samples, num_points, 3), where diff[i, :, :]
            # is the different between each point to the ith sample
            diff = points[None] - points[list(samples)][:, None]
            
            # Find all of the distances, then get the minimum for each sample,
            # giving dist.shape == (n_points, )
            dist = np.min(np.linalg.norm(diff, axis=2), axis=0)
            
            # Find all of the possible points and then remove those already
            # sampled
            possible_samples = dist < self.max_point_distance
            possible_samples[list(samples)] = False
            
            # Continue only if enough existing samples
            N_possible = sum(possible_samples)
            if N_possible < self.ransac_n - len(samples):
                return None
            
            idx_possible = np.where(possible_samples)[0]
            idx_sample = random.randrange(N_possible)
            samples.add(idx_possible[idx_sample])

        return list(samples)
    
    def get_inliers_from_residuals(self, distances, angles):
        """ Return indices of inliers: points whose distance to shape and
        angle with normal vector are below the given thresholds.
        
        Parameters
        ----------
        distances : array
            Distances of each point to shape
        angles : array or None
            Angles between each input and theoretical normal vector
        
        Returns
        -------
        list
            Indices of inliers
        """
        is_inlier = distances < self.threshold_distance
        if angles is not None:
            is_inlier *= (angles < self.threshold_angle)
        return np.where(is_inlier)[0]
    
    def get_inliers_from_points(self, shape, points, normals=None):
        """ Return indices of inliers: points whose distance to shape and
        angle with normal vector are below the given thresholds.
        
        Parameters
        ----------
        shape : primitive
            Fitted shape
        points : array
            All input points
        normals, optional : array
            All input normal vectors 
        
        Returns
        -------
        list
            Indices of inliers
        """
        distances, angles = shape.get_residuals(points, normals)
        return self.get_inliers_from_residuals(distances, angles)

    def fit(self, points, normals=None, debug=False):#, filter_model=True):
        """ Main loop implementing RANSAC algorithm.
        
        Parameters
        ----------
        points : array
            All input points
        normals, optional : array
            All input normal vectors 
        debug, optional : bool
            Gives info if true
        
        Returns
        -------
        primitive
            Best fitted primitive
        list
            List of inliers indices of fitted shape
        dict
            Metrics of best fitted shape
        """
        primitive = self.primitive
        points = np.asarray(points)
        num_points = len(points)
        inliers_min = self.inliers_min

        if num_points < self.ransac_n:
            raise ValueError(f'Pointcloud must have at least {self.ransac_n} '
                             f'points, {num_points} given.')

        if normals is not None:
            if len(normals) != num_points:
                raise ValueError('Numbers of points and normals must be equal')
            normals = np.asarray(normals)

        if inliers_min and num_points < inliers_min:
            if debug:
                print('Remaining points less than inliers_min, stopping')
            return None, None, self.get_metrics(None)

        # metrics dict stores metricsrmation used to decide which model is best
        metrics_best = self.get_metrics(None)
        
        shape_best = None
        iteration_count = 0

        times = {'get_inliers_and_error': 0,
                 'get_model': 0}
        
        if debug:
            print(f'Starting loop, fitting {primitive.name}...')
        
        for itr in range(self.num_iterations):

            if (iteration_count > metrics_best['break_iteration']):
                
                # if debug:
                    # print('Breaking iteration.')
                continue

            start_itr = time.time()

            samples = self.get_samples(points, num_points)
            if samples is None:
                # if debug:
                    # print('Sampling not possible.')
                continue
            
            t_ = time.time()
            shape = self.get_model(points, normals, samples)
            # print(shape)
            times['get_model'] += time.time() - t_

            if shape is None:
                # if debug:
                    # print('Model not found.')
                continue
            # elif debug:
                # print(f'Fitted model = {shape.model}')

            t_ = time.time()
            distances, angles = shape.get_residuals(points, normals)
            inliers = self.get_inliers_from_residuals(distances, angles)
            num_inliers = len(inliers)
            
            times['get_inliers_and_error'] += time.time() - t_

            if num_inliers == 0 or (inliers_min and num_inliers < inliers_min):
                
                # if debug:
                    # print('No inliers.')
                continue
            
            metrics = self.get_metrics(num_points, num_inliers, 
                                       distances, angles)

            if self.compare_metrics(metrics, metrics_best):
                metrics_best = metrics
                shape_best = shape

            iteration_count += 1

            if debug:
                print(f'Iteration {itr+1}/{self.num_iterations} : '
                      f'{time.time() - start_itr:.5f}s')
        
        if shape_best is None:
            return None, None, self.get_metrics(None)
        
        # Find the final inliers using model_best ...
        distances, angles = shape_best.get_residuals(points, normals)
        inliers_final = self.get_inliers_from_residuals(distances, angles)
        num_inliers = len(inliers_final)
        metrics_final = self.get_metrics(num_points, num_inliers, 
                                         distances, angles)

        # ... and then find the final model using the final inliers
        # if filter_model:
        # n = None if normals is None else normals[inliers_final]
        # shape_best = primitive.fit(points[inliers_final], n)
        shape = self.get_model(points, normals, inliers_final)
        if shape:
            shape_best = shape
        
        if shape_best is None:
            raise ValueError('None value found for shape at the last '
                             'filtering step, this should not happen')
            
        if debug:
            print(f'\nFinished fitting {primitive.name}!')
            print(f'model: {shape_best.model}')
            print('Execution time:')
            for t_ in times:
                print(f'{t_} : {times[t_]:.5f}s')
            print(f'{num_points} points and {num_inliers} inliers, '
                  f'{int(100*metrics_final["fitness"])}% fitness')
            print(f'RMSE for distances: {metrics_final["rmse_distances"]}')
            print(f'RMSE for angles: {metrics_final["rmse_angles"]}\n')

        return shape_best, inliers_final, metrics_final
