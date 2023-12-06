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
import copy

from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector

from pyShapeDetector.primitives import Primitive
from pyShapeDetector.utility import PrimitiveLimits, RANSAC_Options

# random.seed(time.time())
random.seed(931)

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
        
    weight_distances(distances, threshold_distances):
        Gives weights for each point based on the distances.

    weight_angles(angles, threshold_angles):
        Gives weights for each point based on the distances.

    get_total_weight(distances, angles):
        Total weight of shape.
        
    get_metrics(num_points=None, num_inliers=None, distances=None, 
    angles=None):
        Gives a dictionary with metrics that can be used to compared fits.
        
    get_model(points, normals, samples):
        Fits shape, then test if its model parameters respect input
        max and min values. If it does, return shape, otherwise, return None.
        
    get_samples(points), num_samples:
        Sample points and return indices of sampled points.

    get_inliers(distances, angles, distances, angles):
        Return indices of inliers: points whose distance to shape and
        angle with normal vector are below the given thresholds.
        
    fit(points, normals=None, debug=False):#, filter_model=True):
        Main loop implementing RANSAC algorithm.
               
    get_residuals(points, normals):
        Convenience function returning both distances and angles.
        
    copy_options(value):
        Copies options instead of pointing to the same options.
        
    """


    def __init__(self, options=RANSAC_Options()):
        
        self._opt = options
        self._num_primitives = 0
        self._primitives = []
        self._limits = []
    
    @property
    def num_samples(self):
        if self._opt._num_samples is None:
            return max([p._fit_n_min for p in self.primitives])
        else:
            return self._opt._num_samples
        
    @property
    def max_weight(self):
        return self.weight_distances(np.zeros(1), 100)[0]
    
    def add(self, primitive, limit=None):
        
        limit = PrimitiveLimits(limit)
        
        if self._opt._num_samples and primitive.fit_n_min:
            raise ValueError(f'{primitive.name}s need a minimum of '
                             f'{primitive.fit_n_min} for fitting, but current'
                             f'num_samples option is set to {self._opt._num_samples}'
                             '. Either raise the value on the options, or set '
                             'it to None.')
            
        limit.check_compatibility(primitive)
        self._primitives.append(primitive)
        self._limits.append(limit)
        self._num_primitives += 1
        
    def remove(self, idx):
        if idx < 0 or idx > self._num_primitives - 1:
            raise ValueError('invalid index')
        _ = self._primitives.pop(idx)
        _ = self._limits.pop(idx)
        self._num_primitives -= 1
        
    @property
    def primitives(self):
        return self._primitives
    
    @primitives.setter
    def primitives(self, value):
        raise RuntimeError('primitives, limits and num_primitives should be '
                           'updated together with add and remove functions')
        
    @property
    def limits(self):
        return self._limits
    
    @limits.setter
    def limits(self, value):
        raise RuntimeError('primitives, limits and num_primitives should be '
                           'updated together with add and remove functions')
                           
    @property
    def num_primitives(self):
        return self._num_primitives
    
    @num_primitives.setter
    def num_primitives(self, value):
        raise RuntimeError('primitives, limits and num_primitives should be '
                           'updated together with add and remove functions')
        
    @property
    def options(self):
        return self._opt
    
    @options.setter
    def options(self, value):
        if not isinstance(value, RANSAC_Options):
            raise ValueError('must be an instance of RANSAC_Options')
        # self._opt = copy.copy(value)
        self._opt = value
    
    def copy_options(self, value):
        """ Copies options instead of pointing to the same options.

        Actual implementation depends on method.
        
        Parameters
        ----------
        value : RANSAC method or RANSAC_Options instance
            Options to copy
        """
        if isinstance(value, RANSAC_Base):
            value = value.options
        elif not isinstance(value, RANSAC_Options):
            raise ValueError('input must be an instance of RANSAC_Options '
                             'or of a RANSAC method.')
        self.options = copy.copy(value)
        # self._opt = value
            
    @property
    @abstractmethod
    def _type(self):
        """ Name of method."""
        pass
    
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
                self.weight_distances(distances, self._opt.threshold_distance))
            
        if angles is None:
            weight_angle = 1
        else:
            weight_angle = sum(
                self.weight_angles(angles, self._opt.threshold_angle))
            
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
    
    # @abstractmethod
    # def compare_metrics(self, metrics, metrics_best):
    #     """ Compare metrics to decide if new fit is better than current
    #     best fit.

    #     Actual implementation depends on exact type of RANSAC-based method.
        
    #     Parameters
    #     ----------
    #     metrics : dict
    #         Metrics analyzing current fit
    #     metrics_best : dict
    #         Metrics analyzing current best fit
        
    #     Returns
    #     -------
    #     bool
    #         True if `metrics` is considered better than `metrics_best`
    #     """
    #     pass
    
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
        
        den = np.log(1 - metrics['fitness'] ** self.num_samples)
        if den != 0:
            return min(self._opt.num_iterations, 
                       np.log(1 - self._opt.probability) / den)
        else:
            return self._opt.num_iterations
    
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
                    'rmse_distances': 0, 'rmse_angles': 0, 
                    'weight': 0, 'weight_ratio': 0}
            
        else:
            rmse_distances = np.sqrt(distances.dot(distances)) / len(distances)
            if angles is not None:
                rmse_angles = np.sqrt(angles.dot(angles)) / len(angles)
            else:
                rmse_angles = None
                
            weight_norm = (self.max_weight ** 2) * (num_points ** 2)
            metrics = {
                'num_inliers': num_inliers,
                'fitness': num_inliers / num_points,
                'rmse_distances': rmse_distances,
                'rmse_angles': rmse_angles,
                'weight': (weight:= self.get_total_weight(distances, angles)),
                'weight_ratio': weight / weight_norm
                }
            
        metrics['break_iteration'] = self.termination_criterion(metrics)
        return metrics

    def get_model(self, idx, points, normals, samples):
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
        primitive = self.primitives[idx]
        
        n = None if normals is None else normals[samples]
        shape = primitive.fit(points[samples], n)
        if shape is None or not self.limits[idx].check(shape):
            return None
                
        return shape

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
        num_points = len(points)
        if self._opt.max_point_distance is None:
            samples = random.sample(range(num_points), num_samples)

        else:
            samples = set([random.randrange(num_points)])
            
            while len(samples) < num_samples:
                
                # Array of shape (num_samples, num_points, 3), where diff[i, :, :]
                # is the different between each point to the ith sample
                diff = points[None] - points[list(samples)][:, None]
                
                # Find all of the distances, then get the minimum for each sample,
                # giving dist.shape == (n_points, )
                dist = np.min(np.linalg.norm(diff, axis=2), axis=0)
                
                # Find all of the possible points and then remove those already
                # sampled
                possible_samples = dist < self._opt.max_point_distance
                possible_samples[list(samples)] = False
                
                # Continue only if enough existing samples
                N_possible = sum(possible_samples)
                if N_possible < num_samples - len(samples):
                    return None
                
                idx_possible = np.where(possible_samples)[0]
                idx_sample = random.randrange(N_possible)
                samples.add(idx_possible[idx_sample])
                
        if len(samples) != num_samples:
            raise RuntimeError(f'Got {len(samples)} but expected '
                               f'{num_samples}, this should not happen')

        return list(samples)
    
    def get_biggest_connected_component(self, shape, points, inliers, 
                                        min_points=10):
        if self._opt.eps is None:
            return inliers
        
        points_flat = shape.flatten_points(points[inliers])
        
        pcd = PointCloud(Vector3dVector(points_flat))
        labels = pcd.cluster_dbscan(eps=self._opt.eps, min_points=min_points)
        labels = np.array(labels)
        if len(labels) < 2:
            return inliers
        size_connected = {}
        for label in set(labels):
            size_connected[label] = sum(labels == label)
        idx = labels == max(size_connected)
        return inliers[idx]
            
        # pcd_segmented = copy.copy(pcd_full)
    
    
    def get_inliers(self, shape, points, normals=None,
                    distances=None, angles=None, refit=False):
        """ Return indices of inliers: points whose distance to shape and
        angle with normal vector are below the given thresholds.
        
        Distances and angles can be taken as input if they have already been
        calculated to speed up execution. Otherwise, they are computed from 
        points.
        
        Parameters
        ----------
        shape : primitive
            Fitted shape
        points : array
            All input points
        normals, optional : array
            All input normal vectors 
        distances : optional, array
            Distances of each point to shape
        angles : optional, array
            Angles between each input and theoretical normal vector
        refit : boolean, optional
            Multiply threshold_refit_ratio with threshold_distance to have
            points in border
        
        Returns
        -------
        list
            Indices of inliers
        """
        
        if distances is None:
            distances = shape.get_distances(points)
        if angles is None and normals is not None:
            angles = shape.get_angles(points, normals)
        # distances, angles = shape.get_residuals(points, normals)
            
        threshold_distance = self._opt.threshold_distance
        if refit:
            threshold_distance *= self._opt.threshold_refit_ratio

        is_inlier = distances < threshold_distance
        
        if angles is not None:
            is_inlier *= (angles < self._opt.threshold_angle)
        inliers = np.where(is_inlier)[0]
        
        if self._opt.connected_components_density is not None:
            inliers = self.get_biggest_connected_component(
                shape, points, inliers)
            
        return inliers


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
        points = np.asarray(points)
        num_points = len(points)
        inliers_min = self._opt.inliers_min
        fitness_min = self._opt.fitness_min
        
        # if num_samples option is not set, get the max minimum value between
        # primitives
        num_samples = self.num_samples

        if num_points < num_samples:
            raise ValueError(f'Pointcloud must have at least '
                             f'{self.num_samples} points, {num_points} '
                             'given.')

        if normals is not None:
            if len(normals) == 0:
                normals = None
            elif len(normals) != num_points:
                raise ValueError('Numbers of points and normals must be equal')
            else:
                normals = np.asarray(normals)

        if inliers_min and num_points < inliers_min:
            if debug:
                print('Remaining points less than inliers_min, stopping')
            return None, None, self.get_metrics(None)

        # metrics dict is used to decide which model is best
        metrics_best = self.get_metrics(None)
        shape_best = None
        idx_best = None
        iteration_count = 0

        times = {'get_inliers': 0,
                 'get_model': 0}
        
        if debug:
            print(f'Starting loop, fitting {[p._name for p in self.primitives]}')
        
        for itr in range(self._opt.num_iterations):
            if (iteration_count > metrics_best['break_iteration']):
                continue

            start_itr = time.time()

            samples = self.get_samples(points, num_samples)
            if samples is None:
                continue

            t_ = time.time()
            metrics_itr = self.get_metrics(None)
            shape_itr = None
            idx_itr = None
            # idx_best_shape = 0
            for i in range(self._num_primitives):
                t_ = time.time()
                shape = self.get_model(i, points, normals, samples)
                times['get_model'] += time.time() - t_
                
                if shape is None:
                    continue
                
                distances, angles = shape.get_residuals(points, normals)
                inliers = self.get_inliers(
                    shape, points, normals, distances, angles, refit=False)
                num_inliers = len(inliers)
            
                times['get_inliers'] += time.time() - t_

                if num_inliers == 0 or (inliers_min and num_inliers < inliers_min):
                    continue
                
                if fitness_min and num_inliers/num_points < fitness_min:
                    continue
                
                metrics_this = self.get_metrics(
                    num_points, num_inliers, distances, angles)
                
                if self.compare_metrics(metrics_this, metrics_itr):
                    metrics_itr = metrics_this
                    shape_itr = shape
                    idx_itr = i

            if self.compare_metrics(metrics_itr, metrics_best):
                metrics_best = metrics_itr
                shape_best = shape_itr
                idx_best = idx_itr

            iteration_count += 1

            if debug:
                print(f'Iteration {itr+1}/{self._opt.num_iterations} : '
                      f'{time.time() - start_itr:.5f}s')
        
        if shape_best is None:
            return None, None, self.get_metrics(None)
        
        # Find the final inliers using model_best ...
        distances, angles = shape_best.get_residuals(points, normals)
        
        inliers_final = self.get_inliers(
            shape_best, points, normals, distances, angles, refit=True)
        
        num_inliers = len(inliers_final)
        
        metrics_final = self.get_metrics(
            num_points, num_inliers, distances, angles)

        # ... and then find the final model using the final inliers
        shape = self.get_model(idx_best, points, normals, inliers_final)
        if shape:
            shape_best = shape
        
        if shape_best is None:
            raise ValueError('None value found for shape at the last '
                             'filtering step, this should not happen')
            
        if debug:
            print(f'\nFinished fitting {shape_best.name}!')
            print(f'model: {shape_best.model}')
            print('Execution time:')
            for t_ in times:
                print(f'{t_} : {times[t_]:.5f}s')
            print(f'{num_points} points and {num_inliers} inliers, '
                  f'{int(100*metrics_final["fitness"])}% fitness')
            print(f'RMSE for distances: {metrics_final["rmse_distances"]}')
            print(f'RMSE for angles: {metrics_final["rmse_angles"]}\n')
            
        # shape_best.inlier_points = points[inliers_final]

        return shape_best, inliers_final, metrics_final
