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
# import open3d as o3d
# import multiprocessing
# from multiprocessing import Process, Manager

random.seed(951)

DEG = 0.017453292519943295


class RANSAC_Base(ABC):

    def __init__(self,
                 primitive,
                 reduction_rate=1.0,
                 threshold_distance=0.1,
                 threshold_angle=10,
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

        if not (model_max is None or len(model_max) == primitive._model_args_n):
            raise ValueError(f'for {self._type}s, model_max is either None or a'
                             f' list of size {primitive._model_args_n}, got '
                             f'{model_max}')

        if not (model_min is None or len(model_min) == primitive._model_args_n):
            raise ValueError(f'for {self._type}s, model_min is either None or a'
                             f' list of size {primitive._model_args_n}, got '
                             f'{model_min}')

        self.primitive = primitive
        self.reduction_rate = reduction_rate
        self.threshold_distance = threshold_distance
        self.threshold_angle = threshold_angle * DEG
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.probability = probability
        self.max_point_distance = max_point_distance
        self.model_max = None if model_max is None else np.array(model_max)
        self.model_min = None if model_min is None else np.array(model_min)
        self.inliers_min = inliers_min
    
    @abstractmethod
    def compare_metrics(self, metrics, metrics_best):
        pass
    
    def termination_criterion(self, metrics):
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

    def get_model(self, points, samples):
        shape = self.primitive.fit(points[samples])
        
        if shape is not None:
            model_array = np.array(shape.model)
    
            if self.model_max is not None:
                idx_max = np.where(self.model_max != None)[0]
                if np.any(self.model_max[idx_max] < model_array[idx_max]):
                    return None
    
            if self.model_min is not None:
                idx_min = np.where(self.model_min != None)[0]
                if np.any(self.model_min[idx_min] > model_array[idx_min]):
                    return None

        return shape

    def get_probabilities(self, distances, angles=None):

        if angles is not None and len(distances) != len(angles):
            raise ValueError('distances and angles must have the same size, '
                             f'got {distances.shape} and {angles.shape}.')

        probabilities = np.zeros(len(distances))

        mask = distances < self.threshold_distance
        if angles is not None:
            mask *= angles < self.threshold_angle

        probabilities[mask] = 1 - distances[mask] / self.threshold_distance

        if angles is not None:
            probabilities[mask] *= 1 - angles[mask] / self.threshold_angle

        return probabilities

    def get_samples(self, points, num_points, tries_max=5000):

        if self.max_point_distance is None:
            return random.sample(range(num_points), self.ransac_n)

        samples = set([random.randrange(num_points)])
        
        tries = 0
        while len(samples) < self.ransac_n:
            
            # if the algorithm cannot find another sample in the minimal
            # neighborhood, stop
            if tries > tries_max:
                return None
                # samples = set([random.randrange(num_points)])
                # tries = 0
            
            sample = random.randrange(num_points)
            if sample in samples:
                continue
            
            # tries = 0
            point = points[sample]

            distances = np.linalg.norm(points[list(samples)] - point, axis=1)
            if min(distances) > self.max_point_distance:
                tries += 1
                continue

            samples.add(sample)
            tries = 0

        return list(samples)
    
    def get_inliers_from_points(self, shape, points, normals=None):
        distances, angles = shape.get_residuals(points, normals)
        return self.get_inliers_from_residuals(distances, angles)
    
    def get_inliers_from_residuals(self, distances, angles):
        is_inlier = distances < self.threshold_distance
        if angles is not None:
            is_inlier *= (angles < self.threshold_angle)
        return np.where(is_inlier)[0]

    def fit(self, points, normals=None, debug=False):#, filter_model=True):
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
            shape = self.get_model(points, samples)
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
        shape_best = primitive.fit(points[inliers_final])
        if shape_best is None:
            raise ValueError('None value found for shape at the last '
                             'filtering step, this should not happen')
            
        if debug:
            print(f'\nFinished fitting {primitive.name}!')
            print('Execution time:')
            for t_ in times:
                print(f'{t_} : {times[t_]:.5f}s')
            print(f'{num_points} points and {num_inliers} inliers, '
                  f'{int(100*metrics_final["fitness"])}% fitness')
            print(f'RMSE for distances: {metrics_final["rmse_distances"]}')
            print(f'RMSE for angles: {metrics_final["rmse_angles"]}\n')

        return shape_best, inliers_final, metrics_final
