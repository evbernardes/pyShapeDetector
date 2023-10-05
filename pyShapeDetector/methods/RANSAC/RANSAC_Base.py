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
            raise ValueError(f'for {primitive._name}s, ransac_n should be set '
                             'to higher than or equal '
                             f'to {primitive._fit_n_min}.')

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
    def weight_distance(self, distances):
        pass

    @abstractmethod
    def weight_angle(self, angles):
        pass
    
    @abstractmethod
    def compare_info(self, info, info_best):
        pass

    def get_total_weight(self, distances, angles):
        weight_distance = sum(self.weight_distance(distances))
        weight_angle = sum(self.weight_angle(angles))
        return weight_distance * weight_angle

    def get_model(self, points, samples):
        shape = self.primitive.create_from_points(points[samples])
        if shape is None:
            return None

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
        # probabilities[mask] = 1 - distances[mask] / self.threshold_distance

        if angles is not None:
            mask *= angles < self.threshold_angle

        probabilities[mask] = 1 - distances[mask] / self.threshold_distance

        if angles is not None:
            probabilities[mask] *= 1 - angles[mask] / self.threshold_angle

        return probabilities

    def get_samples(self, points, num_points):

        if self.max_point_distance is None:
            return random.sample(range(num_points), self.ransac_n)

        sample = random.randrange(num_points)
        samples = set([sample])

        while len(samples) < self.ransac_n:

            sample = random.randrange(num_points)
            if sample in samples:
                continue
            
            point = points[sample]

            distances = np.linalg.norm(points[list(samples)] - point, axis=1)
            if min(distances) > self.max_point_distance:
                continue

            samples.add(sample)

        return list(samples)
    
    def get_inliers(self, distances, angles):

        is_inlier = distances < self.threshold_distance
        if angles is not None:
            is_inlier *= (angles < self.threshold_angle)
        return np.where(is_inlier)[0]
    
    def get_error(self, distances, angles=None):
        return distances.dot(distances)
    
    def get_rmse(self, distances, angles=None):
        return np.sqrt(self.get_error(distances, angles)) / len(distances)

    def fit(self, points, normals=None, debug=False, filter_model=True):
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
            return None, None, 0

        info_best = {'num_inliers': 0,
                     'fitness': 0,
                     'weight': 0,
                     'rmse': None}
        
        shape_best = None
        break_iteration = 18446744073709551615
        iteration_count = 0

        times = {'get_inliers_and_error': 0,
                 'get_inliers_and_error_final': 0,
                 'get_model': 0,
                 'get_model_final': 0}

        for itr in range(self.num_iterations):

            if (iteration_count > break_iteration):
                continue

            start_itr = time.time()

            samples = self.get_samples(points, num_points)
            
            t_ = time.time()
            shape = self.get_model(points, samples)
            times['get_model'] += time.time() - t_

            if shape is None:
                continue

            t_ = time.time()
            distances, angles = shape.get_distances_and_angles(points, normals)
            inliers = self.get_inliers(distances, angles)
            inlier_num = len(inliers)
            
            times['get_inliers_and_error'] += time.time() - t_

            if inlier_num == 0 or (inliers_min and inlier_num < inliers_min):
                continue
            
            info = {'num_inliers': inlier_num,
                    'fitness': inlier_num / num_points,
                    'weight': self.get_total_weight(distances, angles),
                    'rmse': self.get_rmse(distances)}

            if self.compare_info(info, info_best):
                
                info_best = info
                shape_best = shape

                if (info['fitness'] < 1.0):
                    num = np.log(1 - self.probability)
                    den = np.log(1 - info['fitness'] ** self.ransac_n)

                    break_iteration = min(self.num_iterations, num / den) \
                        if den else self.num_iterations
                else:
                    break_iteration = 0

            iteration_count += 1

            if debug:
                print(f'Iteration {itr+1}/{self.num_iterations} : '
                      f'{time.time() - start_itr:.5f}s')

        # Find the final inliers using model_best...
        if shape_best is None:
            return None, None, {'num_inliers': 0, 'fitness': 0, 
                                'weight': 0, 'rmse': None}
        
        t_ = time.time()
        distances, angles = shape_best.get_distances_and_angles(
            points, normals)
        inliers_final = self.get_inliers(distances, angles)
        num_inliers = len(inliers_final)
        
        info_final = {'num_inliers': inlier_num,
                      'fitness': inlier_num / num_points,
                      'weight': self.get_total_weight(distances, angles),
                      'rmse': self.get_rmse(distances)}
        
        times['get_inliers_and_error_final'] = time.time() - t_

        if filter_model:
            # ... and then find the final model using the final inliers
            t_ = time.time()
            shape_best = primitive.create_from_points(points[inliers_final])
            if shape_best is None:
                raise ValueError('None value found for shape at the last '
                                 'filtering step, this should not happen')
            times['get_model_final'] = time.time() - t_
            
        if debug:
            print('times:')
            for t_ in times:
                print(f'{t_} : {times[t_]:.5f}s')
            print(f'{num_points} points and {num_inliers} inliers')
            print(f'fitness: {int(100*info_final["fitness"])}%')
            print(f'rmse: {info_final["rmse"]}')

        return shape_best, inliers_final, info_final
