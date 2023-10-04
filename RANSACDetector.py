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

from .PrimitiveBase import PrimitiveBase

DEG = 0.017453292519943295

class RANSACDetector(ABC):
    
    def __init__(self, 
                primitive,
                threshold_distance=0.1,
                threshold_angles=10,
                ransac_n=None, 
                num_iterations=100, 
                probability=0.99999,
                max_point_distance=None):
        
        if threshold_angles < 0:
            raise ValueError('threshold_angles must be positive')
        
        if probability <= 0 or probability > 1:
            raise ValueError('Probability must be > 0 and <= 1.0')
            
        if ransac_n is None:
            ransac_n = primitive._fit_n_min
        elif ransac_n < primitive._fit_n_min:
            raise ValueError(f'ransac_n should be set to higher than or equal '
                             f'to {self._fit_n_min}.')
        
        self.primitive = primitive
        self.threshold_distance = threshold_distance
        self.threshold_angles = threshold_angles * DEG
        self.threshold_cos = np.cos(self.threshold_angles)
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.probability = probability
        self.max_point_distance = max_point_distance
        # self.threshold_angles = threshold_angles
    
    def get_distances(self, shape, points):
        return shape.get_distances(points)
    
    def get_normal_angles_cos(self, points, normals, model):
        return self.primitive.get_normal_angles_cos(points, normals, model)
    
    def get_model(self, points, inliers):
        return self.primitive.get_model(points, inliers)
        
    def get_probabilities(self, distances):
        
        probabilities = np.zeros(len(distances))
        mask = distances < self.threshold_distance
        
        probabilities[mask] = 1 - distances[mask] / self.threshold_distance
        
        return probabilities
    
    def get_samples(self, points, num_points):
        
        if self.max_point_distance is None:
            return random.sample(range(num_points), self.ransac_n)
        
        samples = set()
        samples.add(random.randint(0, num_points))
        
        while len(samples) < self.ransac_n:
            sample = random.randint(0, num_points-1)
            point = points[sample]
            
            if sample in samples:
                continue
            
            distances = np.linalg.norm(points[list(samples)] - point, axis=1)
            if min(distances) > self.max_point_distance:
                continue
            
            samples.add(sample)
            
        return list(samples)
   
    def get_inliers_and_error(self, shape, points, normals=None):
        points_ = np.asarray(points)
        distances = shape.get_distances(points_)
        error = distances.dot(distances)
        inliers = distances < self.threshold_distance
        
        if normals is not None:
            normals_ = np.asarray(normals)
            normal_angles_cos = shape.get_normal_angles_cos(
                points_, normals_)
            inliers *= (normal_angles_cos > self.threshold_cos)
            
        inliers = np.where(inliers)[0]
        return inliers, error 
    
    def fit(self, points, debug=False, filter_model=True, normals=None):
        primitive = self.primitive
        points = np.asarray(points)
        num_points = len(points)
        
        if num_points < self.ransac_n:
            raise ValueError(f'Pointcloud must have at least {self.ransac_n} '
                             f'points, {num_points} given.')
            
        if normals is not None:
            if len(normals) != num_points:
                raise ValueError('Numbers of points and normals must be equal')
            normals = np.asarray(normals)
        
        fitness_best = 0
        best_rmse = 0
        
        shape_best = None

        break_iteration = 18446744073709551615
        iteration_count = 0
        
        times = {
            'get_inliers_and_error': 0,
            'get_inliers_and_error_final': 0,
            'get_model': 0,
            'get_model_final': 0,
            }
        
        for itr in range(self.num_iterations):
            
            start_itr = time.time()
            
            # if debug:
            #     print(f'Starting iteration {itr+1}/{self.num_iterations}...')
            
            if(iteration_count > break_iteration):
                continue
            
            # samples = 
            samples = self.get_samples(points, num_points)
            t_ = time.time()
            # model = self.get_model(points, samples)
            shape = primitive.create_from_points(points[samples])
            times['get_model'] += time.time() - t_
            
            # if shape is None
            if not shape:
                continue
            
            t_ = time.time()
            inliers, error = self.get_inliers_and_error(shape, points, normals)
            times['get_inliers_and_error'] += time.time() - t_
            inlier_num = len(inliers)
            
            
            if inlier_num == 0:
                fitness = 0
                rmse = 0
            else:
                fitness = inlier_num / len(points)
                rmse = np.sqrt(error / inlier_num)
            
            if (fitness > fitness_best or \
                (fitness == fitness_best and rmse < best_rmse)):
                
                fitness_best, best_rmse = fitness, rmse
                shape_best = shape
                
                if (fitness_best < 1.0):
                    num = np.log(1 - self.probability)
                    den = np.log(1 - fitness_best ** self.ransac_n)
                    
                    break_iteration = min(self.num_iterations, num / den) \
                        if den else self.num_iterations
                else:
                    break_iteration = 0
                
            iteration_count += 1
            
            if debug:
                print(f'Iteration {itr+1}/{self.num_iterations} : '
                      f'{time.time() - start_itr:.5f}s')
        
        # Find the final inliers using model_best...
        t_ = time.time()
        inliers_final, error_final = self.get_inliers_and_error(
            shape, points, normals)
        times['get_inliers_and_error_final'] = time.time() - t_
        fitness_final = len(inliers_final)/num_points
        
        if filter_model:
            # ... and then find the final model using the final inliers
            t_ = time.time()
            shape_best = primitive.create_from_points(points[inliers_final])
            times['get_model_final'] = time.time() - t_
        
        if debug:
            print('times:')
            for t_ in times:
                print (f'{t_} : {times[t_]:.5f}s')
            print(f'{num_points} points and {len(inliers_final)} inliers')
            print(f'fitness: {int(100*fitness_final)}%')
            print(f'rmse: {np.sqrt(error_final / len(inliers_final))}')
        
        return shape_best, inliers_final, fitness_final