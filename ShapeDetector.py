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

class ShapeDetector(ABC):
    
    def __init__(self, 
                distance_threshold, 
                ransac_n, 
                num_iterations, 
                probability):
        
        if probability <= 0 or probability > 1:
            raise ValueError('Probability must be > 0 and <= 1.0')
            
        if ransac_n < self._ransac_n_min:
            raise ValueError(f'ransac_n should be set to higher than or equal '
                             f'to {self._ransac_n_min}.')
        
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.probability = probability
    
    @staticmethod
    @abstractmethod
    def get_distances(points, model):
        pass
        
    @staticmethod
    @abstractmethod
    def get_model(points, inliers):
        pass
   
    def get_inliers_and_error(self, points, model):

        distances = self.get_distances(np.asarray(points), model)
        error = distances.dot(distances)
        inliers = np.where(distances < self.distance_threshold)[0]
            
        return inliers, error 
    
    def run_ransac(self, points, debug=False, filter_model=False):
        
        points = np.asarray(points)
        num_points = len(points)
        
        if num_points < self.ransac_n:
            raise ValueError('There must be at least \'ransac_n\' points.')
        
        fitness_best = 0
        best_rmse = 0
        
        model_best = np.zeros(self._model_args_n)

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
            
            samples = random.sample(range(num_points), self.ransac_n)
            t_ = time.time()
            model = self.get_model(points, samples)
            times['get_model'] += time.time() - t_
            
            # if model is all zeros
            if not np.any(model):
                continue
            
            t_ = time.time()
            inliers, error = self.get_inliers_and_error(points, model)
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
                model_best = model
                
                if (fitness_best < 1.0):
                    break_iteration = min(
                        self.num_iterations,
                        np.log(1 - self.probability) / np.log(1 - fitness_best ** self.ransac_n)
                        )
                else:
                    break_iteration = 0
                
            iteration_count += 1
            
            if debug:
                print(f'Iteration {itr+1}/{self.num_iterations} : {time.time() - start_itr:.5f}s')
        
        # Find the final inliers using model_best...
        t_ = time.time()
        inliers_final, _ = self.get_inliers_and_error(points, model_best)
        times['get_inliers_and_error_final'] = time.time() - t_
        
        if filter_model:
            # ... and then find the final model using the final inliers
            t_ = time.time()
            model_best = self.get_model(points, inliers_final)
            times['get_model_final'] = time.time() - t_
        
        if debug:
            print('times:')
            for t_ in times:
                print (f'{t_} : {times[t_]:.5f}s')
        
        return model_best, inliers_final