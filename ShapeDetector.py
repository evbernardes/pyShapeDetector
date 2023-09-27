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
    def get_distance(point, model):
        pass
        
    @staticmethod
    @abstractmethod
    def get_model(points, inliers):
        pass
   
    def get_inliers_and_error(self, points, model):
        error = 0
        inliers = []
        
        for idx in range(len(points)):
                
            distance = self.get_distance(points[idx], model)

            if (distance < self.distance_threshold):
                error += distance * distance
                inliers.append(idx)
            
        return inliers, error 
    
    def run_ransac(self, points, debug=False):
        
        points = np.asarray(points)
        num_points = len(points)
        
        if num_points < self.ransac_n:
            raise ValueError('There must be at least \'ransac_n\' points.')
        
        # result = {'fitness': 0, 'inlier_rmse': 0}
        best_fitness = 0
        best_rmse = 0
        
        best_model = np.zeros(self._model_args_n)

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
                fitness, rmse = 0
            else:
                fitness = inlier_num / len(points)
                rmse = np.sqrt(error / inlier_num)
            
            if (fitness > best_fitness or \
                (fitness == best_fitness and rmse < best_rmse)):
                
                best_fitness, best_rmse = fitness, rmse
                best_model = model
                
                if (best_fitness < 1.0):
                    break_iteration = min(
                        self.num_iterations,
                        np.log(1 - self.probability) / np.log(1 - best_fitness ** self.ransac_n)
                        )
                else:
                    break_iteration = 0
                
            iteration_count += 1
            
            if debug:
                print(f'Iteration {itr+1}/{self.num_iterations} : {time.time() - start_itr:.5f}s')
            
            # times[f'itr_{itr}'] = time.time() - start_itr
        
        # Find the final inliers using best_plane_model
        t_ = time.time()
        final_inliers, _ = self.get_inliers_and_error(points, best_model)
        times['get_inliers_and_error_final'] += time.time() - t_
        t_ = time.time()
        best_plane_model = self.get_model(points, final_inliers)
        times['get_model_final'] += time.time() - t_
        
        
        if debug:
            print('times:')
            for t_ in times:
                print (f'{t_} : {times[t_]:.5f}s')
        
        return best_plane_model, final_inliers