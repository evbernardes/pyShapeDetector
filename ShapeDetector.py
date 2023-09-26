#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:59:55 2023

@author: ebernardes
"""

from abc import ABC, abstractmethod
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
    
    def run_ransac(self, points, print_iteration=False):
        
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
        
        for itr in range(self.num_iterations):
            if print_iteration:
                print(f'Iteration {itr+1}/{self.num_iterations}')
            
            if(iteration_count > break_iteration):
                continue
            
            samples = random.sample(range(num_points), self.ransac_n)
            model = self.get_model(points, samples)
            
            # if model is all zeros
            if not np.any(model):
                continue
            
            inliers, error = self.get_inliers_and_error(points, model)
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
        
        # Find the final inliers using best_plane_model
        final_inliers, _ = self.get_inliers_and_error(points, best_model)
        best_plane_model = self.get_model(points, final_inliers)
        
        return best_plane_model, final_inliers