#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:30:28 2023

@author: ebernardes
"""

import copy
import time
import numpy as np

#%% Parameters and input

class MultiDetector():
    
    def __init__(self, detectors, pcds, points_min=500, num_iterations=20):
    
        if not isinstance(detectors, list):
            detectors = [detectors]
        
        if not isinstance(pcds, list):
            pcds = [pcds]
            
        self.pcds = pcds
        self.n_pcds = len(pcds)
        self.detectors = detectors
        self.n_detectors = len(detectors)
        self._shapes_detected = None
        self._meshes_detected = None
        self._pcds_inliers = None
        self._pcds_rest = None
        self._finished = False
        self.points_min = points_min
        self.num_iterations = num_iterations
        
    @property
    def pcds_inliers(self):
        if not self._finished:
            raise RuntimeError('MultiDetector still did not fit, try to run, '
                               'see: MultiDetector.run')
            
        return self._pcds_inliers
        
    @property
    def pcds_rest(self):
        if not self._finished:
            raise RuntimeError('MultiDetector still did not fit, try to run, '
                               'see: MultiDetector.run')
            
        return self._pcds_rest

    @property
    def shapes_detected(self):
        if not self._finished:
            raise RuntimeError('MultiDetector still did not fit, try to run, '
                               'see: MultiDetector.run')
            
        return self._shapes_detected
    
    @property
    def meshes_detected(self):
        if not self._finished:
            raise RuntimeError('MultiDetector still did not fit, try to run, '
                               'see: MultiDetector.run')
            
        if self._meshes_detected is None:
            meshes_detected = []
            for shape, pcd in zip(self._shapes_detected, self._pcds_inliers):                
                mesh = shape.get_mesh(pcd.points)
                mesh.paint_uniform_color(np.random.random(3))
                meshes_detected.append(mesh)
            self._meshes_detected = meshes_detected
            
        return self._meshes_detected
        
    def run(self, debug=0, compare_metric='fitness', fitness_min=0.1, 
            normals_reestimate=False):
        
        self._shapes_detected = None
        self._meshes_detected = None
        self._pcds_inliers = None
        self._pcds_rest = None
        self._finished = False
        
        debug_detectors = debug > 1
        debug = debug > 0
        
        times = {detector.primitive.name: 0 for detector in self.detectors}
        
        shapes_detected = []
        pcds_inliers = []
        pcds_rest = []
        
        start = time.time()
        if debug:
            print('\n-------------------------------------------')
            print('\nStarting... ')
            
        for idx in range(self.n_pcds):
            if debug:
                print(f'Testing cluster {idx+1}...')
                
            pcd_ = copy.copy(self.pcds[idx])
            
            iteration = 0
            while(len(pcd_.points) > self.points_min and \
                  iteration < self.num_iterations):
                
                if debug:
                    print(f'\niteration {iteration}')
                
                if normals_reestimate:
                    pcd_.estimate_normals()
                normals = pcd_.normals
                
                output_shapes = []
                output_inliers = []
                output_metrics = []
                output_fitness = []
                compare = []
                
                for detector in self.detectors:
                    start = time.time()
                    shape, inliers, metrics = detector.fit(
                        pcd_.points, debug=debug_detectors, normals=normals)
                    times[detector.primitive.name] += time.time() - start
                    output_shapes.append(shape)
                    output_inliers.append(inliers)
                    output_metrics.append(metrics)
                    output_fitness.append(metrics['fitness'])
                    compare.append(metrics[compare_metric])
                    
                if np.all(np.array(output_shapes) == None):
                    if debug:
                        print('No shapes found anymore, breaking...')
                    break
                
                if max(output_fitness) < fitness_min:
                    if debug:
                        print('Fitness to small, breaking...')
                    break
                
                idx = np.where(np.array(compare) == max(compare))[0][0]
                    
                shape = output_shapes[idx]
                inliers = output_inliers[idx]
                if debug:
                    print(f'-> {shape.name.capitalize()} found!')

                pcd_inliers = pcd_.select_by_index(inliers)
                pcd_ = pcd_.select_by_index(inliers, invert=True)
                
                shapes_detected.append(shape)
                pcds_inliers.append(pcd_inliers)
                iteration += 1
            
            if len(pcd_.points) != 0:
                pcds_rest.append(pcd_)

        if debug:
            print('\n-------------------------------------------')
            print(f'Finished after {time.time() - start:.5f}s')
            print('Time spend with each detector:')
            
            for detector in self.detectors:
                name = detector.primitive.name
                print(f'- {name}: {times[name]:.3f}s')
                
        self._shapes_detected = shapes_detected
        self._pcds_inliers = pcds_inliers
        self._pcds_rest = pcds_rest
        self._finished = True

