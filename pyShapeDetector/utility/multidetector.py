#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:30:28 2023

@author: ebernardes
"""

import copy
import time
import numpy as np

class MultiDetector():
    
    def __init__(self, detectors, pcds, points_min=500, shapes_per_cluster=20,
                 debug=0, compare_metric='fitness', metric_min=0.1, 
                 normals_reestimate=False, fuse_shapes=False):
    
        if not isinstance(detectors, list):
            detectors = [detectors]
        
        if not isinstance(pcds, list):
            pcds = [pcds]
            
        self.pcds = pcds
        self.n_pcds = len(pcds)
        self.detectors = detectors
        self.n_detectors = len(detectors)
        self.points_min = points_min
        self.shapes_per_cluster = shapes_per_cluster
        
        # Start:
        self._shapes_detected = None
        self._pcds_rest = None
        self._finished = False
        
        self.run(debug, compare_metric, metric_min, normals_reestimate, fuse_shapes)
        
    def __repr__(self):
        line = type(self).__name__+'('
        line += str([d._type for d in self.detectors]).replace("'", "")
        line +=', '
        line += f'{len(self.shapes)} shapes detected, '
        points_classified = sum([len(s.inlier_points) for s in self.shapes])
        points_remaining = sum([len(p.points) for p in self.pcds_rest])
        line += f'{points_classified}/{points_classified+points_remaining} points classified)'
        return line
        
    @property
    def pcds_inliers(self):
        if not self._finished:
            raise RuntimeError('MultiDetector still did not fit, try to run, '
                               'see: MultiDetector.run')
            
        return [s.inliers for s in self._shapes_detected]
        
    @property
    def pcds_rest(self):
        if not self._finished:
            raise RuntimeError('MultiDetector still did not fit, try to run, '
                               'see: MultiDetector.run')
            
        return self._pcds_rest

    @property
    def shapes(self):
        if not self._finished:
            raise RuntimeError('MultiDetector still did not fit, try to run, '
                               'see: MultiDetector.run')
            
        return self._shapes_detected

    @property
    def metrics(self):
        if not self._finished:
            raise RuntimeError('MultiDetector still did not fit, try to run, '
                               'see: MultiDetector.run')
            
        return [s.metrics for s in self._shapes_detected]
        
    def run(self, debug, compare_metric, metric_min, 
            normals_reestimate, fuse_shapes):
        
        debug_detectors = debug > 1
        debug = debug > 0
        
        # times = {}
        # for detector in self.detectors:
        #     for primitive in detector.primitives:
        #         if primitive.name in times:
        #             continue
        #         times[primitive.name] = 0
        
        shapes_detected = []
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
                  iteration < self.shapes_per_cluster):
                
                if debug:
                    print(f'\niteration {iteration+1}/{self.shapes_per_cluster}, '
                          f'cluster {idx+1}/{len(self.pcds)}')
                
                if normals_reestimate:
                    pcd_.estimate_normals()
                # normals = pcd_.normals
                
                output_shapes = []
                output_inliers = []
                output_metrics = []
                compare = []
                
                for detector in self.detectors:
                    # start = time.time()
                    shape, inliers, metrics = detector.fit(
                        pcd_, debug=debug_detectors)
                    # times[detector.primitive.name] += time.time() - start
                    output_shapes.append(shape)
                    output_inliers.append(inliers)
                    output_metrics.append(metrics)
                    compare.append(metrics[compare_metric])
                    
                iteration += 1
                
                if np.all(np.array(output_shapes) == None):
                    if debug:
                        print('No shapes found anymore, breaking...')
                    continue
                
                if metric_min is not None and max(compare) < metric_min:
                    if debug:
                        print(f'{compare_metric} of {max(compare)} is too '
                              'small, breaking...')
                    continue
                
                idx = np.where(np.array(compare) == max(compare))[0][0]
                    
                shape = output_shapes[idx]
                inliers = output_inliers[idx]
                metrics = output_metrics[idx]
                if debug:
                    print(f'-> {shape.name.capitalize()} found, with '
                          f'{compare_metric} = {max(compare)}')

                pcd_inliers = pcd_.select_by_index(inliers)
                pcd_ = pcd_.select_by_index(inliers, invert=True)
                shape.set_inliers(pcd_inliers)
                
                shape.metrics = metrics
                shapes_detected.append(shape)
                iteration += 1
            
            if len(pcd_.points) != 0:
                pcds_rest.append(pcd_)

        if debug:
            print('\n-------------------------------------------')
            print(f'Finished after {time.time() - start:.5f}s')
            print('Time spend with each detector:')                
                
        self._shapes_detected = shapes_detected
        self._pcds_rest = pcds_rest
        self._finished = True