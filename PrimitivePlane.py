#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
from abc import ABC, abstractmethod
import random
import copy
import numpy as np
import open3d as o3d
from open3d.geometry import TriangleMesh, PointCloud
from open3d.utility import Vector3iVector, Vector3dVector

from .PrimitiveBase import PrimitiveBase
    
class Plane(PrimitiveBase):
    
    _fit_n_min = 3
    _model_args_n = 4
    
    @staticmethod
    def get_mesh(model, pcd):
        
        # center = np.mean(np.asarray(points), axis=0)
        # center, size = args
        # center = np.mean(pcd.points, axis=0)
        # bb = pcd.get_axis_aligned_bounding_box()
        # half_length = max(bb.max_bound - bb.min_bound) / 2
        
        # normal = model[:3]
        # if list(normal) == [0, 0, 1]: 
        #     v1 = np.cross([0, 1, 0], normal)
        # else:
        #     v1 = np.cross([0, 0, 1], normal)
            
        # v2 = np.cross(v1, normal)
        # v1 /= np.linalg.norm(v1)
        # v2 /= np.linalg.norm(v2)

        # vertices = np.vstack([
        #     center + v1 * half_length,
        #     center + v2 * half_length,
        #     center - v1 * half_length,
        #     center - v2 * half_length])

        # triangles = Vector3iVector(np.array([
        #     [0, 1, 2], 
        #     [2, 1, 0],
        #     [0, 2, 3],
        #     [3, 2, 0]]))
        # vertices = Vector3dVector(vertices)

        # return TriangleMesh(vertices, triangles)
        
        pcd_flat = PointCloud()
        model[:3] /= np.linalg.norm(model[:3])
        distances = Plane.get_distances(pcd.points, model)
        pcd_flat.points = Vector3dVector(
            pcd.points - distances[..., np.newaxis] * model[:3])
        return pcd_flat.compute_convex_hull(joggle_inputs=True)[0]
        
    
    @staticmethod
    def get_distances(points, model):
        points = np.asarray(points)
        return np.abs(points.dot(model[:3]) + model[3])
    
    @staticmethod
    def get_model(points, samples):
        points_ = np.asarray(points)[samples]
        
        # if simplest case, the result is direct
        if len(samples) == 3:
            p0, p1, p2 = points_
            
            e0 = p1 - p0
            e1 = p2 - p0
            abc = np.cross(e0, e1)
            centroid = p0
        
        # for more points, find the plane such that the summed squared distance 
        # from the plane to all points is minimized.
        # Reference: https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        else:
            centroid = sum(points_) / len(samples)
            x, y, z = np.asarray(points_ - centroid).T
            xx = x.dot(x)
            yy = y.dot(y)
            zz = z.dot(z)
            xy = x.dot(y)
            yz = y.dot(z)
            xz = z.dot(x)
                
            det_x = yy * zz - yz * yz
            det_y = xx * zz - xz * xz
            det_z = xx * yy - xy * xy
            
            if (det_x > det_y and det_x > det_z):
                abc = np.array([det_x, 
                                xz * yz - xy * zz, 
                                xy * yz - xz * yy])
            elif (det_y > det_z):
                abc = np.array([xz * yz - xy * zz, 
                                det_y, 
                                xy * xz - yz * xx])
            else:
                abc = np.array([xy * yz - xz * yy, 
                                xy * xz - yz * xx, 
                                det_z])
        
        norm = np.linalg.norm(abc)
        if norm == 0.0:
            return np.array([0, 0, 0, 0])
        
        abc = abc / norm
        return np.array([abc[0], abc[1], abc[2], -abc.dot(centroid)]) 
        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    