#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:42:59 2023

@author: ebernardes
"""
import numpy as np
from scipy.spatial import ConvexHull
from open3d.geometry import TriangleMesh, PointCloud
from open3d.utility import Vector3iVector, Vector3dVector

from .primitivebase import PrimitiveBase
    
class Plane(PrimitiveBase):
    
    _fit_n_min = 3
    _model_args_n = 4
    name = 'plane'
    
    @property
    def normal(self):
        return np.array(self.model[:3])
    
    def get_distances(self, points):
        points = np.asarray(points)
        return np.abs(points.dot(self.normal) + self.model[3])
    
    def get_normals(self, points):
        return np.tile(self.normal, (len(points), 1))
    
    def get_mesh(self, points):
        
        points = np.asarray(points)

        # needs signed distance
        distances = points.dot(self.normal) + self.model[3]
        points -= (distances * self.normal[..., np.newaxis]).T
        
        rot = self.get_rotation_from_axis(self.normal)
        projection = (rot @ points.T).T[:, :2]
        chull = ConvexHull(projection)       
        
        pcd_flat = PointCloud(
            Vector3dVector(points[chull.vertices]))
        
        pcd_flat.normals = Vector3dVector(
            np.repeat(self.normal, chull.nsimplex).reshape(chull.nsimplex, 3))
        
        return None
    
    def get_square_mesh(self, pcd):
        
        center = np.mean(np.asarray(pcd.points), axis=0)
        bb = pcd.get_axis_aligned_bounding_box()
        half_length = max(bb.max_bound - bb.min_bound) / 2
        
        if list(self.normal) == [0, 0, 1]: 
            v1 = np.cross([0, 1, 0], self.normal)
        else:
            v1 = np.cross([0, 0, 1], self.normal)
            
        v2 = np.cross(v1, self.normal)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        vertices = np.vstack([
            center + v1 * half_length,
            center + v2 * half_length,
            center - v1 * half_length,
            center - v2 * half_length])

        triangles = Vector3iVector(np.array([
            [0, 1, 2], 
            [2, 1, 0],
            [0, 2, 3],
            [3, 2, 0]]))
        vertices = Vector3dVector(vertices)

        return TriangleMesh(vertices, triangles)
    
    @staticmethod
    def fit(points, normals=None):
        points = np.asarray(points)        
        num_points = len(points)
        
        if num_points < 3:
            raise ValueError('A minimun of 3 points are needed to fit a '
                             'plane')
        
        # if simplest case, the result is direct
        elif num_points == 3:
            p0, p1, p2 = points
            
            e0 = p1 - p0
            e1 = p2 - p0
            abc = np.cross(e0, e1)
            centroid = p0
        
        # for more points, find the plane such that the summed squared distance 
        # from the plane to all points is minimized.
        # Reference: https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        else:
            centroid = sum(points) / num_points
            x, y, z = np.asarray(points - centroid).T
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
            return None
        
        abc = abc / norm
        return Plane([abc[0], abc[1], abc[2], -abc.dot(centroid)]) 
        

            
        
        
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    