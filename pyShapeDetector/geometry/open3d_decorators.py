#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:00:58 2024

@author: ebernardes
"""
import functools
import numpy as np
from open3d import geometry, utility

# Maps the dimension and type of variable with the converter function
# that creates Eigen instances from the original lists/arrays
converters_vector = {
    (1, (), int): utility.IntVector,
    (1, (), float): utility.DoubleVector,
    (2, (2,), int): utility.Vector2iVector,
    (2, (2,), float): utility.Vector2dVector,
    (2, (3,), int): utility.Vector3iVector,
    (2, (3,), float): utility.Vector3dVector,
    (2, (4,), int): utility.Vector4iVector,
    (3, (3, 3), float): utility.Matrix3dVector,
    (3, (4, 4), float): utility.Matrix4dVector,
    }
 
def _convert_args_to_open3d(*args, **kwargs):
    # Convert every argument to Eigen instances whenever possible
    args = list(args)
    
    for i, arg in enumerate(args):
        if hasattr(arg, '_open3d'):
            args[i] = arg._open3d
        
        if isinstance(arg, (list, tuple)):
            arg = np.array(arg)
        
        if isinstance(arg, np.ndarray) and arg.shape == (0, ):
            arg = np.empty((0, 3))
        
        if isinstance(arg, np.ndarray):
            
            if np.issubdtype(arg.dtype, np.integer):
                dtype = int
            elif np.issubdtype(arg.dtype, np.floating):
                dtype = float
            else:
                dtype = None
            
            if (key := (arg.ndim, arg.shape[1:], dtype)) in converters_vector:
                try:
                    args[i] = converters_vector[key](arg)
                except TypeError:
                    args[i] = converters_vector[key](arg.tolist())
                
    return tuple(args), kwargs

def _convert_args_to_numpy(args):
    # Convert every argument back from Eigen instances, recursively  
    from .pointcloud import PointCloud
    from .trianglemesh import TriangleMesh
    from .axis_aligned_bounding_box import AxisAlignedBoundingBox

    converters_classes = {
        geometry.PointCloud: PointCloud,
        geometry.TriangleMesh: TriangleMesh,
        geometry.AxisAlignedBoundingBox: AxisAlignedBoundingBox,
    }

    if isinstance(args, (list, tuple)):
        for i, value in enumerate(args):
            args[i] = _convert_args_to_numpy(value)
    
    elif type(args) in converters_vector.values():
        return np.asarray(args)
    
    elif type(args) in converters_classes:
        return converters_classes[type(args)](args)

    return args

def result_as_numpy(func):
    @functools.wraps(func)
    
    def wrapper(*args, **kwargs):
        return _convert_args_to_numpy(func(*args, **kwargs))
    return wrapper

def args_to_open3d(func):
    if func is None:
        return None
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = _convert_args_to_open3d(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

def to_open3d_and_back(func):
    if func is None:
        return None
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = _convert_args_to_open3d(*args, **kwargs)
        return _convert_args_to_numpy(func(*args, **kwargs))
    return wrapper