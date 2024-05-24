#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:00:58 2024

@author: ebernardes
"""
import copy
import functools
import numpy as np
from open3d import geometry, utility

converters_vector = {
    (1, int): utility.IntVector,
    (1, float): utility.DoubleVector,
    (2, int): utility.Vector2iVector,
    (2, float): utility.Vector2dVector,
    (3, int): utility.Vector3iVector,
    (3, float): utility.Vector3dVector,
    (4, int): utility.Vector4iVector,
    }

empty_3d_array = np.array([])
empty_3d_array.shape = (0, 3)

def _convert_args_to_numpy(args):
    
    from .pointcloud import PointCloud
    from .trianglemesh import TriangleMesh

    converters_classes = {
        geometry.PointCloud: PointCloud,
        geometry.TriangleMesh: TriangleMesh,
        }

    if isinstance(args, (list, tuple)):
        for i, value in enumerate(args):
            args[i] = _convert_args_to_numpy(value)
    
    elif type(args) in converters_vector.values():
        return np.asarray(args)
    
    elif type(args) in converters_classes:
        return converters_classes[type(args)](args)

    return args

def _convert_args_to_open3d(*args, **kwargs):
    args = list(args)
    
    for i, arg in enumerate(args):
        if hasattr(arg, '_open3d'):
            args[i] = arg._open3d
        
        if isinstance(arg, (list, tuple)):
            arg = np.array(arg)
        
        if isinstance(arg, np.ndarray) and arg.shape == (0, ):
            arg = empty_3d_array
        
        if isinstance(arg, np.ndarray):
            
            if np.issubdtype(arg.dtype, np.integer):
                dtype = int
            elif np.issubdtype(arg.dtype, np.floating):
                dtype = float
            else:
                dtype = None
            
            if (dim := arg.ndim) > 1:
                dim = arg.shape[1]
            else:
                arg = arg.tolist() 
            
            if (dim, dtype) in converters_vector:
                args[i] = converters_vector[dim, dtype](arg)
                
    return tuple(args), kwargs

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

def link_to_open3d_geometry(original_class):
    def decorator(cls):
        
        setattr(cls, '__open3d_class__', original_class)

        # create new __init__ that generates an internal variable
        @args_to_open3d
        def new_init(self, *args, **kwargs):
            self._open3d = original_class(*args, **kwargs)
        setattr(cls, '__init__', new_init)

        # removing two last subclasses, 'object' and 'pybind11_object'
        for subclass in original_class.__mro__[:-2]:
            for attr_name, attr_value in subclass.__dict__.items():
                
                # checking if already exists ir Open3D_Geometry or its child
                if attr_name == '__init__' or attr_name in cls.__mro__[0].__dict__ or attr_name in cls.__mro__[1].__dict__:
                    continue

                # if attr_name not in cls.__mro__[0].__dict__:
                if callable(attr_value):
                    setattr(cls, attr_name, to_open3d_and_back(attr_value))
                
                elif isinstance(attr_value, property):
                    # Apply decorators to the original getter and setter functions                        
                    getter = to_open3d_and_back(attr_value.fget)
                    setter = args_to_open3d(attr_value.fset)
                    
                    # Create a new property with the decorated getter and setter functions
                    new_property = property(getter, setter, attr_value.fdel, attr_value.__doc__)
                    
                    setattr(cls, attr_name, new_property)
                else:
                    setattr(cls, attr_name, attr_value)
        return cls
    return decorator

class Open3D_Geometry():
    __open3d_class__ = None
    
    @property
    def as_open3d(self):
        return self._open3d
    
    def __copy__(self, *args):
        _open3d = copy.copy(self.as_open3d)
        return type(self)(_open3d)
    
    def __deepcopy__(self, *args):
        _open3d = copy.deepcopy(self.as_open3d)
        return type(self)(_open3d)
        # return "test"
        
    @classmethod
    def is_instance_or_open3d(cls, elem):
        return isinstance(elem, (cls, cls.__open3d_class__))