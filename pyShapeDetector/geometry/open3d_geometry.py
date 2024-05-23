#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:00:58 2024

@author: ebernardes
"""

import numpy as np
from open3d import utility

def _in_module(value, module):
    return type(value).__module__ == module.__name__

def _convert_args_to_numpy(args):
    if isinstance(args, (list, tuple)):
        for i, value in enumerate(args):
            args[i] = _convert_args_to_numpy(value)
    
    elif _in_module(args, utility):
        return np.asarray(args)
    return args

def _convert_args_to_open3d(*args, **kwargs):
    args = list(args)
    
    for i, arg in enumerate(args):
        if hasattr(arg, '_open3d'):
            args[i] = arg._open3d
        
        if isinstance(arg, (list, tuple)):
            arg = np.array(arg)
        
        if isinstance(arg, np.ndarray):
            if arg.ndim > 2:
                continue
            
            if arg.ndim == 1:
                if arg.dtype == int:
                    args[i] = utility.IntVector(arg.tolist())
                else:
                    args[i] = utility.DoubleVector(arg.tolist())
                    
            if arg.ndim == 2:
                d = 'i' if arg.dtype == int else 'd'
                class_name = f'Vector{arg.shape[1]}{d}Vector'
                
                converter = getattr(utility, class_name, lambda x: x)
                args[i] = converter(arg)
                
    return tuple(args), kwargs

def result_as_numpy(func):
    def wrapper(*args, **kwargs):
        return _convert_args_to_numpy(func(*args, **kwargs))
    return wrapper

def args_to_open3d(func):
    if func is None:
        return None
    
    def wrapper(*args, **kwargs):
        args, kwargs = _convert_args_to_open3d(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

def to_open3d_and_back(func):
    if func is None:
        return None
    
    def wrapper(*args, **kwargs):
        args, kwargs = _convert_args_to_open3d(*args, **kwargs)
        return _convert_args_to_numpy(func(*args, **kwargs))
    return wrapper

def link_to_open3d_geometry(original_class):
    def _copy_from_open3d(cls):
        
        setattr(cls, '__open3d_class__', original_class)

        # create new __init__ that generates an internal variable
        @args_to_open3d
        def new_init(self, *args, **kwargs):
            self._open3d = original_class(*args, **kwargs)
        setattr(cls, '__init__', new_init)

        # removing two last subclasses, 'object' and 'pybind11_object'
        for subclass in original_class.__mro__[:-2]:
            for attr_name, attr_value in subclass.__dict__.items():
                if attr_name == '__init__':
                    continue

                if attr_name not in cls.__mro__[0].__dict__:
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
    return _copy_from_open3d

class Open3D_Geometry():
    
    @property
    def as_open3d(self):
        return self._open3d