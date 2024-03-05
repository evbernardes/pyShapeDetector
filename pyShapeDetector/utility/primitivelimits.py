#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:01:30 2023

@author: ebernardes
"""
import numpy as np

class PrimitiveLimits:
    
    def check_compatibility(self, primitive):
        if self.args is None:
            return 
        for attribute in self.attribute:
            if not hasattr(primitive, attribute):
                raise ValueError(f"Primitive of type {primitive.name} does "
                                 f"not have attribute {attribute}.")
                
    def check(self, shape):
        if self.args is None:
            return True
        for arg in self.args:
            test_value = getattr(shape, arg['attribute'])
            if (func := arg['func']) is not None:
                test_value = func(test_value)
            if not (arg['limits'][0] <= test_value <= arg['limits'][1]):
                return False
            
        return True
        
    @property
    def func(self):
        if self.args is None:
            return None
        return [a['func'] for a in self.args]
    
    @property
    def attribute(self):
        if self.args is None:
            return None
        return [a['attribute'] for a in self.args]
    
    @property
    def limits(self):
        if self.args is None:
            return None
        return [a['limits'] for a in self.args]
    
    
    @property
    def args_list(self):
        # func, attribute, limit_type, value = zip(self.args_list)
        return self.func, self.attribute, self.value
    
    def __add__(self, limits_other):
        assert isinstance(limits_other, PrimitiveLimits)
        
        limits_sum = PrimitiveLimits(None)
        limits_sum.args = self.args + limits_other.args
        return limits_sum            
    
    def __init__(self, args):
        
        self.args = []
        
        if args is None:
            self.args = None
        elif isinstance(args, PrimitiveLimits):
            self.args = args.args
        
        else:
            if type(args) == tuple or \
                (type(args) == list and not isinstance(args[0], (list, tuple))):
                args = [args]
            
            for arg in args:
                
                if len(arg) == 3:
                    func, attribute, limits = arg
                    if not callable(func):
                        raise ValueError(f"{func} is not callable")
                            
                elif len(arg) == 2:
                    attribute, limits = arg
                    func = None
                    
                else:
                    raise ValueError("Limits are defined by 2 or 3 attributes: "
                                     "(func), attribute, limits.")
                    
                if not isinstance(limits, (tuple, list)) or len(limits) != 2:
                    raise ValueError("limits must be a list or tuple of 2 "
                                     f"elements, got {limits}")
                    
                if limits[0] != None and limits[1] != None:
                    raise ValueError(f"Limit ranges cannot all be None, got {limits}.")
                    
                if limits[0] is None:
                    limits[0] = -np.inf
                if limits[1] is None:
                    limits[1] = np.inf
                
                limits = sorted(limits)
                
                if type(attribute) is not str:
                    raise ValueError("attribute must be a string.")
                    
                limits_dict = {
                    'func': func,
                    'attribute': attribute,
                    'limits': limits
                    }
                    
                self.args.append(limits_dict)