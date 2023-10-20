#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:01:30 2023

@author: ebernardes
"""
class PrimitiveLimits:
    
    def check_compatibility(self, primitive):
        for attribute in self.attribute:
            if not hasattr(primitive, attribute):
                raise ValueError(f"Primitive of type {primitive.name} does "
                                 f"not have attribute {attribute}.")
                
    def check(self, shape):
            for arg in self.args:
                test_value = getattr(shape, arg['attribute'])
                if arg['func'] is not None:
                    test_value = arg['func'](test_value)
                if (arg['limit_type']=='max' and test_value > arg['value']) or \
                    (arg['limit_type']=='min' and test_value < arg['value']):
                    return False
            return True
        
    @property
    def func(self):
        return [a['func'] for a in self.args]
    
    @property
    def attribute(self):
        return [a['attribute'] for a in self.args]
    
    @property
    def limit_type(self):
        return [a['limit_type'] for a in self.args]
    
    @property
    def value(self):
        return [a['value'] for a in self.args]
    
    
    @property
    def args_list(self):
        # func, attribute, limit_type, value = zip(self.args_list)
        return self.func, self.attribute, self.limit_type, self.value
    
    def __init__(self, args):
        
        self.args = []
        
        if type(args) == tuple or \
            (type(args) == list and type(args[0]) != list):
            args = [args]
        
        for arg in args:
            if len(arg) == 4:
                func, attribute, limit_type, value = arg
                if not callable(func):
                    raise ValueError(f"{func} is not callable")
                        
            elif len(arg) == 3:
                attribute, limit_type, value = arg
                func = None
                
            else:
                raise ValueError("Limits are defined by 3 or 4 attributes: "
                                 "(func), attribute, limit_type, value.")
            
            if type(attribute) is not str:
                raise ValueError("attribute must be a string.")
                
            if type(limit_type) is not str:
                raise ValueError("limit_type must be a string.")
                
            if limit_type not in ('max', 'min'):
                raise ValueError("limit_type must be a 'max' or 'min'.")
                
            possible_types = {'max', 'min'}
            if limit_type not in possible_types:
                raise ValueError(f"limit_type must be in set {possible_types},"
                                 f" got {limit_type}.")
                
            limit = {
                'func': func,
                'attribute': attribute,
                'limit_type': limit_type,
                'value': value
                }
                
            self.args.append(limit)