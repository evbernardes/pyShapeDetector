#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:01:30 2023

@author: ebernardes
"""
import numpy as np

class PrimitiveLimits:
    """
    Class to test if a shape's model is within desired limits.
    
    To initialize it, a desired attribute and the desired bounds are needed.
    First, define limits instance and primitive:
    
    >>> from pyShapeDetector.primitives import Sphere, Plane
    >>> from pyShapeDetector.utility import PrimitiveLimits
    >>> attribute = 'radius'  # a string defining the attribute
    >>> bounds = [0, 2]  # a tuple or list defining the bounds
    >>> limits = PrimitiveLimits([attribute, bounds])

    This instance will test if a shape has a `radius` is in the closed interval
    `[0, 2]`. In order to apply the check:
    
    >>> sphere1 = Sphere.from_center_radius([0, 0, 0], 1.5)
    >>> limits.check(sphere1)
    True
    >>> limits.check(sphere2)
    >>> sphere2 = Sphere.from_center_radius([0, 0, 0], 3)
    False
    
    Alternatively, a function can also be given:
    >>> func = lambda x: x ** 2
    >>> limits2 = PrimitiveLimits([func, attribute, bounds])
    
    In this case, the function will be applied to the attribute before testing.
    
    Check an uncompatible primitive gives an error:
    >>> plane = Plane.random()
    >>> limits.check(plane)
    AttributeError: 'Plane' object has no attribute 'radius'
    
    To check compatibility, you can use:
    >>> limits.check_compatibility(plane)
    False
    >>> limits.check_compatibility(sphere1)
    True
    
    If `None` is given as a lower bound, negative infinite is supposed:
    >>> limits = PrimitiveLimits(['radius', [None, 2]])  # any radius smaller than 2
    
    If `None` is given as an upper bound, positive infinite is supposed:
    >>> limits = PrimitiveLimits(['radius', [1, None]])  # any radius bigger than 1
    
    If both values are `None`, throws an exception:
    >>> limits = PrimitiveLimits(['radius', [1, None]])
    ValueError: Limit ranges cannot all be None, got [None, None].
    
    Attributes
    ----------
    func
    attribute
    bounds
    args_list
        
    Methods
    -------
    __init__
    check_compatibility
    check
    __add__
    """
        
    @property
    def func(self):
        """ List with all functions. """
        if self.args is None:
            return None
        return [a['func'] for a in self.args]
    
    @property
    def attribute(self):
        """ List with all attribute. """
        if self.args is None:
            return None
        return [a['attribute'] for a in self.args]
    
    @property
    def bounds(self):
        """ List with all bounds. """
        if self.args is None:
            return None
        return [a['bounds'] for a in self.args]
    
    @property
    def args_list(self):
        """ Tuple of lists with functions, attributes and bounds. """
        # func, attribute, limit_type, value = zip(self.args_list)
        return self.func, self.attribute, self.bounds
    
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
                    func, attribute, bounds = arg
                    if not callable(func):
                        raise ValueError(f"{func} is not callable")
                            
                elif len(arg) == 2:
                    attribute, bounds = arg
                    func = None
                    
                else:
                    raise ValueError("Limits are defined by 2 or 3 attributes: "
                                     "(func), attribute, limits.")
                    
                if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                    raise ValueError("limits must be a list or tuple of 2 "
                                     f"elements, got {bounds}")
                    
                if bounds[0] is None and bounds[1] is None:
                    raise ValueError(f"Limit ranges cannot all be None, got {bounds}.")
                    
                if bounds[0] is None:
                    bounds[0] = -np.inf
                if bounds[1] is None:
                    bounds[1] = np.inf
                
                bounds = sorted(bounds)
                
                if type(attribute) is not str:
                    raise ValueError("attribute must be a string.")
                    
                limits_dict = {
                    'func': func,
                    'attribute': attribute,
                    'bounds': bounds
                    }
                    
                self.args.append(limits_dict)
                
    def check_compatibility(self, primitive):
        """ Checks if primitive is compatible and has all attributes defined
        in PrimitiveLimits instace.

        Parameters
        ----------
        primitive : Primitive instance
            Shape to be tested
            
        Return
        ------
        boolean
        """
        if self.args is None:
            return True
        
        for attribute in self.attribute:
            if not hasattr(primitive, attribute):
                return False
        
        return True
                
    def check(self, shape):
        """ Checks if primitive's paremeters are within desired bounds.

        Parameters
        ----------
        primitive : Primitive instance
            Shape to be tested
            
        Return
        ------
        boolean
        """
        if self.args is None:
            return True
        for arg in self.args:
            test_value = getattr(shape, arg['attribute'])
            if (func := arg['func']) is not None:
                test_value = func(test_value)
            if not (arg['bounds'][0] <= test_value <= arg['bounds'][1]):
                return False
            
        return True
    
    def __add__(self, limits_other):
        """ Fuse args of PrimitiveLimits instances. """
        if not isinstance(limits_other, PrimitiveLimits):
            raise TypeError(f"unsupported operand type(s) for +: '{limits_other}' and PrimitiveLimits")
        
        limits_sum = PrimitiveLimits(None)
        limits_sum.args = self.args + limits_other.args
        return limits_sum