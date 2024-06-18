#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:00:58 2024

@author: ebernardes
"""
import copy
from .open3d_decorators import args_to_open3d, to_open3d_and_back


def link_to_open3d_geometry(original_class):
    def decorator(cls):
        setattr(cls, "__open3d_class__", original_class)

        # create new __init__ that generates an internal variable
        @args_to_open3d
        def new_init(self, *args, **kwargs):
            self._open3d = original_class(*args, **kwargs)

        setattr(cls, "__init__", new_init)

        # removing two last subclasses, 'object' and 'pybind11_object'
        for subclass in original_class.__mro__[:-2]:
            for attr_name, attr_value in subclass.__dict__.items():
                # checking if already exists ir Open3D_Geometry or its child
                if (
                    attr_name == "__init__"
                    or attr_name in cls.__mro__[0].__dict__
                    or attr_name in cls.__mro__[1].__dict__
                ):
                    continue

                # if attr_name not in cls.__mro__[0].__dict__:
                if callable(attr_value):
                    setattr(cls, attr_name, to_open3d_and_back(attr_value))

                elif isinstance(attr_value, property):
                    # Apply decorators to the original getter and setter functions
                    getter = to_open3d_and_back(attr_value.fget)
                    setter = args_to_open3d(attr_value.fset)

                    # Create a new property with the decorated getter and setter functions
                    new_property = property(
                        getter, setter, attr_value.fdel, attr_value.__doc__
                    )

                    setattr(cls, attr_name, new_property)
                else:
                    setattr(cls, attr_name, attr_value)
        return cls

    return decorator


class Open3D_Geometry:
    __open3d_class__ = None

    @property
    def as_open3d(self):
        return self._open3d

    @property
    def bbox(self):
        return self.get_axis_aligned_bounding_box()

    @property
    def bbox_bounds(self):
        bbox = self.bbox
        return bbox.min_bounds, bbox.max_bounds

    def __copy__(self, *args):
        _open3d = copy.copy(self.as_open3d)
        return type(self)(_open3d)

    def __deepcopy__(self, *args):
        _open3d = copy.deepcopy(self.as_open3d)
        return type(self)(_open3d)
        # return "test"

    def __repr__(self):
        return "Open3D-Compatible " + self._open3d.__repr__()

    @classmethod
    def is_instance_or_open3d(cls, elem):
        return isinstance(elem, (cls, cls.__open3d_class__))
