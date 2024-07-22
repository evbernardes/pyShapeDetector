#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:50:14 2024

@author: ebernardes
"""
import numpy as np
from multiprocessing import Manager, Process


def parallelize(cores=2):
    def decorator_parallelize(func):
        def wrapper(*args, **kwargs):
            if cores == 1:
                return func(*args, **kwargs)
            
            array = args[0]
            args = args[1:]
            array_split = np.array_split(array, cores)

            def func_internal(i, data, *args, **kwargs):
                values = func(array_split[i], *args, **kwargs)
                data[i] = values
                # print(f'{i} worked!')

            manager = Manager()
            data = manager.dict()
            processes = [
                Process(target=func_internal, args=(i, data)) for i in range(cores)
            ]

            for process in processes:
                process.start()

            for process in processes:
                process.join()

            try:
                return np.hstack([data[i] for i in range(cores) if i in data])
            except KeyError:
                raise RuntimeError(
                    f"Only {len(data)} out of {cores} processes "
                    "worked, possibly memory problem."
                )

        return wrapper

    return decorator_parallelize


def _set_and_check_3d_array(input_array, name="array", num_points=None):
    if input_array is None or len(input_array) == 0:
        return np.array([])

    array = np.asarray(input_array)
    if array.shape == (3,):
        array = np.reshape(array, (1, 3))
    elif array.shape[1] != 3:
        raise ValueError(
            "Invalid shape for {name}, must be a single"
            " point or an array of shape (N, 3), got "
            f"{array.shape}"
        )
    if num_points is not None and len(array) != num_points:
        raise ValueError(
            f"Expected shape of array is ({num_points}, 3), got {array.shape}."
        )

    return array
