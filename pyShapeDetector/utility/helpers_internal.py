#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:50:14 2024

@author: ebernardes
"""
import functools
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


def accept_one_or_multiple_elements(*dimensions):
    """This decorator simplifies some functions so that they can be used
    with either a single point, or a multidimensional array.

    For example, since it is applied to the get_distances methods of the
    Primitive classes:

        accept_one_or_multiple_elements(3)
        def get_distances(self, points, *args):
            ...

    For a single point `point`, with shape (3, ), I can simplify this:

        shape.get_distances([point])[0]

    To this:

        shape.get_distances(point)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if len(args) != len(dimensions):
                raise ValueError(
                    f"Expected {len(dimensions)} arguments, got {len(args)}."
                )

            num_args = len(dimensions)
            args = list(args)
            for i in range(num_args):
                args[i] = np.asarray(args[i])

            input_ndim = [x.ndim for x in args[:num_args]]
            ndim = input_ndim[0]
            if input_ndim.count(ndim) != num_args:
                raise ValueError(
                    f"Number of dimensions should all be equal, got {input_ndim}."
                )

            if ndim == 1:
                input_nelems = 1

            elif ndim == 2:
                input_nelems = [x.shape[1] for x in args[:num_args]]
                nelems = input_nelems[0]
                if input_nelems.count(nelems) != num_args:
                    raise ValueError(
                        "Number of elements should all be equal, got "
                        f"{input_nelems}."
                    )

            else:
                raise ValueError(f"Inputs must be either 1D or 2D, got {ndim}D.")

            for i, (x, expected_dim) in enumerate(zip(args, dimensions)):
                if ndim == 1:  # Input is 1D, shape: (D,)
                    if x.shape[0] != expected_dim:
                        raise ValueError(
                            f"Input {i} dimension does not match expected D={expected_dim}. Got shape {x.shape[0]}."
                        )
                elif ndim == 2:  # Input is 2D, shape: (nelems, D)
                    if x.shape[1] != expected_dim:
                        raise ValueError(
                            f"Input {i} dimension does not match expected D={expected_dim}. Got shape {x.shape[1]}."
                        )

                # If input is 1D, reshape to (1, D)
                if x.ndim == 1:
                    args[i] = np.expand_dims(x, axis=0)  # Reshape to (1, D)

            results = func(self, *args, **kwargs)

            if ndim == 1:
                if isinstance(results, tuple):
                    # in case there are many returns, like Plane.get_projections
                    # which can return the rotation
                    results = tuple([results[0][0]] + list(results[1:]))
                else:
                    results = results[0]

            return results

        return wrapper

    return decorator
