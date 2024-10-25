#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:50:14 2024

@author: ebernardes
"""
import functools
import copy
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


def combine_indices_to_remove(idx_groups):
    """
    Combine multiple lists of indices to be removed.

    For example, suppose you have:
        input = ['a', 'b', 'c', 'd', 'e', 'f']

    And you want to remove the elements at indices [0, 3]. You end up with:
        step1 = [_, 'b', 'c', _, 'e', 'f']  # removed input[0] and input[3]

    Let's say you later also remove the elements [1, 3] from this new list:
        step2 = ['b', _, 'e', _]  # removed step1[1] and step1[3]

    If you want to remove the same elements at a single step, you must remove
    the elements at indices [0, 2, 3, 5].

    This function calculates this modified and combined indices list.

    Parameters
    ----------
    idx_groups: list
        List containing sublists of integers.

    Returns
    -------
    list of integers

    """
    if not isinstance(idx_groups, list):
        raise ValueError(f"Expected list of lists, got {type(idx_groups)}.")

    for indices in idx_groups:
        if not isinstance(indices, list):
            raise ValueError(f"Expected list of lists, got {type(indices)}.")

        for idx in indices:
            if not isinstance(idx, int):
                raise ValueError(f"Expected integers, got {idx}.")

    indices = []
    if len(idx_groups) > 0:
        N = sum([len(idx_group) for idx_group in idx_groups])
        lim = max([i for idx_group in idx_groups for i in idx_group])
        dummy_values_1 = list(range(N + lim))
        dummy_values_2 = copy.deepcopy(dummy_values_1)

        for idx_group in idx_groups:
            for i in idx_group[::-1]:
                elem = dummy_values_1.pop(i)
                indices.append(dummy_values_2.index(elem))

    return sorted(indices)
