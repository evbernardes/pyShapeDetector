#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:36:00 2024

@author: ebernardes
"""

import pytest
import copy
import numpy as np
from pyShapeDetector.utility import combine_indices_to_remove


def test_combine_indices_to_remove():
    assert combine_indices_to_remove([[0], [0]]) == [0, 1]
    assert combine_indices_to_remove([[0, 1], [0]]) == [0, 1, 2]
    assert combine_indices_to_remove([[0], [0, 1]]) == [0, 1, 2]
    assert combine_indices_to_remove([[0, 3], [1]]) == [0, 2, 3]
    assert combine_indices_to_remove([[0, 3], [1, 3]]) == [0, 2, 3, 5]

    num_elements = 100
    num_lists = 5
    num_indices = 5

    input_elements = np.random.random(num_elements).tolist()
    input_idxs = []
    N = 0
    for i in range(num_lists):
        lim = num_elements - N
        idx = np.random.randint(0, lim, num_indices).tolist()
        idx = sorted(list(set(idx)))
        N += len(idx)
        input_idxs.append(idx)

    indices_combined = combine_indices_to_remove(input_idxs)
    assert sum([len(idx_group) for idx_group in input_idxs]) == len(indices_combined)
    assert len(set(indices_combined)) == len(indices_combined)
    assert max(indices_combined) < len(input_elements)

    test1 = copy.deepcopy(input_elements)
    test2 = copy.deepcopy(input_elements)

    for idx in input_idxs:
        for i in idx[::-1]:
            test1.pop(i)
    for j in indices_combined[::-1]:
        test2.pop(j)

    assert test1 == test2
