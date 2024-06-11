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
            array = args[0]
            args = args[1:]
            array_split = np.array_split(array, cores)
            
            def func_internal(i, data, *args, **kwargs):
                values = func(array_split[i], *args, **kwargs)
                data[i] = values
                # print(f'{i} worked!')
                
            manager = Manager()    
            data = manager.dict()
            processes = [Process(target=func_internal, args=(i, data)) for i in range(cores)]
        
            for process in processes:        
                process.start()
        
            for process in processes:        
                process.join()
            
            try:
                return np.hstack([data[i] for i in range(cores) if i in data])
            except KeyError:
                raise RuntimeError(f'Only {len(data)} out of {cores} processes '
                                   'worked, possibly memory problem.')

        return wrapper
    return decorator_parallelize