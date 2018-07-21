import multiprocessing as mp
import numpy as np
import scipy as sp
import functools
import itertools as it

def w_pool(func, targ, shared_kwargs=None, multiplier_kwargs=None):
    '''Pooler. Casts a dict of shared_kwargs to all jobs, runs a list of dicts multplier_kwargs as multiple, separate batches.'''
    new_func = func if shared_kwargs is None else functools.partial(func, **shared_kwargs)
    func_list = [new_func]
    
    if multiplier_kwargs is not None:
        func_list = [] # reset the stack of functions to run
        for kwarg_dict in multiplier_kwargs:
            func_list += [functools.partial(new_func, **kwarg_dict)]
    
    with mp.Pool(processes = max(1,mp.cpu_count()-1)) as pool:
        results_list = [pool.map(f, targ) for f in func_list]
    
    results = list(it.chain.from_iterable(results_list))
            
    return results
        