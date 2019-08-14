import multiprocessing as mp
import itertools as it
import functools as ft
import pickle
import sys
import numpy as np
import pandas as pd
import time
import sklearn
import sklearn.preprocessing as pre
import scipy.sparse as sp

dat = pd.read_csv('./cord_blood_kinases.csv', sep=',', header=0, index_col=0);

def get_cell_sets(row, oe_csr):
    return oe_csr[row['lower']:row['upper']].sum(axis=0)

def first_candidates(cells, cell_sets, min_shared_cells):
    count_filter = cell_sets.apply(len) > min_shared_cells
    return list(map(lambda x: frozenset([x]), ((cells[count_filter])))), {frozenset([x]):y for x,y in cell_sets[count_filter].to_dict().items()}

def intersector(tuple_of_candidates, tuple_of_sets):
    return ft.reduce(lambda x,y: x.union(y), tuple_of_candidates), tuple_of_sets[0] & tuple_of_sets[1]

def cell_set_getter(input_list, cell_sets):
    for i in input_list:
        yield cell_sets[i]
        
def make_gener(left, right, min_shared_cells, cell_sets, q):
    left_gen = cell_set_getter(left, cell_sets)
    right_gen = cell_set_getter(right, cell_sets)        
    gener = ((x, y) for x, y in map(intersector, *(zip(left, right),zip(left_gen, right_gen))) if len(y)>min_shared_cells)
    q.put(dict(list(gener)))
    return 

def take_from(gener, size):
    yield it.takewhile(lambda x: len(x) > 0, list(it.islice(gener, size)))

def parallel_combo_generator(chunk, full_list, k, q):
    local_list = list(filter(lambda x: len(x[0]|x[1]) == k, it.product(chunk, full_list)))
    if(len(local_list) == 0):
        q.put(None)
    else:
        left, right = zip(*local_list)
        q.put((len(local_list),iter(left), iter(right)))
    return
        
def pickle_cells(cell_sets, k):
    '''These files are gonna be decently big. Do not want to keep them in memory.'''
    with open('cell_sets_' + str(k) + '.pickle', 'wb') as f:
        pickle.dump(cell_sets, f, pickle.HIGHEST_PROTOCOL)
        
def zip_helper(gener_slice, q):
    q.put(list(gener_slice))
    return
    
def fast_gather_gene_sets(dat, min_shared_cells = 100, min_percent_cells = None, max_cluster_size = sys.maxsize):
    st = time.time()
    begin = st
    cores = max(mp.cpu_count()-1, 1)
    
    total_cells = dat['barcode'].nunique()
    
    if(min_percent_cells is not None):
        min_shared_cells = min_percent_cells * total_cells
    
    cell_id_dict = {y:x for x,y in enumerate(dat['symbol'].unique())}
    dat['symbol'] = dat['symbol'].map(cell_id_dict)
    cells = dat['symbol'].unique()
    
    barcode_id_dict = {y:x for x,y in enumerate(dat['barcode'].unique())}
    dat['barcode'] = dat['barcode'].map(barcode_id_dict)
    
    cell_sets = dat.groupby('symbol')['barcode'].apply(set)
    
    en = time.time()
    
    print('Formatted data in ' + str(en-st) + ' seconds')
    
    cells, cell_sets = first_candidates(cells, cell_sets, min_shared_cells)
    
    print(str(len(cells)) + ' genes made have > ' + str(min_shared_cells) + ' cells')
    
    k = 2
    n = len(cells)
    
    pickle_cells(cell_sets, k)
    
    while(len(cells) > 0 and k < max_cluster_size):
        st = time.time()
        
        q = mp.JoinableQueue()
        
        candidates_output = []
        candidate_iter = iter(cell_sets.keys())
        kwarg_dict={'k':k,'full_list':cell_sets.keys(),'q':q}
        
        for i in range(cores-1):
            p = mp.Process(target=parallel_combo_generator, args=(it.islice(candidate_iter, n//cores),), kwargs=kwarg_dict)
            p.start()

        for i in range(1):
            p = mp.Process(target=parallel_combo_generator, args=(it.islice(candidate_iter, n//cores),), kwargs=kwarg_dict)
            p.start()
            
        for i in range(cores):
            t = q.get()
            candidates_output += [t] if t is not None else []
            q.task_done()
            
        while(not q.empty()):
            q.get()
            q.task_done()
            
        q.join()
        q.close()
        
        if(len(candidates_output) == 0):
            print('No new candidates!')
            print('Terminated! Total run time: ' + str(en - begin) + ' seconds')
            break
            
        lengths, candidates_left, candidates_right = zip(*candidates_output)
        candidates_left = it.chain.from_iterable(candidates_left)
        candidates_right = it.chain.from_iterable(candidates_right)
        cand_len = sum(lengths) 
        
        q = mp.JoinableQueue()
        
        kwarg_dict={'min_shared_cells':min_shared_cells,'cell_sets':cell_sets,'q':q}

        for i in range(cores-1):
            p = mp.Process(target=make_gener, args=(list(it.islice(candidates_left, cand_len//cores)), list(it.islice(candidates_right, cand_len//cores))), kwargs=kwarg_dict)
            p.start()

        for i in range(1):
            p = mp.Process(target=make_gener, args=(list(candidates_left), list(candidates_right)), kwargs=kwarg_dict)
            p.start()

        en = time.time()
        print('Finished launching processes in: '+ str(time.time()-st) + ' seconds')
            
        output = []
        for i in range(cores):
            t = q.get()
            output += [t] if t is not None else []
            q.task_done()
            
        print('Finished merging cells in: ' + str(time.time() - en) + ' seconds')
            
        if(len(output) == 0):
            print('No results from parallel output!')
            print('Terminated! Total run time: ' + str(en - begin) + ' seconds')
            break
            

        cell_sets = ft.reduce(lambda x,y: {**x, **y}, output)
        
        k+= 1
        n = len(cell_sets)
        
        en = time.time()
        
        print('Found ' + str(n) + ' remaining gene clusters with > ' + str(min_shared_cells) + ' of size: ' +str(k-1))
        print('Iteration took: ' + str(en-st) + ' seconds')
        print('Running time: ' + str(en - begin) + ' seconds')
        
        if(n == 0):
            print('Terminated! Total run time: ' + str(en - begin) + ' seconds')
            break
        else:
            print('Pickling!')
            pickle_cells(cell_sets, k-1)
            
fast_gather_gene_sets(dat, min_percent_cells = 0.07)
        