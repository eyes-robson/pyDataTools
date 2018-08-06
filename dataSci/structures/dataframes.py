import multiprocessing as mp
import numpy as np
import pandas as pd
import itertools as it
import heapq
import functools as ft

def heap_rows(df, k, parallel = True, verbose=False):
    """Return the top k entries for each row. Similar to pd.DataFrame.transpose followed by pd.nlargest """
    available_procs = max(1, mp.cpu_count()-1) if parallel else 1
    if(verbose):
        print('Using ' + str(available_procs) + ' cores')

    def get_top_k(df, slice, queue):
        pop_k = lambda x: [heapq.heappop(x) for i in range(k)]
        listify = lambda x: x.values.tolist()

        df_slice = df.iloc[slice]
        ind = df_slice.index
        out = listify(df_slice)
        list(map(heapq.heapify, out))
        queue.put(pd.DataFrame(list(map(pop_k, out)), index=ind))
        return

    queue = mp.JoinableQueue()
    out = []

    for i in range(available_procs):
        if(verbose):
            print('Launching process '+ str(i))
        kw = {'df':df,'slice': range(i, df.shape[0], available_procs), 'queue':queue}
        proc = mp.Process(target = get_top_k, kwargs=kw)
        proc.start()

    for i in range(available_procs):
        if(verbose):
            print('Closing process '+ str(i))
        out.append(queue.get())
        queue.task_done()

    queue.join()
    queue.close()

    return pd.concat(out, axis=0)
