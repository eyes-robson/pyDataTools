import multiprocessing as mp
import numpy as np
import itertools as it
from sklearn import linear_model

def fs_lasso_cv(X,y,feat_list, n_alphas=1000, cv=10, tol=0.00001, max_iter=10000):
    '''Wrapper function to build a LassoCV model from sklearn and return important features'''
    
    lcv = linear_model.LassoCV(n_jobs = max(1, mp.cpu_count()-1), n_alphas=n_alphas, cv=cv, tol=tol, max_iter=max_iter)
    lcv.fit(X,y)
    return list(it.compress(feat_list, lcv.coef_))

def fs_lars_cv(X,y,feat_list, n_alphas=1000, cv=10, max_iter=1000):
    '''Wrapper function to build a LarsCV model from sklearn and return important features'''
    
    lcv = linear_model.LarsCV(n_jobs = max(1, mp.cpu_count()-1), max_n_alphas=n_alphas, cv=cv, max_iter=max_iter)
    lcv.fit(X,y)
    return list(it.compress(feat_list, lcv.coef_))