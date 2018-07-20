import multiprocessing as mp
import numpy as np
import itertools as it
from sklearn import linear_model

def fs_lasso_cv(X,y,feat_list, n_alphas=1000, cv=10, tol=0.00001, max_iter=10000, hard_shrink=None):
    '''Wrapper function to build a LassoCV model from sklearn and return important features'''
    
    lcv = linear_model.LassoCV(n_jobs = max(1, mp.cpu_count()-1), n_alphas=n_alphas, cv=cv, tol=tol, max_iter=max_iter)
    coefs = lcv.fit(X,y).coef_

    # force shrinkage to zero if hard_shrink is provided
    if hard_shrink is not None: np.place(coefs, coefs < hard_shrink, 0)
    
    selected_feats = list(it.compress(feat_list, coefs))
    
    return selected_feats

def fs_lars_cv(X,y,feat_list, n_alphas=1000, cv=10, max_iter=1000, hard_shrink=None):
    '''Wrapper function to build a LarsCV model from sklearn and return important features'''
    
    lcv = linear_model.LarsCV(n_jobs = max(1, mp.cpu_count()-1), max_n_alphas=n_alphas, cv=cv, max_iter=max_iter)
    coefs = lcv.fit(X,y).coef_

    # force shrinkage to zero if hard_shrink is provided
    if hard_shrink is not None: np.place(coefs, coefs < hard_shrink, 0)
    
    selected_feats = list(it.compress(feat_list, coefs))
    
    return selected_feats