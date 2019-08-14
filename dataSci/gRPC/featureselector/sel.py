import multiprocessing as mp
import numpy as np
import itertools as it
from sklearn import linear_model

def fs_lasso_cv(X,y,feat_list, n_alphas=1000, cv=3, tol=0.00001, max_iter=10000, hard_shrink=None):
    '''Wrapper function to build a LassoCV model from sklearn and return important features'''

    lcv = linear_model.LassoCV(n_jobs = max(1, mp.cpu_count()-1), n_alphas=n_alphas, cv=cv, tol=tol, max_iter=max_iter)
    coefs = lcv.fit(X,y).coef_

    # force shrinkage to zero if hard_shrink is provided
    if hard_shrink is not None: np.place(coefs, np.abs(coefs) < hard_shrink, 0)

    selected_feats = list(it.compress(feat_list, coefs))

    return selected_feats

def fs_lars_cv(X,y,feat_list, n_alphas=1000, cv=10, max_iter=1000, hard_shrink=None):
    '''Wrapper function to build a LarsCV model from sklearn and return important features'''

    lcv = linear_model.LarsCV(n_jobs = max(1, mp.cpu_count()-1), max_n_alphas=n_alphas, cv=cv, max_iter=max_iter)
    coefs = lcv.fit(X,y).coef_

    # force shrinkage to zero if hard_shrink is provided
    if hard_shrink is not None: np.place(coefs, np.abs(coefs) < hard_shrink, 0)

    selected_feats = list(it.compress(feat_list, coefs))

    return selected_feats


def fs_ardr(X,y,feat_list, max_iter=300, tol=0.001, hard_shrink=.01):
    '''Wrapper function to build a ARDRegression model from sklearn and return important features'''
    '''Automatic Relevance Determination Regression (ARDR) can be thought of as a Sparse Bayesian Ridge Regression'''
    '''http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html'''

    bridge = linear_model.ARDRegression(tol=tol, n_iter=max_iter)
    coefs = bridge.fit(X,y).coef_

    # use hard_shrink as the cutoff for 'significant' features
    np.place(coefs, np.abs(coefs) < hard_shrink, 0)

    selected_feats = list(it.compress(feat_list, coefs))

    return selected_feats
