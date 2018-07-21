import multiprocessing as mp
import numpy as np
import itertools as it
from sklearn import preprocessing as pre

def fe_polynomial(X, feat_list, deg=2, interaction_only=False, include_bias=False, include_lower=True):
    '''Wrapper for sklearn.preprocessing PolynomialFeatures to create higher order interaction terms. Also gives names.'''
    poly = pre.PolynomialFeatures(degree=deg, interaction_only=interaction_only, include_bias=include_bias)
    data_out = poly.fit_transform(X)
    feat_list_out = poly.get_feature_names(feat_list)
    
    # in case we want to remove lower order terms from output
    if not include_lower:
        # get the column number of the last lower degree interaction term
        len_feat_list_lower = len(list(it.combinations_with_replacement(feat_list, deg-1))) + len(feat_list)
        
        if include_bias: len_feat_list_lower += 1
        
        data_out = data_out[len_feat_list_lower:,:]
        feat_list_out = feat_list_out[len_feat_list_lower:]
        
    return (data_out,feat_list_out)
    