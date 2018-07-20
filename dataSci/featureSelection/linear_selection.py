def fs_lasso_cv(X,y,feat_list, n_alphas=1000, cv=10, tol=0.00001, max_iter=10000):
    from multiprocessing import cpu_count
    from sklearn.linear_model import LassoCV
    import numpy as np
    from itertools import compress 

    lcv = LassoCV(n_jobs = max(1, cpu_count()-1), n_alphas=n_alphas, cv=cv, tol=tol, max_iter=max_iter)
    lcv.fit(X,y)
    return list(compress(feat_list, lcv.coef_))