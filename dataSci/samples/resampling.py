def dg_bagg(X,y,model,n_samples = None, allow_repeats = True, random_seed = None):
    '''Wrapper for sklearn.utils resample to generate a bootstrapped dataset, both X and y'''

    y_hats = model.predict(X)
    resids = y - y_hats
    
    # shuffle the residuals and draw matching samples from X and y_hats
    X_new, y_hats_new, resids_new = utils.resample(X,y_hats,utils.shuffle(resids),n_samples=n_samples,random_state=random_seed,replace=allow_repeats)
    y_new = y_hats_new + resids_new
    
    return(X_new, y_new)
    