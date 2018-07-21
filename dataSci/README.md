The dataSci folder here contains several useful wrappers and pipelining tools for data analysis & visualization, model fitting, and other statistical things. 

The folder contains several subdirectories, each designed to manipulate data in some way.
**samples/** (aka observations) - deals with removing and synthesizing new examples (e.g. diagnostics for outliers or bagging)
**features/** (aka covariates) - deals with subsetting, generating, or engineering new information about existing examples (e.g. interaction terms or stepwise regression)
**parallel/** - useful wrappers and decorators for parallelizing computation
**pipeline/** - useful wrappers for using sci-kit learn's Pipeline module with functions from this repo

A useful page for inspiration and intuition ~
http://scikit-learn.org/stable/data_transforms.html

**samples/** has tools for outlier removal and bagging  

**features/** is subdivided into **sel** (selection) which covers reduction and cleaning of features, **eng** (engineering) which covers expansion and generation of new features as well as **imp** (imputation) which has missing data discovery and patching tools