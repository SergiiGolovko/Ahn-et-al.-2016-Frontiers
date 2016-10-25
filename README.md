### Description
* This repository contains the replication of Ahn et al. 2016 Frontiers paper.
* To run (in python 3):
	* copy the cocaineData_frontiers.txt file (either from Dr. Ahn's website or from Input folder here) to Input folder in the working directory
	* make sure that you have all dependencies (see below) installed
	* change the parameters in globals.py
	* run main.py
* There are minor differences in the results caused mainly by two reasons:
	* sklearn (python) versus glmnet (R) - different libraries for running Logistic Regression with lasso penalty. 
		* they use slightly different methods to fit the model
		* different randomization for cross validation splits generation
		* the intercept is not regularized in glmnet, but regularized in sklearn.
		* performance - Glmnet needs to fit the model nfolds+1 times, does not depend of the number of different values of lambda to be tuned. In contrast sklearn fits the model nfolds+k times, where k=100 is the number of different values of C (proportional to 1/lambda) to be fitted. This results in slow performance of sklearn versus glmnet.  
	* scaling - scaling is done differently in R and python because of using different formulas for standard error (biased vs unbiased estimate of standard deviation)

### Dependencies
* hyperopt - To install run: pip install git+git://github.com/hyperopt/hyperopt.git@master
* progressbar - To install run: pip install git+git://github.com/WoLpH/python-progressbar@master

### Folders
* Input - Contains cocaineData_frontiers.txt file
* Figures - Contains Figure 1 , Figure 2 (both for N_FOLD_SPLITS = 1000, the same as in the paper) and Figure 4 (for N_REPLICATIONS = 100, N_FOLD_SPLITS = 10,  in the paper N_REPLICATIONS = 1000, N_FOLD_SPLITS = 100)

### Files
* main.py - main function, preprocessing data by scaling (!!! note that scaling is different from R, as standard errors computed differently by default - biased vs unbiased estimation of unbiased estimation) calling estimation function from compute.py.  
* compute.py - utils to make estimation on a single train/test split (replicating figures 1 and 2) or on series of test/train replications (replicating figure 4).
* tune_params.py - hyperopt and gridsearch utils for hyperparameters tunning for lasso regression. 
* split_utils.py - utils for splitting entire data set into train/test sets as well creating cross validation splits. 
* plot_utils.py - utils for plotting roc curves, auc scores hystograms and beta coefficients (with error bars).
* globals.py - global parameters.
 
