from sklearn.metrics import roc_auc_score, log_loss
from split_utils import *
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from globals import *
from sklearn.grid_search import GridSearchCV

# global variables for objective function
_estimator = []
_x = []
_y = []
_cv = []

# space of parameters for hyperopt
hparam_space_lasso = {
    'C': hp.loguniform('C', np.log(0.001), np.log(10000))
}

# space of parameters for grid search
gparam_space_lasso = {
    'C': [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 0.8, 0.9, 1, 2, 5, 10, 100, 1000]
}


def hyperopt_parameters(estimator, X_train, y_train, rs):
    """Returns the best set of parameters

    Parameters
    ----------
    estimator : estimator for which parameters are tuned
    X_train : data frame with features
    y_train : data frame with labels

    Return
    ------
    best_parameter : dictionary of best parameters
    best_score : float
    """

    # setup global parameters to be passed to objective function
    global _estimator, _cv, _x, _y
    _estimator = estimator
    _x, _y = X_train, y_train
    _cv = create_KFolds(y_train, rs)

    # create trials
    trials = Trials()

    # find best parameters
    best_params = fmin(objective,
                       hparam_space_lasso,
                       algo=tpe.suggest,
                       trials=trials,
                       max_evals=HYPEROPT_MAX_EVALS)
    best_params = space_eval(hparam_space_lasso, best_params)

    return best_params


def objective(param):
    """ Returns the mean value of log loss over cross validation splits in _cv

    Parameters
    ----------
    param: parameters for the model _estimator

    Return
    ------

    dictionary loss: value of log loss for value of parameters param
               status: STATUS_OK
    """

    global _estimator, _cv, _x, _y

    # do cross validation
    scores = my_cross_validation(_estimator, _x, _y, _cv, param)

    # mean score across all folds
    mean_score = sum(scores) / len(scores)

    return {'loss': mean_score, 'status': STATUS_OK}


def my_cross_validation(estimator, x, y, cv, param):
    """ Returns the scores for each train/validation split

    Parameters
    ----------
    estimator: estimator
    x: data frame with features
    y: data frame with labels
    cv: cross validation folds
    param: parameters for estimator

    Return
    ------
    scores: list of scores for each cross validation split
    """

    # initialize scores
    scores = []

    # set parameters
    estimator.set_params(**param)

    for train_ind, test_ind in cv:
        # fit the model on training set
        iX, iy = x.values[train_ind], y.values[train_ind]
        estimator.fit(iX, iy)

        # make a prediction for test set
        iX, iy = x.values[test_ind], y.values[test_ind]
        pred_y = estimator.predict_proba(iX)

        # calculate the score
        # score = roc_auc_score(y_true=iy, y_score=pred_y[:, 1])
        score = log_loss(y_true=iy, y_pred=pred_y[:, 1])
        scores.append(score)

    return scores


def gridsearch_parameters(estimator, X_train, y_train, rs):
    """Returns the best set of parameters out of ones specified in

    Parameters
    ----------
    estimator : estimator for which parameters are tuned
    name : name of estimator
    X_train : data frame with features
    y_train : data frame with labels

    Return
    ------
    best_parameter : dictionary of best parameters
    """

    # create KFolds
    cv = create_KFolds(y_train, rs)

    # find the best set of parameters, grid search
    gscv = GridSearchCV(estimator,
                        gparam_space_lasso,
                        cv=cv,
                        scoring='roc_auc')
    gscv.fit(X_train, y_train)

    return gscv.best_params_
