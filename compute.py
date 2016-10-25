from sklearn.linear_model import LogisticRegression
import time
import random
from split_utils import *
from tune_params import hyperopt_parameters
from plot_utils import *
import progressbar


def main_estimation(X, y):
    """Estimates Lasso Model. If ESTIMATION_MODE = 'single' - the model is estimated for a single split only, replicating
    figures 1 and 2 in the paper; If ESTIMATION_MODE = 'replicate' - the model is estimated for N_REPLICATION splits of
    entire dataset into train and test set, replicating figure 4 in the paper; If ESTIMATION_MODE = 'both' then all of
    the above is reproduced

    Parameters
    ----------

    X: array-like data set (features)
    y: array-like labels

    Return
    ------

    """

    if CONFIG['ESTIMATION_MODE'] != 'replicate': # either single or both

        # split data set into train and test set, exactly replicated the split in the paper
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)

        # run estimation
        rep_folds_estimation(X_train, y_train, X_test, y_test)

    elif CONFIG['ESTIMATION_MODE'] != 'single':  # either single or both

        # run estimation
        rep_splits_estimation(X, y)


def rep_splits_estimation(X, y):
    """ Estimate Lasso model for N_REPLICATION splits of entire dataset into train and test set, replicating figure 4
    in the paper


    Parameters
    ----------

    X: array-like data set (features)
    y: array-like labels

    Return
    ------
    """

    print("Estimating Lasso, %d Replications in Total")

    time_start = time.time()

    # initializing nsplits random seeds
    nsplits = CONFIG['N_REPLICATIONS']
    random.seed(RANDOM_SEED)
    random_states = [random.randint(RANDOM_SEED_MIN, RANDOM_SEED_MAX) for i in range(nsplits)]

    # initializing auc scores for train and test sets
    aucs_train, aucs_test = np.zeros([nsplits, 1]), np.zeros([nsplits, 1])

    # start progress bar
    bar = progressbar.ProgressBar(max_value=nsplits).start()

    # estimate a model for each random seed - split of entire data set into train and test sets
    for i, rs in enumerate(random_states):

        # split data set into train and test set
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, rs=rs)

        # estimate a model and get predictions
        pred_y_train, pred_y_test = rep_folds_estimation(X_train, y_train, X_test, y_test, show=False)

        # calculate auc scores
        aucs_train[i], aucs_test[i] = roc_auc_score(y_train, pred_y_train), roc_auc_score(y_test, pred_y_test)

        # update progress bar
        bar.update(i+1)

    # finish progress bar
    bar.finish()

    # calculate elapsed time
    elapsed_time = time.time() - time_start

    # plot histograms
    plot_histograms(aucs_train, aucs_test)

    print("Finished Estimating Lasso, Elapsed Time %d sec." % (elapsed_time))


def rep_folds_estimation(X_train, y_train, X_test, y_test, show=True):
    """ Estimate Lasso model for N_FOLDS cross-validation splits of train set. Allows to replicate figures 1 and 2 in the
    paper, is a building block for rep_splits_estimation function

    Parameters
    ----------

    X_train: array-like train set (features)
    y_train: array-like train labels

    X_test: array-like train set (features)
    y_test: array-like train labels

    show: bool
        If True then replicate figures 1 and 2
        If False then is a building block for rep_splits_estimation function, figures 1 and 2 are not reproduced

    Return
    ------

    pred_y_train:
    pred_y_test:
    """

    if show:
        print("Estimating Lasso, Single Split of Train and Test Set")
        time_start = time.time()

    # initialize nfold random seeds
    nfolds = CONFIG['N_FOLD_SPLITS']
    random.seed(RANDOM_SEED)
    random_states = [random.randint(RANDOM_SEED_MIN, RANDOM_SEED_MAX) for i in range(nfolds)]

    # initialize betas; auc scores ad predictions for both train and test sets
    aucs_train, aucs_test = np.zeros([nfolds, 1]), np.zeros([nfolds, 1])
    preds_y_train, preds_y_test = np.zeros([nfolds, X_train.shape[0]]), np.zeros([nfolds, X_test.shape[0]])

    # create entire data set - for calculating betas
    if show:
        X, y = pd.concat((X_train, X_test)), pd.concat((y_train, y_test))
        betas = np.zeros([nfolds, X.shape[1]])

    # start progress bar
    if show:
        bar = progressbar.ProgressBar(max_value=nfolds).start()

    # estimate a model for each random seed - fold split
    for i, rs in enumerate(random_states):
        # estimate a model and get betas, auc scores and predictions
        C, auc_train, auc_test, pred_y_train, pred_y_test = estimation(X_train, y_train, X_test, y_test, rs)

        # save the results
        aucs_train[i], aucs_test[i] = auc_train, auc_test
        preds_y_train[i, :], preds_y_test[i, :] = pred_y_train[:, 1], pred_y_test[:, 1]

        # fit the model on entire data set
        if show:
            model = LogisticRegression(penalty='l1', C=C, fit_intercept=True, random_state=RANDOM_SEED)
            model.fit(X, y)
            betas[i] = model.coef_

        # update progress bar
        if show:
            bar.update(i + 1)

    # calculate elapsed time
    if show:
        elapsed_time = time.time() - time_start

    # close progress bar
    if show:
        bar.finish()

    # plot betas -> figure 1 in the paper
    if show:
        plot_betas(betas, X.columns.tolist())

    # calculate mean of predictions
    preds_y_train, preds_y_test = np.mean(preds_y_train, axis=0), np.mean(preds_y_test, axis=0)

    # plot roc curves -> figure 2 in the paper
    if show:
        plot_roc_curves(y_train, preds_y_train, y_test, preds_y_test)
        print("Finished Estimating Lasso, Elapsed Time %d sec." %(elapsed_time))

    return preds_y_train, preds_y_test


def estimation(X_train, y_train, X_test, y_test, rs):
    """ Estimating Lasso Model. For now Logistic Regression from sklearn library is used. Parameters are tuned by using
     hyperopt over HYPEROPT_MAX_EVALS draws from exp(uniform(log(0.001), log(10000))) distribution that maximizes mean
     score of log loss function (in the paper deviance is used that is equivalent to log loss function) on N_FOLDS folds
     cross-validation.

    Parameters
    ----------

    X_train: array-like train set (features)
    y_train: array-like train labels

    X_test: array-like train set (features)
    y_test: array-like train labels

    rs: int
        Random seed, used to split train set in N_FOLDS folds

    Return
    ------

    # beta: array-like list of beta coefficients
    C: double, fitted value for parameter C

    auc_train: double, auc score on train set
    auc_test: double, auc score on test set

    pred_y_train: array-like list of train set predictions
    pred_y_test: array-like list of test set predictions


    """

    # assign estimator to be lasso, set fit_intercept=True
    estimator = LogisticRegression(penalty='l1', fit_intercept=True, random_state=RANDOM_SEED)

    # find the best parameters for lasso, using hyperopt
    best_params = hyperopt_parameters(estimator, X_train, y_train, rs)

    # fit the model
    estimator.set_params(**best_params)
    estimator.fit(X_train, y_train)

    # beta coefficients
    # beta = estimator.coef_

    # auc train score
    pred_y_train = estimator.predict_proba(X_train)
    auc_train = roc_auc_score(y_true=y_train, y_score=pred_y_train[:, 1])

    # auc test score
    pred_y_test = estimator.predict_proba(X_test)
    auc_test = roc_auc_score(y_true=y_test, y_score=pred_y_test[:, 1])

    return best_params['C'], auc_train, auc_test, pred_y_train, pred_y_test
