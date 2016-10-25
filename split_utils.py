from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from globals import *


def create_train_test_split(X, y, rs=None):
    """ Returns train test split, with CONFIG['TEST_SIZE'] samples being in test set

    Parameters
    ----------

    X: array-like data set of features
    y: array-like set of labels
    rs: int, optional
        If rs = None, deterministic split is returned
        If rs is integer, random stratified split is returned

    Return
    ------

    X_train: array-like train set (features)
    X_test: array-like test set (features)
    y_train: array-like train labels
    y_test: array-like test labels
    """

    # create random split
    if rs:
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=CONFIG['TEST_SIZE'],
                                                            stratify=y,
                                                            random_state=rs)
    # create deterministic split
    else:
        # create test_indexes 0, 3, 6, ... and test set
        test_ind = np.arange(0, len(X), 3)
        X_test, y_test = X.loc[test_ind], y[test_ind]

        # create train_indexes 1, 2, 4, 5, ... and train set
        train_ind = np.sort(np.concatenate((np.arange(1, len(X), 3),
                                            np.arange(2, len(X), 3))))
        X_train, y_train = X.loc[train_ind], y[train_ind]

    return X_train, X_test, y_train, y_test


def create_KFolds(y, rs):
    """ Returns train/test indices to split data in train and validation sets

    Parameters
    ----------

    y: array-like set of labels
    rs: int, random state

    Return
    ------
    cv: train/test indices to split data in train and validation sets
    """

    cv = StratifiedKFold(y, n_folds=CONFIG['N_FOLDS'], shuffle=True, random_state=rs)

    return cv
