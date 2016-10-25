from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from globals import *
import matplotlib.pyplot as plt


def plot_roc_curves(y_train, pred_y_train, y_test, pred_y_test):
    """ Plots roc curves for train and test set predictions

    Parameters
    ----------
    y_train: array-like train set labels
    pred_y_train: array-like train set predictions (probabilities)
    y_test: array-like test set labels
    pred_y_test: array-like test set predictions (probabilities)

    Return
    ------
    Figure of train and test set roc curves
    """

    plt.figure(1)

    # training set roc curve
    plt.subplot(1, 2, 1)
    plot_roc_curve(y_train, pred_y_train, 'Training Set', show=False)

    # test set roc curve
    plt.subplot(1, 2, 2)
    plot_roc_curve(y_test, pred_y_test, 'Test Set', show=False)

    # show figure
    # plt.show()


def plot_roc_curve(y_true, y_pred, name, show=True):
    """ Plots roc curve

    Parameters
    ----------
    y_true: array-like set of labels
    y_pred: array-like set of predictions (probabilities)
    name: name of ROC curve
    show: bool, default True.
        If True plot roc curve on a separate figure.
        If False create roc curve without plotting it. Can be used for creating subplots

    Return
    ------
    Figure/Sub Plot of roc curve
    """

    # auc score
    auc_score = roc_auc_score(y_true, y_pred)

    # false positive and true positive rates
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # add point (0, 0) to roc curve
    fpr = np.concatenate((np.array([0.0]), fpr))
    tpr = np.concatenate((np.array([0.0]), tpr))

    # create figure
    if show:
        plt.figure()

    # plot roc curve and create boundaries
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    # create labels
    plt.text(0.2, 0.1, 'AUC = %0.4f' % auc_score, fontsize=30)
    plt.xlabel('1 - Specifity', fontsize=30)
    plt.ylabel('Sensitivity', fontsize=30)
    plt.title('ROC Curve (%s)' % name, fontsize=30)

    # show grid
    plt.grid(True)

    # show figure
    if show:
        plt.show()


def plot_histograms(aucs_train, aucs_test):
    """ Plots histograms curves for train and test set AUC

    Parameters
    ----------
    aucs_train: array-like train set AUC scores
    aucs_test: array-like test set AUC scores

    Return
    ------
    Figure of train and test set AUC scores histograms
    """

    # create figure
    plt.figure(2)

    # plot train set AUC scores histogram
    plt.subplot(1, 2, 1)
    plot_histogram(aucs_train, 'Training Set', show=False)

    # plot train set AUC scores histogram
    plt.subplot(1, 2, 2)
    plot_histogram(aucs_test, 'Test Set', show=False)

    # show figure
    # plt.show()


def plot_histogram(aucs, name, show=True):
    """ Plots histogram

    Parameters
    ----------
    aucs: array-like set of AUC scores
    name: name of AUC scores histogram
    show: bool, default True.
        If True plot AUC scores histogram on a separate figure.
        If False create AUC score histogram without plotting it. Can be used for creating subplots

    Return
    ------
    Figure/Sub Plot of AUC score histogram
    """

    # create figure
    if show:
        plt.figure()

    #
    mean_score = np.mean(aucs)
    n, _, _ = plt.hist(aucs) #, label = 'Mean AUC = %0.4f' % mean_score)
    plt.axvline(x=mean_score, color='k', linewidth=2, ls='dashed')

    # setup boundaries
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, np.max(n) + 1])

    # create labels
    plt.xlabel('AUC', fontsize=25)
    plt.ylabel('Frequency', fontsize=25)
    plt.text(0.1, (np.max(n)+1)/2, 'Mean AUC = %0.4f' % mean_score, fontsize=25)
    plt.title('Distribution of AUCs (%s)' % name, fontsize=25)
    plt.legend(loc="upper left")

    # plot grid
    plt.grid(True)

    # show figure
    if show:
        plt.show()


def plot_betas(betas, names):
    """ Plot beta coefficients (with 95% confidence intervals error bars)

    Parameters
    ----------
    betas - array-like list of beta coefficients
    names - array-like list of names corresponding to betas

    Return
    ------
    Figure of beta coefficients (with 95% confidence intervals error bars)
    """

    # replace betas with low survival rate by zeros
    betas_survival = np.mean(betas != 0, axis=0)
    betas_survived = betas_survival > CONFIG['SURVIVAL_RATE_CUTOFF']
    betas = np.multiply(betas, betas_survived)

    # create 95 confidence interval of the mean estimate
    # lb - 2.5 percentile, ub - 97.5 percentile, mean - mean
    lb = np.reshape(np.percentile(betas, q=2.5, axis=0), [betas.shape[1], 1])
    ub = np.reshape(np.percentile(betas, q=97.5, axis=0), [betas.shape[1], 1])
    means = np.reshape(np.mean(betas, axis=0), [betas.shape[1], 1])

    # combine lb, ub and means in a data frame for convenience
    df = pd.DataFrame(np.concatenate((lb, ub, means), axis=1), index=names, columns=['lb', 'ub', 'mean'])

    # to reproduce the same order as in the paper
    # list of indexes in data frame and list of corresponding labels
    index = ["Male", "AGE", "EDU_YRS", "BIS_attention", "BIS_motor", "BIS_NonPL",
             "Stop_ssrt", "IMT_OMIS_errors", "IMT_COMM_errors", "A_IMT", "B_D_IMT",
             "lnk_adjdd", "lkhat_Kirby", "REVLR_per_errors", "IGT_TOTAL"]
    labels = ["Sex", "Age", "Education", "BIS Attn", "BIS Motor", "BIS Nonpl",
              "SSRT", "IMT FN", "IMT FP", "IMT Discriminability", "IMT Response bias",
              "ln(k)", "ln(k), Kirby", "PRL perseverance", "IGT Score"]

    # ys - equally spaced
    y = np.arange(0, len(means))

    # plot figure
    plt.figure(3)

    # plot ys
    plt.ylim([-0.5, len(y) - 0.5])
    plt.yticks(y, labels, rotation='horizontal')

    # plot error bars
    plt.errorbar(df['mean'][index], y, xerr=[df['mean'][index]-df['lb'][index], df['ub'][index]-df['mean'][index]],
                 fmt='o', capthick=2)

    # create labels
    plt.xlabel('Beta Estimates')
    plt.ylabel('Predictors')
    plt.title('Predicting Cocaine Group Membership')

    # show figure
    # plt.show()


