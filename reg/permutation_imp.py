
from __future__ import absolute_import
import matplotlib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
from mpl_toolkits import mplot3d
from sklearn.inspection import permutation_importance
import os

__all__ =["plot_benchmark_permutation_importance", "plot_lasso_permutation_importance",
            "plot_svr_permutation_importance", "plot_rfr_permutation_importance",
            "plot_br_permutation_importance"]
def plot_benchmark_permutation_importance(trainlist, plots_obj, features):
    benchmark_dt, X_train, y_train = trainlist
    result = permutation_importance(benchmark_dt, X_train, y_train, n_repeats=10,
                                    random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(benchmark_dt.feature_importances_)
    tree_indices = np.arange(0, len(benchmark_dt.feature_importances_)) + 0.5

    plots_obj.plot_perm_importances(features[perm_sorted_idx].tolist(), 
            result.importances[perm_sorted_idx].T, ['Decision Tree Regression', 'dtr'])
    
def plot_lasso_permutation_importance(trainlist, plots_obj, features):
    gridsearch_lasso_cv, X_train, y_train = trainlist
    result_lasso = permutation_importance(gridsearch_lasso_cv, X_train, y_train, n_repeats=10,
                                random_state=42)
    perm_sorted_idx = result_lasso.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(gridsearch_lasso_cv.best_estimator_.coef_)
    tree_indices = np.arange(0, len(gridsearch_lasso_cv.best_estimator_.coef_)) + 0.5

    print(perm_sorted_idx,tree_importance_sorted_idx )

    plots_obj.plot_perm_importances(features[perm_sorted_idx].tolist(), 
        result_lasso.importances[perm_sorted_idx].T, ['Lasso Regression', 'lr'])

def plot_svr_permutation_importance(trainlist, plots_obj, features):
    gridsearch_svr_feat, X_train, y_train = trainlist
    print(features, type(X_train))
    result_svr = permutation_importance(gridsearch_svr_feat, X_train, y_train, n_repeats=10,
                                random_state=42)
    perm_sorted_idx = result_svr.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(gridsearch_svr_feat.best_estimator_.coef_[0])
    tree_indices = np.arange(0, len(gridsearch_svr_feat.best_estimator_.coef_[0])) + 0.5

    print(perm_sorted_idx,tree_importance_sorted_idx )

    plots_obj.plot_perm_importances(features[perm_sorted_idx].tolist(), 
        result_svr.importances[perm_sorted_idx].T,['Support vector Regression', 'svr'])


def plot_rfr_permutation_importance(trainlist, plots_obj, features):
    rfr_gridsearch_feat, X_train, y_train = trainlist
    result_rfr = permutation_importance(rfr_gridsearch_feat, X_train, y_train, n_repeats=10,
                                random_state=42)
    perm_sorted_idx = result_rfr.importances_mean.argsort()
    tree_importance_sorted_idx = np.argsort(rfr_gridsearch_feat.best_estimator_.feature_importances_)
    tree_indices = np.arange(0, len(rfr_gridsearch_feat.best_estimator_.feature_importances_)) + 0.5

    print(perm_sorted_idx,tree_importance_sorted_idx )

    plots_obj.plot_perm_importances(features[perm_sorted_idx].tolist(), result_rfr.importances[perm_sorted_idx].T,
                    ['Random Forest Regression', 'rfr'])

def plot_br_permutation_importance(trainlist, plots_obj, features):
    bay_feat, X_train, y_train = trainlist
    result_br = permutation_importance(bay_feat, X_train, y_train, n_repeats=10,
                                random_state=42)
    perm_sorted_idx = result_br.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(bay_feat.coef_)
    tree_indices = np.arange(0, len(bay_feat.coef_)) + 0.5

    print(perm_sorted_idx,tree_importance_sorted_idx )

    plots_obj.plot_perm_importances(features[perm_sorted_idx].tolist(), 
        result_br.importances[perm_sorted_idx].T,   ['Bayesian Ridge Regression', 'br'])
