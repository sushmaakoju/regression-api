from __future__ import absolute_import
import re
__all__ = ['Constants', 'constants_obj']

class Constants:

    def __init__(self):
        self.filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        self.default_list_c = [0.5, 1.0, 10.0, 50.0]
        self.default_list_espilon = [0, 0.1, 0.5, 0.7, 0.9]
        self.default_list_n_estimators = [10,15,20,50,100]
        self.default_max_features = ['auto', 'sqrt', 'log2']
        self.default_max_depth = [2,3,5,7,10]
        self.default_n_alphas = 1000
        self.default_max_iter = 3000
        self.gradient_n_estimators = 70
        self.gradient_learning_rate = 0.1
        self.max_features_all = ['auto', 'log2', 'sqrt']
        self.ALL_DEFAULT_OPTIONS = "ALL"
        self.SELECTED_PARAM_OPTIONS = "SELECTED"
        self.TRAINING_COMPLETE = "TRAINING_COMPLETE"
        self.ALL_ALGORITHMS = ['svr', 'lasso', 'rfr']
        self.SVR = "SVR"
        self.SVR_WITH_GRIDSEARCH = "SVR_with_Gridsearch"
        self.LASSO = "LASSO"
        self.LASSO_WITH_GRIDSEARCH = "LASSO_with_Gridsearch"
        self.RFR = "RFR"
        self.LASSO_WEIGHTS_ALPHA_PLOT = "LASSO_Weights_versus_Alpha_plot"
        self.LASSO_ALPHAS_R2_SCORES_PLOT = "LASSO_ALPHAS_R2_SCORES_plot" 
        self.RFR_WITH_GRIDSEARCH = "RFR_with_Gridsearch"
        self.BENCHMARK = "BENCHMARK"
        self.BR = "BR"
        self.RMSE = 'RMSE'
        self.RSQUARED = 'R-Squared'
        self.EXPLAINED_VARIANCE = 'EXPLAINED_VARIANCE'

        # Model training status complete
        self.STATUS_COMPLETE = 'COMPLETE'

        VARIABLE_PATTERN = "^[a-zA-Z]+[_]*[1-9]+[0]*" #patterns x_1, x_2 or x100 x23
        NUMBER_PATTERN = "^[1-9]+[0]*" #2, 22, 20
        OPERATOR_PATTERN = "(\+|\-|\*|\\|\^){1}" #+-*\^
        INVALID_PATTERN = "[\n\t]+"

constants_obj = Constants()
