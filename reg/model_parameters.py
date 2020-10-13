from __future__ import absolute_import
import numpy as np
import json
from json import JSONEncoder
from collections import namedtuple

__all__ = ['SvrParams', 'RfrParams', 'LassoParams', 'ModelParameters', 
            'IncorrectParamsError', 'ObjEncoder', 'model_params_decoder']

class IncorrectParamsError(Exception):
    pass

class SvrParams:
    """
    This class creates parameters for SVR regression analysis using Scikit learn library. 

    Attributes:
        minC (int): Minimum value for Regularization Parameter C
        maxC (int): Maximum value for Regularization Parameter C
        minEpsilon (int): Minimum value for epsilon for epsilon-SVR model
        maxEpsilon (int): Maximum value for epsilon for epsilon-SVR model
    
    Usage:
        svr_params_obj = SvrParams()
    """
    def __init__(self):
        """
        This method creates default parameters for SVR regression analysis using Scikit learn library. 
 
        """
        self.minC = 0.5
        self.maxC = 50.0
        self.minEpsilon = 0.0
        self.maxEpsilon = 0.9
        self.svr_list_c = np.arange(self.minC, self.maxC, 5.0, dtype=float).tolist()
        self.svr_list_espilon = np.arange(self.minEpsilon, self.maxEpsilon, step=0.2,dtype=float).tolist()
        self.kernel = ["linear", "rbf","poly", "sigmoid", "precomputed"]
    
    def setSvrParams(self, minC, maxC, minEpsilon, maxEpsilon, kernel="rbf"):
        """
        This class creates default parameters for SVR regression analysis using Scikit learn library.
        C is the Regularization parameter, must be positive. 

        Args:
        minC (int): Minimum value for Regularization Parameter C
        maxC (int): Maximum value for Regularization Parameter C
        minEpsilon (int): Minimum value for epsilon for epsilon-SVR model
        maxEpsilon (int): Maximum value for epsilon for epsilon-SVR model
        
        """
        if minC <= 0 or maxC <= 0 or minEpsilon < 0 or minEpsilon >= maxEpsilon:
            raise IncorrectParamsError(Exception('Check Support Vector Regressor parameters'))

        self.minC = minC
        self.maxC = maxC
        self.minEpsilon = minEpsilon
        self.maxEpsilon = maxEpsilon
        self.svr_list_c = np.arange(self.minC, self.maxC, 5.0, dtype=float).tolist()
        self.svr_list_espilon = np.arange(self.minEpsilon, self.maxEpsilon, step=0.2,dtype=float).tolist()
        self.kernel = kernel
        return self  
    
class RfrParams:
    """
    Set default parameters for Random Forest regression analysis using Scikit learn library.
    Number of trees (estimators) . 

    Attributes:
        minEstimators (int): Minimum value for # Estimators
        maxEstimators (int): Maximum value for # Estimators
        maxFeatures: Maximum # of features to consider during a split
            default: 'auto', 'sqrt', 'log2' or int or float
        maxDepthLowerLimit (int): Maximum depth of the tree Minimum value
        maxDepthUpperLimit (int): Maximum depth of the tree Minimum value

    Usage:
        rfr_params_obj = RfrParams()
    """
    def __init__(self):
        """
        This method creates default parameters for Random Forest regression analysis using Scikit learn library.
        Number of trees (estimators). 
        """
        self.minEstimators = 10
        self.maxEstimators = 100
        self.maxFeatures = ['auto', '0.8']
        self.maxDepthLowerLimit = 2
        self.maxDepthUpperLimit = 10
        self.rfr_list_n_estimators = np.arange(self.minEstimators, self.maxEstimators, step=10,dtype=int).tolist()
        self.rfr_max_features = [0.8, 'auto', 'sqrt', 'log2']
        self.rfr_max_depth = np.arange(self.maxDepthLowerLimit, self.maxDepthUpperLimit, step=2,dtype=int).tolist()
    
    def setRfrParams(self, minEstimators, maxEstimators, maxFeatures, 
            maxDepthLowerLimit, maxDepthUpperLimit):
        """
        Set parameters for Random Forest regression analysis using Scikit learn library.
        Number of trees (estimators) . 

        Args:
            minEstimators (int): Minimum value for # Estimators
            maxEstimators (int): Maximum value for # Estimators
            maxFeatures: Maximum # of features to consider during a split
                default: 'auto', 'sqrt', 'log2' or int
            maxDepthLowerLimit (int): Maximum depth of the tree Minimum value
            maxDepthUpperLimit (int): Maximum depth of the tree Minimum value
        """
        if minEstimators == 0 or maxEstimators == 0 or maxFeatures == [''] or \
        maxDepthLowerLimit >= maxDepthUpperLimit:
            raise IncorrectParamsError(Exception('Check Random forest Regressor parameters'))

        self.minEstimators = minEstimators
        self.maxEstimators = maxEstimators
        self.maxDepthLowerLimit = maxDepthLowerLimit
        self.maxDepthUpperLimit = maxDepthUpperLimit
        self.maxFeatures = maxFeatures
        self.rfr_list_n_estimators = np.arange(self.minEstimators, self.maxEstimators, 20, dtype=int).tolist()
        self.rfr_max_features = self.maxFeatures
        self.rfr_max_depth = np.arange(self.maxDepthLowerLimit, self.maxDepthUpperLimit, step=2,dtype=int).tolist()
        return self

class LassoParams:
    """
    Set default parameters for Lasso regression analysis using Scikit learn library.
    Number of trees (estimators) . 

    Attributes:
        numAlphas (int): # of alpha values to try along the regularization path
        maxIterations (int): Maximum # of iterations

    Usage:
        lasso_params_obj = LassoParams()
    """
    def __init__(self):
        """
        Set parameters for Lasso regression analysis using Scikit learn library.
        Number of trees (estimators). 
        """
        self.minAlpha = -10.0
        self.maxAlpha = 1.5
        self.lasso_n_alphas = 10000
        self.lasso_max_iter = 10000000
    
    def setLassoParams(self, minAlpha, maxAlpha, numAlphas, maxIterations):
        """
        Set parameters for Lasso regression analysis using Scikit learn library.
        Number of trees (estimators). 

        Args:
            numAlphas (int): # of alpha values to try along the regularization path
            maxIterations (int): Maximum # of iterations
        """
        if numAlphas == 0 or maxIterations == 0:
            raise IncorrectParamsError(Exception('Check Lasso Regressor parameters'))
        self.minAlpha = minAlpha
        self.maxAlpha = maxAlpha
        self.lasso_n_alphas = numAlphas
        self.lasso_max_iter = maxIterations
        return self


class BRParams:
    """
    Set default parameters for Bayesian Ridge regression analysis using Scikit learn library.

    Attributes:
        alpha_1: Shape parameter for Gamma distribution prior for alpha
        alpha_2: Rate parameter for Gamma distribution prior for alpha
        lambda_1: Shape parameter for Gamma distribution prior for lambda
        lambda_2: Rate parameter for Gamma distribution prior for lambda


    Usage:
        br_params_obj = BRParams()
    """
    def __init__(self):
        """
        Set parameters for Bayesian Ridge regression analysis using Scikit learn library.
        """
        self.alpha_1 = None
        self.alpha_2 = None
        self.lambda_1 = None
        self.lambda_2 = None
    
    def setBRParams(self, alpha_1, alpha_2, lambda_1, lambda_2):
        """
        Set parameters for Bayesian Ridge regression analysis using Scikit learn library.

        Args:
        alpha_1: Shape parameter for Gamma distribution prior for alpha
        alpha_2: Rate parameter for Gamma distribution prior for alpha
        lambda_1: Shape parameter for Gamma distribution prior for lambda
        lambda_2: Rate parameter for Gamma distribution prior for lambda
        """
        self.alpha_1 = alpha_1 if alpha_1 != None  else 1e-6
        self.alpha_2 = alpha_2 if alpha_2 != None  else 1e-6
        self.lambda_1 = lambda_1 if lambda_1 != None  else 1e-6
        self.lambda_2 = lambda_2 if lambda_2 != None  else 1e-6

        return self


class ModelParameters:
    """
    Set parameters for regression training and analysis using Scikit learn library.  

        Usage:
        model_params = ModelParameters()
    """
    def __init__(self, svr_params_obj=None, rfr_params_obj=None, lasso_params_obj=None, br_params_obj=None):
        """
        Initialize default parameters for SvrParams, RfrParams and LassoParams. 

        The param classes for regression training and analysis using Scikit learn library. 
        """
        self.svr_params_obj = SvrParams()
        self.rfr_params_obj = RfrParams()
        self.lasso_params_obj = LassoParams()
        self.br_params_obj = BRParams()
    
    def getSvrParams(self):
        return self.svr_params_obj

    def getRfrParams(self):
        return self.rfr_params_obj

    def getLassoParams(self):
        return self.lasso_params_obj
    
    def getBRParams(self):
        return self.br_params_obj


class ObjEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__

def model_params_decoder(obj):
    if '__type__' in obj and obj['__type__'] == 'ModelParameters':
        return ModelParameters(obj['svr_params_obj'], obj['rfr_params_obj'], obj['lasso_params_obj'], obj['br_params_obj'])
    return obj