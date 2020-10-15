import matplotlib
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from mpl_toolkits import mplot3d
import json
from json import JSONEncoder
from collections import namedtuple
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,
                                     train_test_split)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, BayesianRidge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import model_selection

from mpl_toolkits import mplot3d
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.exceptions import *
from .reg_plots import RegressionPlots
from .model_parameters import model_params_decoder
from .permutation_imp import *
from queue import Queue

__all__ = ["Regression", "results_queue"]

results_queue = Queue()
def storeInQueue(f):
    def wrapper(*args):
        results_queue.put(f(*args))
    return wrapper

class Regression():
    def __init__(self, plotsobj):
        #input data initialization : columns names (defaults)
        #self.some_columns= ['latitude']
        self.X_columns = ['X1 transaction date','X2 house age',
        'X3 distance to the nearest MRT station',
        'X4 number of convenience stores','X5 latitude', 'X6 longitude']
        self.y_column = ['Y house price of unit area']

        self.filepath = r'realestate.xls'
        self.excelfile = pd.ExcelFile(self.filepath)
        # preprocessed input data initialization
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()
        self.df = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.features = []
        self.response_column_name = ""
        self.dataframe = None

        # scaler initialization
        self.standardy = StandardScaler()
        self.standardX = StandardScaler()
        self.ss_data = ShuffleSplit(n_splits=10, train_size=0.5, test_size=.5,random_state=0)

        # lasso hyper parameter tuning
        self.alphas = []
        self.scores = np.empty_like(self.alphas)
        self.weights = []

        # results scores initialization
        self.rmse = dict()
        self.r2 = dict()
        self.expvarscore = dict()
        self.plotsobj = plotsobj

    def set_y(self, y_column):
        self.y_column = y_column

    def set_X(self, X_columns):
        self.X_columns = X_columns

    def set_selected_file(self, filepath):
        self.excelfile = pd.ExcelFile(filepath)
        self.filepath =  filepath

    # get sheets, 
    def get_data_from_user(self, filepath):
        self.filepath = r'realestate.xls' 
        #replace all_columns with individual dropdown values
        all_columns = [self.X_columns, self.y_column]
        sheets_list = self.excelfile.sheet_names
        print(all_columns)
        excels = list()
        for sheet, columns in zip(sheets_list, all_columns):
            print(sheet, columns)
            print(pd.read_excel(self.filepath, sheet_name=sheet, mangle_dupe_cols=True)[columns])
            excels.append(pd.read_excel(self.filepath, sheet_name=sheet, mangle_dupe_cols=True)[columns])
        self.dataframe = pd.concat(excels, axis=1)
        
        print(self.dataframe)

    def clean_data(self):
        print(type(self.dataframe))
        print(self.dataframe.columns.values)
        self.X = self.dataframe[self.X_columns]
        self.y = self.dataframe[self.y_column]
        self.y = self.y.round(4)
        self.X = self.X.round(4)
        self.df = pd.DataFrame()
        #remove any trailing spaces in column names
        self.X.columns = self.X.columns.str.strip()
        self.y.columns = self.y.columns.str.strip()
        self.X.columns = self.X.columns.str.replace("[^a-zA-Z\d\_]+", "")
        self.y.columns = self.y.columns.str.replace("[^a-zA-Z\d\_]+", "")
        #check if there are any columns of string type 
        #(if so, trim and remove any special characters)
        if self.X.columns[self.X.dtypes=='str'].dtype == 'str':
            self.X.columns[self.X.dtypes=='str'].data.str.strip()
        else:
            print("No string datatypes in dataframe!")
        if self.y.columns[self.y.dtypes=='str'].dtype == 'str':
            self.y.columns[self.y.dtypes=='str'].data.str.strip()
        else:
            print("No string datatypes in dataframe!")

        #look for any null/missing values 
        self.X.isnull().values.any()
        #scale/normalize the values for plotting features
        self.X.columns = pd.io.parsers.ParserBase({'names':self.X.columns})._maybe_dedup_names(self.X.columns)
        self.X = self.X[self.X.columns[(self.X != 0).any()] ]
        self.y=pd.DataFrame(self.y.iloc[:,:],columns=self.y.columns.values)

        scaled_y=self.standardy.fit_transform(self.y.values.reshape(-1,1))
        scaled_X=self.standardX.fit_transform(self.X)
        self.X.loc[:,:] = scaled_X
        self.y.loc[:,:] = scaled_y
        self.features = self.X.columns.values
        self.response_column_name = self.y.columns.values[0]
        self.df = pd.concat([self.X, self.y],  axis=1)
        self.X = self.X
        self.y = self.y
        self.X = self.X.iloc[:, 0:]
        self.split_data()

    def split_data(self):
        self.ss_data = ShuffleSplit(n_splits=10, train_size=0.5, test_size=.5,random_state=42)
        target = pd.DataFrame(self.y.iloc[:, 0])
        validation_y = target[-90:-1]
        validation_X = self.X[-90:-1]
        for train_index, test_index in self.ss_data.split(self.X):
            self.X_train, self.X_test = self.X[:len(train_index)], self.X[len(train_index): (len(train_index)+len(test_index))]
            self.y_train, self.y_test = target[:len(train_index)].values.ravel(), target[len(train_index): (len(train_index)+len(test_index))].values.ravel()

    def populate_fixtures(self, jsonData):
        if jsonData == "":
            with open('fixtures\params.json') as f:
                self.model_params_obj = json.load(f, object_hook=model_params_decoder)
                #return model_params_obj
        else:
            self.model_params_obj = json.loads(jsonData, object_hook=model_params_decoder)
            #return model_params_obj

    @storeInQueue
    def decisiontreeregr(self):
        dt = DecisionTreeRegressor(random_state=0)

        benchmark_dt=dt.fit(self.X_train, self.y_train)
        #solutionmodels[constants_obj.BENCHMARK] = benchmark_dt
        #plot the benchmark results

        ss = model_selection.ShuffleSplit(n_splits=10, random_state=7)

        cvresults = model_selection.cross_validate(benchmark_dt, self.X_train, 
                            self.y_train, cv=ss, scoring=['neg_mean_squared_error', 'explained_variance', 'r2'], n_jobs=5)
        self.rmse["dtr"] = cvresults['test_neg_mean_squared_error'].tolist()
        self.r2["dtr"] = cvresults['test_r2'].tolist()
        self.expvarscore["dtr"] = cvresults['test_r2'].tolist()

        print(cvresults)
        self.compute_results(benchmark_dt, "Decision Tree Regression", 'dtr')
        plot_benchmark_permutation_importance([benchmark_dt, self.X_train, self.y_train], self.plotsobj, self.features)
        return {'results': [self.plotsobj.results['dtr_train'], self.plotsobj.results['dtr_test']], \
                 'rmse':self.rmse['dtr'], 'r2':self.r2['dtr'], 'expvar': self.expvarscore['dtr']} 

    @storeInQueue
    def svr(self):
        type= 'linear'
        lin_svr = SVR(kernel=type)

        svr_feat = lin_svr.fit(self.X_train, self.y_train)

        #hyperparameter tuning using GridSearch CV
        lin_svr_parameters = {
            'C': self.model_params_obj.svr_params_obj.svr_list_c,
            'epsilon':self.model_params_obj.svr_params_obj.svr_list_espilon,
            }
        
        lin_gridsearch_svr_feat = GridSearchCV(estimator=svr_feat, 
        param_grid=lin_svr_parameters, scoring=['neg_mean_squared_error', 'explained_variance', 'r2'], 
        cv=self.ss_data, n_jobs=5,refit='neg_mean_squared_error')

        #fit model
        lin_gridsearch_svr_feat.fit(self.X_train, self.y_train)
        self.rmse["svr"] = lin_gridsearch_svr_feat.cv_results_['mean_test_neg_mean_squared_error'].tolist()
        self.r2["svr"] = lin_gridsearch_svr_feat.cv_results_['mean_test_r2'].tolist()
        self.expvarscore["svr"] = lin_gridsearch_svr_feat.cv_results_['mean_test_explained_variance'].tolist()
        
        self.compute_results(lin_gridsearch_svr_feat, "Support Vector Regression", 'svr')
        plot_svr_permutation_importance([lin_gridsearch_svr_feat, self.X_train, self.y_train,], self.plotsobj, self.features)
        return {'results': [self.plotsobj.results['svr_train'], self.plotsobj.results['svr_test']], \
                 'rmse':self.rmse['svr'], 'r2':self.r2['svr'], 'expvar': self.expvarscore['svr']}
    @storeInQueue
    def rfr(self):
        rfr = RandomForestRegressor(n_estimators=50, random_state=0)
        rfr_feat = rfr.fit(self.X_train, self.y_train)

        #hyperparameter tuning using GridSearch CV
        rfr_parameters = {
        'n_estimators': self.model_params_obj.rfr_params_obj.rfr_list_n_estimators,
        'max_features': self.model_params_obj.rfr_params_obj.rfr_max_features,
        'max_depth': self.model_params_obj.rfr_params_obj.rfr_max_depth,
        }
        rfr_gridsearch_feat = GridSearchCV(estimator=rfr_feat, 
        param_grid=rfr_parameters, scoring=['neg_mean_squared_error', 'explained_variance', 'r2'], 
        cv=self.ss_data, n_jobs=5, refit='neg_mean_squared_error')

        rfr_gridsearch_feat.fit(self.X_train, self.y_train)

        self.rmse["rfr"]= rfr_gridsearch_feat.cv_results_['mean_test_neg_mean_squared_error'].tolist()
        self.r2["rfr"]= rfr_gridsearch_feat.cv_results_['mean_test_r2'].tolist()
        self.expvarscore["rfr"]= rfr_gridsearch_feat.cv_results_['mean_test_explained_variance'].tolist()
        self.compute_results(rfr_gridsearch_feat, "Random Forest Regression", 'rfr')
        plot_rfr_permutation_importance([rfr_gridsearch_feat, self.X_train, self.y_train], self.plotsobj, self.features)
        return {'results': [self.plotsobj.results['rfr_train'], self.plotsobj.results['rfr_test']], \
                 'rmse':self.rmse['rfr'], 'r2':self.r2['rfr'], 'expvar': self.expvarscore['rfr']}

    @storeInQueue
    def lr(self):
        self.alphas = np.logspace(self.model_params_obj.lasso_params_obj.minAlpha, 
            self.model_params_obj.lasso_params_obj.maxAlpha,
            num=self.model_params_obj.lasso_params_obj.lasso_n_alphas,base=np.e)
        print('alphas', self.alphas)
        lasso = Lasso(max_iter=self.model_params_obj.lasso_params_obj.lasso_max_iter, 
            normalize=False, fit_intercept=True)
        #initialize scores for each alpha
        self.scores = np.empty_like(self.alphas)
        self.weights = []
        for i,a in enumerate(self.alphas):
            lasso.set_params(alpha=a)
            lasso.fit(self.X_train, self.y_train)
            self.weights.append(lasso.coef_)
            #add scores for each alpha
            self.scores[i] = lasso.score(self.X_train, self.y_train)
        print('r-2',self.scores)
        tuned_parameters = [{'alpha': self.alphas}]
        n_folds = 6
        gridsearch_lasso_cv = GridSearchCV(lasso, tuned_parameters, 
            scoring=['neg_mean_squared_error', 'explained_variance', 'r2'], 
            cv=n_folds, refit='neg_mean_squared_error')

        gridsearch_lasso_cv.fit(self.X_train, self.y_train)

        #plot_lasso_permutation_importance(gridsearch_lasso_cv, _X_train, _y_train)

        self.rmse["lr"] = gridsearch_lasso_cv.cv_results_['mean_test_neg_mean_squared_error'].tolist()
        self.r2["lr"] = gridsearch_lasso_cv.cv_results_['mean_test_r2'].tolist()
        self.expvarscore["lr"] = gridsearch_lasso_cv.cv_results_['mean_test_explained_variance'].tolist()
        self.plotsobj.weights_vs_alphas(self.weights, self.alphas, self.features)
        
        self.compute_results(gridsearch_lasso_cv, "Lasso Regression", 'lr')
        plot_lasso_permutation_importance([gridsearch_lasso_cv, self.X_train, self.y_train], self.plotsobj, self.features)
        return {'results': [self.plotsobj.results['lr_train'], self.plotsobj.results['lr_test']], \
                 'rmse':self.rmse['lr'], 'r2':self.r2['lr'], 'expvar': self.expvarscore['lr']}

    @storeInQueue
    def br(self):
        bay_ridge = BayesianRidge()
        bay_feat = bay_ridge.fit(self.X_train, self.y_train)

        #plot the scores
        #plot_br_permutation_importance(bay_feat, _X_train, _y_train)

        ss = model_selection.ShuffleSplit(n_splits=10, random_state=7)

        cv_results = model_selection.cross_validate(bay_feat, self.X_train, 
                            self.y_train, cv=ss, scoring=['neg_mean_squared_error', 'explained_variance', 'r2'], n_jobs=5)

        self.rmse['br'] = cv_results['test_neg_mean_squared_error'].tolist()
        self.r2['br'] = cv_results['test_r2'].tolist()
        self.expvarscore['br'] = cv_results['test_explained_variance'].tolist()
        self.compute_results(bay_feat, "Random Forest Regression", 'br')
        plot_br_permutation_importance([bay_feat, self.X_train, self.y_train], self.plotsobj, self.features)
        return {'results': [self.plotsobj.results['br_train'], self.plotsobj.results['br_test']], \
                 'rmse':self.rmse['br'], 'r2':self.r2['br'], 'expvar': self.expvarscore['br']}

    def compute_results(self, model, title, algorithm):
        ypred_test = model.predict(self.X_test)
        ypred_test = self.standardy.inverse_transform(ypred_test)
        yactual_test = self.standardy.inverse_transform(self.y_test)
        ypred_train = model.predict(self.X_train)
        ypred_train = self.standardy.inverse_transform(ypred_train)
        yactual_train = self.standardy.inverse_transform(self.y_train)
        print(ypred_train)
        mse_train = mean_squared_error(yactual_train, ypred_train)
        mse_test = mean_squared_error(yactual_test, ypred_test)
        r2_train = r2_score(yactual_train, ypred_train)
        r2_test = r2_score(yactual_test, ypred_test)
        print( mse_test, r2_test )
        self.plotsobj.plot_result([yactual_train, ypred_train], [title, 
                self.response_column_name, algorithm, "train"], [mse_train, r2_train])
        self.plotsobj.plot_result([yactual_test, ypred_test], [title,  
                    self.response_column_name, algorithm, "test"], [mse_test, r2_test])

    def compare_scores(self):
        self.plotsobj.plot_algorithm_comparison(self.rmse, "RMSE", 'rmse')
        self.plotsobj.plot_algorithm_comparison(self.r2, "R^2", 'r2')
        self.plotsobj.plot_algorithm_comparison(self.expvarscore, "Explained Variance", 'ev')
    
    @storeInQueue
    def get_allresults(self):
        return {'results': self.plotsobj.results,'rmse':self.rmse, 
            'r2':self.r2, 'expvar': self.expvarscore}

