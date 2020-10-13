import dash
from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table

import pandas as pd

from flask import request
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests as req
from dash.dependencies import Input, Output
#import chart_studio.plotly.plotly as py
import plotly.offline as py
import numpy as np
import requests
import sys
import chart_studio.dashboard_objs as dashboard
__all__ = ['RegressionPlots']


class RegressionPlots():
    def __init__(self):
        self.plots = dict()
        self.plots['dtr'] = dict()
        self.plots['svr'] = dict()
        self.plots['rfr'] = dict()
        self.plots['lr'] = dict()
        self.plots['br'] = dict()
        self.plots['rmse'] = dict()
        self.plots['r2'] = dict()
        self.plots['ev'] = dict()

        self.results = dict()
        self.algorithms = {'dtr':'Decision Tree Regression', 
        'svr': 'Support Vector Regression', 
              'rfr':'Random Forest Regression', 
              'lr':'Lasso Regression', 
              'br':'Bayesian Ridge Regression'}
    
    def plot_result(self, yvals : list , plotdetails : list, scores : list):
        this_results = dict()
        title, columnname, algorithm, splittype = plotdetails
        this_results['title'] =  html.H5("The "+title+" results for "+columnname)
        
        #Train
        this_results[splittype+'_scores'] = self.get_scores_div(scores[0], scores[1], splittype)

        fig = go.Figure()
        fig.add_trace( go.Scatter(
            x= yvals[0],
            y = yvals[1],
            mode='markers',
            marker=dict(
                size=10,
                color='black',
            )
        ))
        xrange = np.linspace(yvals[0].min(), yvals[0].max(), 100)
        yrange = np.linspace(yvals[1].min(), yvals[1].max(), 100)

        fig.add_trace(go.Scatter(x=xrange, y=yrange,
                        mode='lines',
                        name='lines',marker_color='blue'))
        fig.update_layout(
            title=title+" "+ splittype+ " set",
            xaxis_title="Observed",
            yaxis_title="Predicted",
            legend_title="Train Set",
            paper_bgcolor = 'rgb(251, 251, 251)',
            plot_bgcolor = 'rgb(251, 251, 251)',
            height=450, #width=1000, 
            margin=dict(
                    l=50,
                    r=20,
                    b=100,
                    t=100,
                    pad=10
                ),
            xaxis=dict(gridcolor='DarkGrey', showline=True, linecolor='DarkGrey', zeroline=True, zerolinecolor='DarkGrey'),
            yaxis=dict(gridcolor='DarkGrey', showline=True, linecolor='DarkGrey', zeroline=True, zerolinecolor='DarkGrey'),
        )
        this_results[splittype+'_plot'] =  (dcc.Graph(figure=fig))
        this_results[splittype+'_values'] =  html.Div([
                    html.Br(),
                self.get_datatable(yvals[0], yvals[1], algorithm+"_"+splittype), 
                html.Br(), html.Br(),] )
        self.plots[algorithm].update( this_results)
        #fig.show()

    
    def get_datatable(self, yobs : np.ndarray, ypred : np.ndarray, key : str):
        df = pd.DataFrame(data=np.column_stack((yobs, ypred)), 
                columns=['Observed','Predicted' ])
        self.results[key] = df.to_json()
        layout = dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            sort_action="native",
            sort_mode="multi",
            page_action="native",
            page_size= 10,
            #style_header={'backgroundColor': 'rgb(251, 251, 251)'},
            style_cell={
                'backgroundColor': 'rgb(255, 255, 255)',
                'color': 'black',
                'textAlign': 'center',
                'font_family': "Tahoma, Geneva, sans-serif",
                'font_size': '14px'
            },
            style_data={
                'whiteSpace': 'normal',
            },
            style_header={
                'fontWeight': 'bold',
                'backgroundColor': 'lightgrey',
                'textAlign': 'center',
                'font_family': "Tahoma, Geneva, sans-serif"
            },
            style_table={
                'margin-left':'0px',
                'margin-right':'0px',
                'padding':'15px'
            },
            export_headers='names',
            export_columns='all',
            export_format='csv', # | 'xlsx',
        )
        return layout

    def get_scores_div(self, mse, r2, split_text):
        div = html.Div([
                #html.Br(),
                html.H5(split_text+' results:'),
                html.Br(),
                html.Div([
                    html.H6('Scores- MSE: '+ str(mse) + ' and R^2: '+ str(r2)),
                ]),
                #html.Br(),
            ], className='mat-card', style={"display": "block", "margin-top":"15px", 
                    "margin-bottom":"5px", "border":"1px solid lightgrey"},)
        return div
    
    def weights_vs_alphas(self, weights, alphas, features):
        figure = go.Figure()
        weight = np.array(weights)
        z = np.zeros((1000,))
        weight = np.array(weights)
        for column in range(weight.shape[1]):
            figure.add_trace(go.Scatter(x=np.log(alphas),
                y=weight[:,column],
                name=features[column], line=dict(width=3)
            ))
        figure.update_yaxes(
            ticktext=[features],
        )
        figure.add_trace(go.Scatter(x=alphas,y=z,
                                line = dict(color='black', width=3, dash='dash'), 
                                name="axis line"))
        figure.update_traces(hoverinfo='text+name', mode='lines')

        figure.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
        title_text="LASSO : Coefficient Weight as Alpha Grows", xaxis_title='alpha',
                yaxis_title='coefficient weights')
        self.plots['lr']['weights_alphas'] = (dcc.Graph(figure=figure))


    def plot_algorithm_comparison(self, results, scoring_name, key):
        # boxplot algorithm comparison
        fig = go.Figure()
        for name, result in results.items():
            fig.add_trace(go.Box(y=result, name=name))
        fig.update_layout(title_text=scoring_name+" : Algorithm comparison",
            paper_bgcolor = 'rgb(251, 251, 251)',
                plot_bgcolor = 'rgb(251, 251, 251)',
                height=450, #width=1000, 
                margin=dict(
                        l=50,
                        r=20,
                        b=100,
                        t=100,
                        pad=10
                    ),
                xaxis=dict(gridcolor='DarkGrey', showline=True, linecolor='DarkGrey', zeroline=True, zerolinecolor='DarkGrey'),
                yaxis=dict(gridcolor='DarkGrey', showline=True, linecolor='DarkGrey', zeroline=True, zerolinecolor='DarkGrey'),
        )

        self.plots[key] = (dcc.Graph(figure=fig))
        print(key)
    
    def plot_perm_importances(self, features, importances, plot_details : list):
        title, algorithm = plot_details
        fig = go.Figure()
        
        for name, result in zip( features, importances):
                fig.add_trace(go.Box(x=result, name=name, orientation='h'))
        fig.update_layout(title_text="Permutation Importances for"+ title,
                paper_bgcolor = 'rgb(251, 251, 251)',
                plot_bgcolor = 'rgb(251, 251, 251)',
                height=450, #width=1000, 
                margin=dict(
                        l=50,
                        r=20,
                        b=100,
                        t=100,
                        pad=10
                    ),
                xaxis=dict(gridcolor='DarkGrey', showline=True, linecolor='DarkGrey', zeroline=True, zerolinecolor='DarkGrey'),
                yaxis=dict(gridcolor='DarkGrey', showline=True, linecolor='DarkGrey', zeroline=True, zerolinecolor='DarkGrey'),
                )
        self.plots[algorithm]['perm_imp'] = (dcc.Graph(figure=fig))


