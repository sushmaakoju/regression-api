from __future__ import absolute_import
import flask
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

import threading
import dash
from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

import flask_cors
from flask_cors import CORS
from flask.helpers import send_file
from werkzeug.exceptions import HTTPException

from queue import Queue

from reg import *
from hcommons.reg_constants import constants_obj
from reg.reg_plots import RegressionPlots

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

__all__ = ['api', 'dash_app', 'plotsobj']

api = Flask(__name__)
CORS(api)
results_path = "results"
path = Path(__file__).parent
filedir = path.joinpath(path.parent, "data")
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u',
        'crossorigin': 'anonymous'
    }]
plotsobj = RegressionPlots()

def serve_layout():
    title = html.Div([
        html.H4("Regression Training for Dataset")], 
        className='title-card', style={"display": "block", "margin-bottom":"0px","margin-left":"200px" 
                 #"border":"1px solid lightgrey"
                 })
    children = html.Div([ title,
                    html.Div( 
                        #style={"margin":"70px"},
                    className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},                 
                        children=list(plotsobj.plots['dtr'].values()) ),
                    html.Div(className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},
                        children=list(plotsobj.plots['svr'].values())),
                    html.Div(className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},
                        children=list(plotsobj.plots['rfr'].values())),
                    html.Div(className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},
                        children=list(plotsobj.plots['lr'].values())),
                    html.Div(className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},
                        children=list(plotsobj.plots['br'].values())),
                    html.Div(className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},
                        children=[plotsobj.plots['rmse']]),
                    html.Div(className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},
                        children=[plotsobj.plots['r2']]),
                    html.Div(className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},
                        children=[plotsobj.plots['ev']]),
                ], className='main-card', style={"display": "block", "margin":"30px", 
                 "border":"1px solid lightgrey"})
    return html.Div([
            # represents the URL bar, doesn't render anything
            dcc.Location(id='url', refresh=False),
            dcc.Link('Navigate to "/"', href='/results'),
            html.Br(),
            html.Hr(),
            # content will be rendered in this element
            html.Div(id='page-content', children=children),
            html.Div(id='results-tables'),
            html.Div(id='results-scores')
        ])

"""Create a Plotly Dash dashboard."""
dash_app = dash.Dash(
    __name__, 
    server=api,
    url_base_pathname='/results/',
    external_stylesheets=[dbc.themes.BOOTSTRAP])

dash_app.layout = serve_layout
reg_obj =  Regression(plotsobj)
reg_obj.get_data_from_user("")
reg_obj.clean_data()
reg_obj.populate_fixtures('')

@api.route("/")
def start():
    return jsonify({'result':"Welcome to Regression api!"})

@api.route('/upload_data', methods=['POST'])
def preprocess_data():
    try:
        reg_obj.set_selected_file(request.files['file'])
    except:
        raise Exception("Invalid file type")
    print(request.files['file'])
    return jsonify({'ready': "uploaded data file"}) 

@api.route('/select_columns', methods=['POST'])
def select_columns():
     columns = json.loads(request.data.decode('utf-8'))
     print(request, columns)
     reg_obj.set_X(columns['features'])
     reg_obj.set_y(columns['response'])
     reg_obj.get_data_from_user()
     return jsonify({'status': "completed"}) #later send json va;ues to plot using chartjs 

@api.route('/select_params', methods=['POST'])
def select_params():
     json_ = request.data # all parameters from parameters form in UI
     model_params_obj = reg_obj.populate_fixtures(json_)
     return jsonify({'status': "completed"}) #later send json values to plot using chartjs 

@api.route('/train_all', methods=['POST'])
def train_all():
    #test default
    results = {}
    reg_obj.get_data_from_user("")
    reg_obj.clean_data()
    reg_obj.populate_fixtures('')
    t = threading.Thread(target=reg_obj.decisiontreeregr())
    t.start()
    t.join()
    results.update(results_queue.get())
    t = threading.Thread(target=reg_obj.svr())
    t.start()
    t.join()
    results.update(results_queue.get())
    t = threading.Thread(target=reg_obj.rfr())
    t.start()
    t.join()
    results.update(results_queue.get())
    t = threading.Thread(target=reg_obj.lr())
    t.start()
    t.join()
    results.update(results_queue.get())
    t = threading.Thread(target=reg_obj.br())
    t.start()
    t.join()
    results.update(results_queue.get())    
    reg_obj.compare_scores()
    return jsonify({'status': "Training completed.",
        "results":json.dumps(results)})


@api.route('/test', methods=['POST'])
def prepare():
    reg_obj.get_data_from_user("")
    reg_obj.clean_data()
    reg_obj.populate_fixtures('')
    reg_obj.decisiontreeregr()
    return jsonify({'status': "completed."})

@api.route('/train_dtr', methods=['POST'])
def train_dtr():
    t = threading.Thread(target=reg_obj.decisiontreeregr())
    t.start()
    t.join()
    results = results_queue.get()
    print(results)
    return jsonify({'status': "Training Decision Tree Regression completed.",
     "results":json.dumps(results)})

@api.route('/train_svr', methods=['POST'])
def train_svr():
    results = {}
    t = threading.Thread(target=reg_obj.svr())
    t.start()
    t.join()
    results.update(results_queue.get())     
    return jsonify({'status': "Training Support Vector Regression completed.",
     "results":json.dumps(results)})

@api.route('/train_rfr', methods=['POST'])
def train_rfr():
    results = {}
    t = threading.Thread(target=reg_obj.rfr())
    t.start()
    t.join()
    results.update(results_queue.get())     
    return jsonify({'status': "Training Random Forest Regression completed.",
     "results":json.dumps(results)})

@api.route('/train_lr', methods=['POST'])
def train_lr():
    results = {}
    t = threading.Thread(target=reg_obj.lr())
    t.start()
    t.join()
    results.update(results_queue.get())     
    return jsonify({'status': "Training Lasso Regression completed.",
     "results":json.dumps(results)})

@api.route('/train_br', methods=['POST'])
def train_br():
    results = {}
    t = threading.Thread(target=reg_obj.br())
    t.start()
    t.join()
    results.update(results_queue.get())     
    return jsonify({'status': "Training Bayesian Ridge Regression completed.",
     "results":json.dumps(results)})

@api.route('/compare_scores', methods=['POST'])
def compare_scores():
     reg_obj.compare_scores()
     return jsonify({'status': " completed."})

@api.route('/get_training_results', methods=['POST'])
def get_training_results():
    results = {}
    t = threading.Thread(target=reg_obj.get_allresults())
    t.start()
    t.join()
    results.update(results_queue.get())     
    return jsonify({'status': "Training results downloaded",
        "results":json.dumps(results)})

@api.route('/results/')
def render_dashboard():
    return flask.redirect('/dashboard')

@api.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

if __name__ == '__main__':
    pass
    #api.run(debug=True, port=8050)