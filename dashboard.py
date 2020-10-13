from __future__ import absolute_import

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from xhtml2pdf import pisa
import dash_bootstrap_components as dbc

import dash
from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
from app import dash_app, plotsobj

def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

#@api.route('/dashboard/', methods=['GET'])
@dash_app.callback(dash.dependencies.Output('page-content', 'children'), 
[dash.dependencies.Input('url', 'pathname')])
def display_plots(pathname):
    if pathname =='/shutdown':
        return shutdown()
    elif pathname == '/dashboard/' or pathname == '/results/':
        if len(plotsobj.plots) > 0:
            plots_tables_div = html.Div([html.H4("The results of \
                    Regression Training are as follows: "),
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
                        children=list(plotsobj.plots['br'])),
                    html.Div(className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},
                        children=list(plotsobj.plots['rmse'])),
                    html.Div(className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},
                        children=list(plotsobj.plots['r2'])),
                    html.Div(className='mat-card', style={"display": "block", "margin":"30px",
                    "border":"1px solid lightgrey"},
                        children=list(plotsobj.plots['ev'])),
                ])
            
            return html.Div([
                            html.H3("The results of Regression Training are as follows:"+' {}'.format(pathname)),
                            plots_tables_div,
                        ])
        else:
            return html.Div([
                    html.H3(
                        "The results of Regression Training will be displayed \
                        once you execute the algorithm: "+' {}'.format(pathname)),
                ])
    else:
        return html.Div([
        html.H3(
            "The results of Regression Training will be displayed \
            once you execute the algorithm: "+' {}'.format(pathname)),
        ])

def convert_html_to_pdf(source_html, output_filename):
    # open output file for writing (truncated binary)
    result_file = open(output_filename, "w+b")

    # convert HTML to PDF
    pisa_status = pisa.CreatePDF(
            source_html,                # the HTML to convert
            dest=result_file)           # file handle to recieve result

    # close output file
    result_file.close()                 # close output file

    # return True on success and False on errors
    return pisa_status.err

def trigger_callback():
    dash_app.callback(dash.dependencies.Output('page-content', 'children'), [dash.dependencies.Input('url', '')])(display_plots)
