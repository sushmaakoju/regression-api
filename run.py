from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import threading
import os
print (os.getcwd())
from app import api, dash_app

dash_app.server = api
#add main flask app and dashboard for results n plots
application = DispatcherMiddleware(api, {
    '/dashboard': dash_app.server,
})

def start_flask_thread():
    run_simple('127.0.0.1', 8050, application, use_reloader=False, use_debugger=False)

if __name__ == '__main__':
    start_flask_thread()