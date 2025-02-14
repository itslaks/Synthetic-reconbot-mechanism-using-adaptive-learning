#server.py
from try01 import *
import os
import warnings
import logging
from flask import Flask, render_template, redirect, url_for
from infocrypt import infocrypt
from cybersentry_ai import cybersentry_ai
from lana_ai import lana_ai
from osint import osint
from portscanner import portscanner
from webseeker import webseeker
from filescanner import filescanner
from infosight_ai import infosight_ai
from snapspeak_ai import snapspeak_ai
from enscan import enscan
from trueshot import trueshot_ai

# Suppress all warnings and logging except Werkzeug
warnings.filterwarnings('ignore')

# Suppress specific environment warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

# Disable all loggers except Werkzeug
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != 'werkzeug':
        if isinstance(log_obj, logging.Logger):
            log_obj.setLevel(logging.ERROR)

# Flask app initialization
app = Flask(__name__, template_folder='static')
app.logger.handlers = []
app.logger.propagate = False

# Rest of your code remains the same...
blueprints = {
    '/infocrypt': infocrypt,
    '/cybersentry_ai': cybersentry_ai, 
    '/lana_ai': lana_ai,
    '/osint': osint,
    '/portscanner': portscanner,
    '/webseeker': webseeker,
    '/filescanner': filescanner,
    '/infosight_ai': infosight_ai,
    '/snapspeak_ai': snapspeak_ai,
    '/enscan': enscan,
    '/trueshot_ai': trueshot_ai,
}

for prefix, blueprint in blueprints.items():
    app.register_blueprint(blueprint, url_prefix=prefix)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login_success')
def login_success():
    return redirect(url_for('landing_page'))

@app.route('/landing')
def landing_page():
    return render_template('landingpage.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
