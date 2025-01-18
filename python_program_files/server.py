import os
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Configure logging to suppress TensorFlow messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

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

app = Flask(__name__, template_folder='static')

# Quick blueprint registration
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
    '/enscan': enscan
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
