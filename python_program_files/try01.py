# silence.py
import os
import sys
import logging
import warnings

# Environment variables - set before any other imports
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'CUDA_VISIBLE_DEVICES': '-1',
    'TF_ENABLE_ONEDNN_OPTS': '0',
    'PYGAME_HIDE_SUPPORT_PROMPT': 'hide',
    'PYTHONWARNINGS': 'ignore',
})

# Configure Werkzeug logger
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.INFO)

# Disable all other loggers
logging.getLogger('tensorflow').disabled = True
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress all other loggers
for name in logging.root.manager.loggerDict:
    if name != 'werkzeug':
        logging.getLogger(name).disabled = True
        logging.getLogger(name).setLevel(logging.ERROR)

# Filter warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)