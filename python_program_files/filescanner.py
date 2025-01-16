import os
from flask import Flask, Blueprint, request, render_template, jsonify
from werkzeug.utils import secure_filename
import requests
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

VIRUSTOTAL_API_KEY = '' #user your own virustotal api key
VIRUSTOTAL_API_URL = 'https://www.virustotal.com/api/v3'

filescanner = Blueprint('filescanner', __name__, template_folder='templates')

@filescanner.route('/')
def index():
    return render_template('filescanner.html')

@filescanner.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        try:
            result = scan_file(file_path)
            os.remove(file_path)
            return jsonify(result)
        except TimeoutError as e:
            os.remove(file_path)
            logger.warning(f"Scan timeout for file: {filename}")
            return jsonify({'error': 'Scan is taking longer than expected. Please try again later.'}), 202
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error during file scan: {str(e)}")
            return jsonify({'error': 'An unexpected error occurred during the scan.'}), 500

def scan_file(file_path):
    url = f'{VIRUSTOTAL_API_URL}/files'
    headers = {'x-apikey': VIRUSTOTAL_API_KEY}
    
    with open(file_path, 'rb') as file:
        files = {'file': (os.path.basename(file_path), file)}
        response = requests.post(url, headers=headers, files=files)
    
    response.raise_for_status()
    upload_result = response.json()
    
    if 'data' not in upload_result or 'id' not in upload_result['data']:
        raise ValueError('Failed to upload file to VirusTotal')

    analysis_id = upload_result['data']['id']
    return get_analysis_result(analysis_id)

def get_analysis_result(analysis_id):
    url = f'{VIRUSTOTAL_API_URL}/analyses/{analysis_id}'
    headers = {'x-apikey': VIRUSTOTAL_API_KEY}
    
    max_attempts = 20  # Increased from 10
    wait_time = 10  # Increased from 5

    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            if result['data']['attributes']['status'] == 'completed':
                return process_analysis_result(result)
            
            logger.info(f"Scan in progress. Attempt {attempt + 1}/{max_attempts}")
            time.sleep(wait_time)
        except requests.RequestException as e:
            logger.error(f"Error getting analysis result (attempt {attempt + 1}): {str(e)}")
    
    raise TimeoutError('Analysis timed out')

def process_analysis_result(result):
    stats = result['data']['attributes']['stats']
    total_scans = sum(stats.values())
    malicious = stats.get('malicious', 0)
    suspicious = stats.get('suspicious', 0)
    
    risk_score = (malicious + suspicious) / total_scans * 100 if total_scans > 0 else 0
    
    return {
        'risk_score': round(risk_score, 2),
        'total_scans': total_scans,
        'malicious': malicious,
        'suspicious': suspicious,
        'full_result': result
    }
# Register the blueprint
app.register_blueprint(filescanner, url_prefix='/filescanner')


if __name__ == '__main__':
    # Ensure temp directory exists
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    app.run(debug=True)