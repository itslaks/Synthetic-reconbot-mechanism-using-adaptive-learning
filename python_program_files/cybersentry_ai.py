import json
import sys
from io import StringIO
from flask import Blueprint, render_template, request, jsonify
import google.generativeai as genai
from fuzzywuzzy import fuzz

# Create a blueprint
cybersentry_ai = Blueprint('cybersentry_ai', __name__, template_folder='templates')

# Load responses from JSON file
def load_responses():
    try:
        with open('responses.json', 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading responses: {e}")
        return []

responses = load_responses()

# Configure Gemini API
genai.configure(api_key='') #user your own gemini api key
model = genai.GenerativeModel('gemini-1.5-flash-latest')

def capture_output(func):
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        result = func(*args, **kwargs)
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        return result, output
    return wrapper

@capture_output
def fuzzy_match(query, responses, threshold=80):
    query = query.lower().strip()
    best_match = None
    best_score = 0
    
    for response in responses:
        if 'question' in response:
            score = fuzz.token_set_ratio(query, response['question'].lower())
            if score > best_score and score >= threshold:
                best_score = score
                best_match = response
    
    return best_match.get('answer') if best_match else None

@capture_output
def get_gemini_response(query):
    try:
        context = "You are a cybersecurity AI assistant. Provide accurate and helpful information about cybersecurity topics. If you're not sure about something, provide the most likely answer based on your knowledge without disclaimers."
        full_prompt = f"{context}\n\nUser: {query}\nAssistant:"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Error fetching response from Gemini API: {e}")
        return None

@cybersentry_ai.route('/')
def index():
    return render_template('cybersentry_AI.html')

@cybersentry_ai.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json['question']
        print(f"Received question: {question}")
        
        answer, json_output = fuzzy_match(question, responses)
        print(f"JSON answer: {answer}")
        
        if answer:
            return jsonify({'answer': answer, 'source': 'JSON', 'terminal_output': json_output})
        else:
            print("No match found in JSON, trying Gemini API")
            gemini_answer, gemini_output = get_gemini_response(question)
            if gemini_answer:
                return jsonify({'answer': gemini_answer, 'source': 'Gemini', 'terminal_output': gemini_output})
            else:
                fallback_answer = "Based on my current knowledge, I don't have a specific answer to that question. However, in cybersecurity, it's important to always prioritize data protection, use strong encryption, keep systems updated, and follow best practices for network security."
                return jsonify({'answer': fallback_answer, 'source': 'Fallback', 'terminal_output': ''})
    except Exception as e:
        print(f"Error in /ask route: {e}")
        return jsonify({'error': str(e), 'terminal_output': ''}), 500

# This is important for Flask to recognize the blueprint
def init_app(app):
    app.register_blueprint(cybersentry_ai, url_prefix='/cybersenty_ai')