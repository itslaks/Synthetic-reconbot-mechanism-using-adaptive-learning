import os
from flask import Flask, render_template, jsonify, Blueprint, request
import speech_recognition as sr
import pygame
from gtts import gTTS
import threading
import numpy as np
from google.generativeai import configure, GenerativeModel

# Create a blueprint
lana_ai = Blueprint('lana_ai', __name__, template_folder='templates')

# API key (Ensure you have proper API access)
GOOGLE_API_KEY = ''   #user your own gemini api key

# Initialize APIs
configure(api_key=GOOGLE_API_KEY)
try:
    model = GenerativeModel('gemini-pro')
except Exception as e:
    model = None
    print(f"Error loading Google Generative AI model: {e}")

pygame.mixer.init()

# Define constants
RECORDING_PATH = "audio/recording.wav"
RESPONSE_PATH = "audio/response.mp3"
PROMPT_TEMPLATE = "You are Lana, Lak's human assistant. You are witty and full of personality. Your answers should be limited to 3 lines short sentences.\nLaks: {user_input}\nLana: "

is_listening = False
latest_transcription = ""
latest_response = ""
conversation_lock = threading.Lock()
stop_event = threading.Event()

# Audio visualization data
audio_data = np.array([])

def log(message: str):
    print(message)
    with open("status.txt", "a") as f:
        f.write(message + "\n")

def request_gemini(prompt: str) -> str:
    if model is None:
        return "Error: Gemini model not available"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        log(f"Error in request_gemini: {e}")
        return "Error: Unable to generate content"

def transcribe_audio() -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(RECORDING_PATH) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        log("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        log(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def record_audio() -> str:
    global audio_data
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        log("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    
    with open(RECORDING_PATH, "wb") as f:
        f.write(audio.get_wav_data())
    log("Done recording")
    
    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
    return "Recording complete"

def listen_and_respond():
    global latest_transcription, latest_response
    while not stop_event.is_set():
        try:
            record_audio()
            if stop_event.is_set():
                break
            words = transcribe_audio()
            if not words:
                continue
            with conversation_lock:
                latest_transcription = words
            prompt = PROMPT_TEMPLATE.format(user_input=words)
            response = request_gemini(prompt)
            with conversation_lock:
                latest_response = response
            tts = gTTS(response)
            tts.save(RESPONSE_PATH)
            sound = pygame.mixer.Sound(RESPONSE_PATH)
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
        except Exception as e:
            log(f"An error occurred: {e}")
        if stop_event.is_set():
            break
    log("Listening thread stopped")

@lana_ai.route('/')
def index():
    return render_template('lana.html')

@lana_ai.route('/start_listening', methods=['POST'])
def start_listening():
    global is_listening, latest_transcription, latest_response, stop_event
    if not is_listening:
        is_listening = True
        latest_transcription = ""
        latest_response = ""
        stop_event.clear()
        threading.Thread(target=listen_and_respond).start()
        return jsonify({"status": "success", "message": "Listening started"})
    else:
        return jsonify({"status": "error", "message": "Already listening"})

@lana_ai.route('/stop_listening', methods=['POST'])
def stop_listening():
    global is_listening, stop_event
    if is_listening:
        is_listening = False
        stop_event.set()
        return jsonify({"status": "success", "message": "Listening stopped"})
    else:
        return jsonify({"status": "error", "message": "Not currently listening"})

@lana_ai.route('/process_audio', methods=['POST'])
def process_audio():
    global latest_transcription, latest_response, audio_data
    try:
        if request.is_json:
            if 'transcript' in request.json:
                transcript = request.json['transcript']
                prompt = PROMPT_TEMPLATE.format(user_input=transcript)
                response = request_gemini(prompt)
                return jsonify({
                    "status": "success",
                    "user_transcript": transcript,
                    "response": response,
                    "audio_data": audio_data.tolist()
                })
        elif request.data:  # Check if there's raw data (likely audio)
            # Save the received audio data
            with open(RECORDING_PATH, "wb") as f:
                f.write(request.data)
            
            # Transcribe the audio
            transcript = transcribe_audio()
            if transcript:
                prompt = PROMPT_TEMPLATE.format(user_input=transcript)
                response = request_gemini(prompt)
                return jsonify({
                    "status": "success",
                    "user_transcript": transcript,
                    "response": response,
                    "audio_data": audio_data.tolist()
                })
            else:
                return jsonify({"status": "error", "message": "Could not transcribe audio"})
        
        # If no new data, return the latest transcription and response
        with conversation_lock:
            if latest_transcription or latest_response:
                response = {
                    "status": "success",
                    "user_transcript": latest_transcription,
                    "response": latest_response,
                    "audio_data": audio_data.tolist()
                }
                latest_transcription = ""
                latest_response = ""
                return jsonify(response)
        
        return jsonify({"status": "error", "message": "No new transcription available"})
    except Exception as e:
        log(f"Error in process_audio: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Initialize Flask app
app = Flask(__name__)

# Increase maximum request size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max-limit

# Register blueprint
app.register_blueprint(lana_ai, url_prefix='/lana_ai')

if __name__ == '__main__':
    app.run(debug=True)