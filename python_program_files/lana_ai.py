import os
from flask import Flask, render_template, jsonify, Blueprint, request
import speech_recognition as sr
import pygame
from gtts import gTTS
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import json
import tempfile
import logging
import subprocess
import time
import threading
import signal
import soundfile as sf
import requests
from io import BytesIO
import whisper
import librosa
import langdetect

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lana_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lana_ai")

# Define log function first before using it
def log(message: str):
    """Log message to file and console"""
    logger.info(message)
    with open("status.txt", "a") as f:
        f.write(f"{time.time()}: {message}\n")

# Create a blueprint
lana_ai = Blueprint('lana_ai', __name__, template_folder='templates')

# API key
GOOGLE_API_KEY = ' '

# Initialize APIs
from google.generativeai import configure, GenerativeModel, ChatSession

configure(api_key=GOOGLE_API_KEY)
model = GenerativeModel('gemini-2.0-flash')
# Define constants
RECORDING_PATH = "audio/recording.wav"
RESPONSE_PATH = "audio/response.mp3"
INITIAL_PROMPT = "You are Lana, Boss human assistant. You are witty and full of personality. Your answers should be limited to 3 lines short sentences. Always reply in the same language as the Boss's input. Do not translate or switch language unless explicitly asked."

# Directory for temp audio files
TEMP_AUDIO_DIR = "audio/temp"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
is_listening = False
thread = None
latest_transcription = ""
latest_response = ""
conversation_lock = threading.Lock()
stop_event = threading.Event()
current_language = "en-US"  # Default language
confidence_level = "high"  # Default confidence level
recognition_attempts = 0
max_recognition_attempts = 3

# Maintain chat history
chat_history = []
chat_session = None

# Language mapping for TTS and detection
LANGUAGE_MAPPING = {
    "en-US": "en",
    "ta-IN": "ta",
    "hi-IN": "hi",
    "ml-IN": "ml",
    "te-IN": "te",
    "kn-IN": "kn",
    "fr-FR": "fr",
    "de-DE": "de",
    "ko-KR": "ko",
    "ja-JP": "ja"
}

# Reverse mapping for language detection
REVERSE_LANGUAGE_MAPPING = {
    "en": "en-US",
    "ta": "ta-IN",
    "hi": "hi-IN",
    "ml": "ml-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "fr": "fr-FR",
    "de": "de-DE",
    "ko": "ko-KR",
    "ja": "ja-JP"
}

# Secondary fallback mapping for difficult languages
FALLBACK_LANGUAGES = {
    "ta-IN": ["en-IN", "en-US"],
    "ml-IN": ["en-IN", "en-US"],
    "te-IN": ["en-IN", "en-US"],
    "kn-IN": ["en-IN", "en-US"],
    "ko-KR": ["en-US"],
    "ja-JP": ["en-US"]
}

# Flag to control audio playback
is_audio_playing = False
pygame_initialized = False

# Initialize WhisperASR
whisper_enabled = False
whisper_model = None
try:
    whisper_model = whisper.load_model("base")
    whisper_enabled = True
    log("WhisperASR model loaded successfully")
except Exception as e:
    whisper_enabled = False
    log(f"Failed to load WhisperASR model: {e}")

def initialize_pygame():
    """Initialize pygame mixer safely"""
    global pygame_initialized
    
    if not pygame_initialized:
        try:
            pygame.mixer.quit()  # Ensure we start clean
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            pygame_initialized = True
            log("Pygame mixer initialized successfully")
        except Exception as e:
            log(f"Failed to initialize pygame mixer: {e}")
            pygame_initialized = False

# Initialize pygame at startup
initialize_pygame()

def initialize_chat_session():
    """Initialize or reset the chat session with Gemini"""
    global chat_session, chat_history
    
    try:
        chat_session = model.start_chat(history=[])
        # Set the initial system prompt only once
        chat_session.send_message(INITIAL_PROMPT)
        chat_history = []
        log("Chat session initialized with system prompt")
        return True
    except Exception as e:
        log(f"Failed to initialize chat session: {e}")
        return False

# Initialize chat session at startup
initialize_chat_session()

def save_conversation_history():
    """Save conversation history to a file"""
    try:
        with open("conversation_history.json", "w") as f:
            json.dump(chat_history, f, indent=2)
        log("Conversation history saved")
    except Exception as e:
        log(f"Error saving conversation history: {e}")

def load_conversation_history():
    """Load conversation history from a file"""
    global chat_history
    
    try:
        if os.path.exists("conversation_history.json"):
            with open("conversation_history.json", "r") as f:
                chat_history = json.load(f)
            log(f"Loaded {len(chat_history)} conversation history items")
            return True
        return False
    except Exception as e:
        log(f"Error loading conversation history: {e}")
        return False

# Try to load existing conversation history
load_conversation_history()

def cleanup_temp_files():
    """Clean up temporary audio files"""
    try:
        for file in os.listdir(TEMP_AUDIO_DIR):
            if file.endswith(".wav") or file.endswith(".mp3"):
                try:
                    os.remove(os.path.join(TEMP_AUDIO_DIR, file))
                except Exception as e:
                    log(f"Error deleting temp file {file}: {e}")
    except Exception as e:
        log(f"Error cleaning temp files: {e}")

# Run cleanup at startup
cleanup_temp_files()

def detect_language(text):
    """Detect language from text and map to our supported languages"""
    try:
        detected = langdetect.detect(text)
        if detected in REVERSE_LANGUAGE_MAPPING:
            return REVERSE_LANGUAGE_MAPPING[detected]
        return "en-US"  # Default fallback
    except:
        return "en-US"  # Default fallback

def request_gemini(user_input: str) -> str:
    """Generate content using the Gemini model with conversation history"""
    global chat_session, chat_history
    
    try:
        # If chat session is not initialized, do it now
        if chat_session is None:
            initialize_chat_session()
            
        # Send the user input to the chat session
        response = chat_session.send_message(user_input)
        
        # Add the exchange to our local history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response.text})
        
        # Save history after each exchange
        save_conversation_history()
        
        return response.text
    except Exception as e:
        log(f"Error generating Gemini response: {e}")
        
        # Try to recover by reinitializing the chat session
        initialize_chat_session()
        
        return "I'm having trouble thinking right now. Could you repeat that?"

def transcribe_with_whisper(audio_file_path: str) -> Optional[str]:
    """
    Transcribe audio using WhisperASR model
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Transcribed text or None if transcription failed
    """
    if not whisper_enabled:
        return None
        
    try:
        # Load audio using librosa for better compatibility
        audio, _ = librosa.load(audio_file_path, sr=16000)
        
        # Transcribe
        result = whisper_model.transcribe(audio)
        
        if result and "text" in result:
            transcribed_text = result["text"].strip()
            log(f"Whisper transcription: {transcribed_text}")
            return transcribed_text
        return None
    except Exception as e:
        log(f"Whisper transcription error: {e}")
        return None

def enhanced_record_audio(timeout: int = 5, phrase_timeout: int = 10, language: str = "en-US") -> Tuple[Optional[str], np.ndarray, str]:
    """
    Enhanced audio recording with better error handling, audio level monitoring, and multi-language support.
    Uses both Google and WhisperASR for better recognition of difficult languages.
    
    Args:
        timeout: Maximum time to wait for speech
        phrase_timeout: Maximum time to wait for a phrase
        language: Language code for speech recognition
        
    Returns:
        Tuple of (transcribed text, audio data, confidence level)
    """
    recognizer = sr.Recognizer()
    audio_data = np.array([])
    confidence = "high"
    
    # Customize speech recognition parameters for different languages
    recognizer.energy_threshold = 300
    
    # Adjust parameters based on language
    if language in ["ja-JP", "ko-KR", "zh-CN"]:
        # Asian languages may need different thresholds
        recognizer.energy_threshold = 280
        recognizer.pause_threshold = 1.0
    elif language in ["ta-IN", "ml-IN", "te-IN", "kn-IN", "hi-IN"]:
        # Indian languages parameters
        recognizer.energy_threshold = 270
        recognizer.pause_threshold = 0.9
    else:
        # Default parameters
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
    
    try:
        with sr.Microphone() as source:
            log(f"Adjusting for ambient noise... (Language: {language})")
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            
            log(f"Energy threshold set to {recognizer.energy_threshold}")
            log(f"Listening in {language}...")
            
            try:
                audio = recognizer.listen(source, 
                                         timeout=timeout,
                                         phrase_time_limit=phrase_timeout)
                
                # Convert audio to numpy array for visualization
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                
                # Save the audio file
                with open(RECORDING_PATH, "wb") as f:
                    f.write(audio.get_wav_data())
                
                # First attempt with Google's speech recognition
                try:
                    text = None
                    
                    try:
                        text = recognizer.recognize_google(audio, language=language)
                        log(f"Google recognition successful in {language}: {text}")
                        return text, audio_data, "high"
                    except sr.UnknownValueError:
                        log(f"Google recognition failed in {language}, trying Whisper...")
                        
                        # For Indian and other challenging languages, try WhisperASR
                        if whisper_enabled and language in ["ta-IN", "ml-IN", "te-IN", "kn-IN", "hi-IN", "ko-KR", "ja-JP"]:
                            whisper_text = transcribe_with_whisper(RECORDING_PATH)
                            if whisper_text:
                                log(f"Whisper recognition successful: {whisper_text}")
                                return whisper_text, audio_data, "high"
                            
                        # Try with adjusted parameters on Google
                        recognizer.energy_threshold = 350
                        try:
                            text = recognizer.recognize_google(audio, language=language)
                            log(f"Secondary Google recognition successful in {language}: {text}")
                            return text, audio_data, "medium"
                        except sr.UnknownValueError:
                            # Try with fallback languages if applicable
                            if language in FALLBACK_LANGUAGES:
                                log(f"Trying fallback languages for {language}")
                                for fallback_lang in FALLBACK_LANGUAGES[language]:
                                    try:
                                        fallback_text = recognizer.recognize_google(audio, language=fallback_lang)
                                        log(f"Fallback recognition successful in {fallback_lang}: {fallback_text}")
                                        return fallback_text, audio_data, "low"
                                    except sr.UnknownValueError:
                                        continue
                            
                            log(f"All recognition attempts failed for {language}")
                            return None, audio_data, "low"
                    
                except sr.RequestError as e:
                    log(f"Google Speech Recognition service error: {e}")
                    return None, audio_data, "low"
                    
            except sr.WaitTimeoutError:
                log("Listening timed out. No speech detected.")
                return None, audio_data, "low"
                
    except Exception as e:
        log(f"Recording error: {e}")
        return None, audio_data, "low"
        
    return None, audio_data, "low"

def record_audio() -> Tuple[str, str]:
    """Main recording function with multiple recognition attempts"""
    global current_language, recognition_attempts, max_recognition_attempts, confidence_level, audio_data
    
    # First attempt
    text, new_audio_data, confidence = enhanced_record_audio(language=current_language)
    audio_data = new_audio_data
    confidence_level = confidence
    
    if text:
        recognition_attempts = 0
        return "Recording complete", confidence
    else:
        # If first attempt fails, increment counter
        recognition_attempts += 1
        if recognition_attempts <= max_recognition_attempts:
            log(f"Recognition attempt {recognition_attempts}/{max_recognition_attempts} failed, will retry")
            return "Recognition failed, retrying", "low"
        else:
            # Reset counter after max attempts
            recognition_attempts = 0
            return "No speech detected", "low"

def transcribe_audio() -> Tuple[str, str]:
    """Transcribe saved audio file using multiple recognition strategies"""
    global current_language
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(RECORDING_PATH) as source:
            audio = recognizer.record(source)
            
            # Try primary language with Google
            try:
                text = recognizer.recognize_google(audio, language=current_language)
                return text, "high"
            except sr.UnknownValueError:
                log(f"Primary Google file transcription failed for {current_language}")
                
                # Try Whisper for challenging languages
                if whisper_enabled and current_language in ["ta-IN", "ml-IN", "te-IN", "kn-IN", "hi-IN", "ko-KR", "ja-JP"]:
                    whisper_text = transcribe_with_whisper(RECORDING_PATH)
                    if whisper_text:
                        log(f"Whisper transcription successful")
                        return whisper_text, "high"
                
                # Try fallback languages if applicable
                if current_language in FALLBACK_LANGUAGES:
                    for fallback_lang in FALLBACK_LANGUAGES[current_language]:
                        try:
                            fallback_text = recognizer.recognize_google(audio, language=fallback_lang)
                            log(f"Fallback file transcription successful in {fallback_lang}")
                            return fallback_text, "medium"
                        except sr.UnknownValueError:
                            continue
                
                # Final attempt with English if everything else fails
                if current_language != "en-US":
                    try:
                        eng_text = recognizer.recognize_google(audio, language="en-US")
                        log("Last resort English transcription successful")
                        return eng_text, "low"
                    except sr.UnknownValueError:
                        pass
                
                return "", "low"
                
    except sr.RequestError as e:
        log(f"Google Speech Recognition service error: {e}")
        return "", "low"
    except Exception as e:
        log(f"Transcription error: {e}")
        return "", "low"

def generate_audio_with_gtts(text: str, lang_code: str) -> Optional[str]:
    """Generate audio with gTTS with better error handling"""
    try:
        # Create a temp file path with timestamp to avoid conflicts
        temp_file = os.path.join(TEMP_AUDIO_DIR, f"gtts_{int(time.time())}.mp3")
        
        # Generate audio
        log(f"Generating audio with gTTS in language: {lang_code}")
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(temp_file)
        
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            log(f"gTTS audio saved to {temp_file}")
            return temp_file
        else:
            log("gTTS created empty file - fallback needed")
            return None
            
    except Exception as e:
        log(f"gTTS error: {e}")
        return None

def play_audio_file(file_path: str) -> bool:
    """Play audio file with error handling and cleanup"""
    global is_audio_playing, pygame_initialized
    
    if not pygame_initialized:
        initialize_pygame()
        
    if not pygame_initialized:
        log("Cannot play audio - pygame not initialized")
        return False
        
    try:
        # Make sure previous playback has stopped
        pygame.mixer.stop()
        
        # Load and play the audio
        is_audio_playing = True
        sound = pygame.mixer.Sound(file_path)
        sound.play()
        
        # Wait for playback to complete
        duration = sound.get_length()
        time.sleep(duration + 0.5)  # Add small buffer
        
        pygame.mixer.stop()
        is_audio_playing = False
        return True
        
    except Exception as e:
        log(f"Audio playback error: {e}")
        is_audio_playing = False
        
        # Try to reinitialize pygame mixer if there was an error
        initialize_pygame()
        return False

def get_tts_with_fallback(text: str, lang_code: str) -> Optional[str]:
    """
    Generate TTS with reliable fallback mechanisms
    Returns the path to the generated audio file
    """
    audio_file = None
    
    # First attempt with primary language
    audio_file = generate_audio_with_gtts(text, lang_code)
    if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
        return audio_file
        
    # If primary language failed or file is too small (likely corrupted)
    log(f"Primary TTS for {lang_code} failed or produced invalid audio, trying English")
    
    # For Indian languages, try to use English TTS but preserve some phonetics
    if lang_code in ["ta", "hi", "ml", "te", "kn"]:
        # Try with English TTS but preserve the text as-is
        audio_file = generate_audio_with_gtts(text, "en")
        if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
            return audio_file
    
    # Final fallback - try with English and potentially simplified text
    try:
        # For non-Latin scripts, try to use a simplified message in English
        if lang_code not in ["en", "fr", "de"]:
            fallback_text = "I'm having trouble speaking in this language right now. Let me try again in English."
            audio_file = generate_audio_with_gtts(fallback_text, "en")
        else:
            audio_file = generate_audio_with_gtts(text, "en")
            
        if audio_file:
            return audio_file
    except Exception as e:
        log(f"Final fallback TTS error: {e}")
    
    # If all attempts failed, return None
    return None

def listen_and_respond():
    """Main loop for listening and responding"""
    global latest_transcription, latest_response, current_language, confidence_level
    
    while not stop_event.is_set():
        try:
            # Record audio
            record_status, conf = record_audio()
            if record_status == "Recognition failed, retrying":
                # Send status to client that we're retrying
                with conversation_lock:
                    latest_transcription = ""
                    latest_response = ""
                    confidence_level = conf
                continue
            elif record_status != "Recording complete":
                confidence_level = conf
                continue

            # Check if stop event is set
            if stop_event.is_set():
                break

            # Transcribe audio
            words, conf = transcribe_audio()
            confidence_level = conf
            
            if not words:
                continue

            # Update latest transcription
            with conversation_lock:
                latest_transcription = words
                confidence_level = conf
            
            # Auto-detect language if needed
            detected_language = detect_language(words)
            if detected_language != current_language:
                log(f"Language detected: {detected_language}, current set to: {current_language}")
                # Optionally update current language based on detection
                # current_language = detected_language

            # Get response from Gemini
            response = request_gemini(words)

            # Update latest response
            with conversation_lock:
                latest_response = response

            # Convert response to audio and play it
            try:
                # Detect language of the response
                response_lang_code = LANGUAGE_MAPPING.get(detect_language(response), "en")
                
                # Get audio file path from TTS function
                audio_file = get_tts_with_fallback(response, response_lang_code)
                
                # Play the audio file
                if audio_file:
                    success = play_audio_file(audio_file)
                    
                    # If playback failed, try with English
                    if not success and response_lang_code != "en":
                        fallback_file = get_tts_with_fallback(response, "en")
                        if fallback_file:
                            play_audio_file(fallback_file)
                else:
                    log("Failed to generate audio")
                            
            except Exception as e:
                log(f"Text-to-speech or playback error: {e}")
                
                # Try to recover pygame if there was an error
                initialize_pygame()

        except Exception as e:
            log(f"Main loop error: {e}")

        # Check for stop event one more time
        if stop_event.is_set():
            break

    log("Listening thread stopped")
    cleanup_temp_files()

@lana_ai.route('/')
def index():
    """Render main page"""
    return render_template('lana.html')

@lana_ai.route('/start_listening', methods=['POST'])
def start_listening():
    """Start the listening process"""
    global is_listening, thread, latest_transcription, latest_response, stop_event, current_language, recognition_attempts
    
    if not is_listening:
        try:
            # Get language from request if available
            data = request.get_json()
            if data and 'language' in data:
                current_language = data['language']
                log(f"Language set to: {current_language}")
            
            is_listening = True
            latest_transcription = ""
            latest_response = ""
            recognition_attempts = 0
            stop_event.clear()
            
            # Make sure pygame is initialized
            initialize_pygame()
            
            thread = threading.Thread(target=listen_and_respond)
            thread.daemon = True  # Make thread daemon so it exits when app exits
            thread.start()
            
            return jsonify({"status": "success", "message": "Listening started", "language": current_language})
        except Exception as e:
            log(f"Error starting listening: {e}")
            return jsonify({"status": "error", "message": f"Error: {str(e)}"})
    
    return jsonify({"status": "error", "message": "Already listening"})

@lana_ai.route('/stop_listening', methods=['POST'])
def stop_listening():
    """Stop the listening process"""
    global is_listening, stop_event, thread, is_audio_playing
    
    if is_listening:
        is_listening = False
        stop_event.set()
        
        # Stop any playing audio
        if is_audio_playing:
            pygame.mixer.stop()
            is_audio_playing = False
            
        if thread:
            thread.join(timeout=5)
            
        # Clean up temp files
        cleanup_temp_files()
        
        return jsonify({"status": "success", "message": "Listening stopped"})
    
    return jsonify({"status": "error", "message": "Not currently listening"})

@lana_ai.route('/process_audio', methods=['POST'])
def process_audio():
    """Process and return audio results"""
    global latest_transcription, latest_response, audio_data
    
    with conversation_lock:
        if latest_transcription or latest_response:
            response = {
                "status": "success",
                "user_transcript": latest_transcription,
                "response": latest_response,
                "audio_data": audio_data.tolist() if len(audio_data) > 0 else [],
                "confidence": confidence_level
            }
            latest_transcription = ""
            latest_response = ""
            return jsonify(response)
            
    return jsonify({"status": "error", "message": "No new transcription available"})

@lana_ai.route('/available_languages', methods=['GET'])
def available_languages():
    """Return list of available languages"""
    languages = [
        {"code": "en-US", "name": "English (US)"},
        {"code": "ta-IN", "name": "Tamil"},
        {"code": "hi-IN", "name": "Hindi"},
        {"code": "ml-IN", "name": "Malayalam"},
        {"code": "te-IN", "name": "Telugu"},
        {"code": "kn-IN", "name": "Kannada"},
        {"code": "fr-FR", "name": "French"},
        {"code": "de-DE", "name": "German"},
        {"code": "ko-KR", "name": "Korean"},
        {"code": "ja-JP", "name": "Japanese"}
    ]
    return jsonify({"status": "success", "languages": languages})

@lana_ai.route('/test_tts', methods=['POST'])
def test_tts():
    """Test TTS for a specific language"""
    data = request.get_json()
    if data and 'language' in data and 'text' in data:
        language = data['language']
        text = data['text']
        
        # Map the language code
        tts_lang = LANGUAGE_MAPPING.get(language, "en")
        
        # Generate audio
        audio_file = get_tts_with_fallback(text, tts_lang)
        
        if audio_file:
            return jsonify({
                "status": "success",
                "message": f"TTS for {language} is working"
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"TTS for {language} failed"
            })
            
    return jsonify({"status": "error", "message": "Language or text not specified"})

@lana_ai.route('/health_check', methods=['GET'])
def health_check():
    """Check if all services are working properly"""
    status = {
        "pygame_initialized": pygame_initialized,
        "whisper_enabled": whisper_enabled,
        "is_listening": is_listening,
        "current_language": current_language,
        "chat_history_length": len(chat_history)
    }
    return jsonify({"status": "success", "health": status})

@lana_ai.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    """Reset the conversation history"""
    global chat_history
    
    try:
        initialize_chat_session()
        return jsonify({"status": "success", "message": "Conversation history reset"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error resetting conversation: {e}"})

# Create required directories
os.makedirs("audio", exist_ok=True)

# Register signal handlers for clean exit
def signal_handler(sig, frame):
    global is_listening, stop_event
    if is_listening:
        is_listening = False
        stop_event.set()
    cleanup_temp_files()
    # Save conversation history before exit
    save_conversation_history()
    logging.info("Signal received, shutting down")
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    app.register_blueprint(lana_ai, url_prefix='/lana_ai')
    app.run(debug=True)
