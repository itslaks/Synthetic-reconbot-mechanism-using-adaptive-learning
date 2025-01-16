from flask import Blueprint, request, jsonify, render_template, send_file
from flask_cors import CORS
import requests
import base64
import google.generativeai as genai
import logging
from PIL import Image
import io
import re
from datetime import datetime, timedelta
from collections import deque
import threading
import time

# Create blueprint
infosight_ai = Blueprint('infosight_ai', __name__, template_folder='templates')
logger = logging.getLogger(__name__)
CORS(infosight_ai)

# API Configuration
GEMINI_API_KEY = "" #user your own gemini api key
HF_API_TOKEN = "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" #user your own hugging face api key

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()

    def can_proceed(self):
        now = datetime.now()
        with self.lock:
            while self.requests and self.requests[0] < now - timedelta(seconds=self.time_window):
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    def wait_time(self):
        if not self.requests:
            return 0
        now = datetime.now()
        oldest_request = self.requests[0]
        return max(0, (oldest_request + timedelta(seconds=self.time_window) - now).total_seconds())

class AIGenerator:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        self.hf_model = "CompVis/stable-diffusion-v1-4"  # Changed to more reliable free model
        self.rate_limiter = RateLimiter(max_requests=10, time_window=60)
        
    def format_text_content(self, text: str) -> str:
        text = re.sub(r'\*+', '', text)
        sections = []
        current_section = []
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            else:
                line = re.sub(r'[#_~]', '', line)
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return '\n\n'.join(sections)

    def generate_text(self, prompt: str) -> str:
        try:
            enhanced_prompt = f"""
            Provide information about {prompt}. Include:
            1. A clear introduction
            2. Key characteristics and features
            3. Interesting and unique aspects
            4. Practical applications or relevant details
            
            Please provide the information in clear paragraphs without using any special characters, asterisks, or markdown formatting.
            Keep the tone professional and informative.
            """
            
            response = self.gemini_model.generate_content(enhanced_prompt)
            if not response.text:
                raise ValueError("No text generated from the model")
                
            return self.format_text_content(response.text)
            
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}")
            raise

    def generate_image(self, prompt: str, retry_count=0, max_retries=3) -> bytes:
        try:
            if not self.rate_limiter.can_proceed():
                wait_time = self.rate_limiter.wait_time()
                raise ValueError(f"Rate limit exceeded. Please wait {wait_time:.1f} seconds.")

            enhanced_prompt = (
                f"A highly detailed visualization of {prompt}, "
                "professional quality, sharp focus, perfect lighting"
            )
            
            api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            
            # Optimized parameters for faster generation
            payload = {
                "inputs": enhanced_prompt,
                "parameters": {
                    "num_inference_steps": 20,  # Reduced for faster generation
                    "guidance_scale": 7.0,
                    "width": 512,  # Standard size for faster generation
                    "height": 512
                }
            }
            
            # Check model status first
            status_response = requests.get(api_url, headers=headers, timeout=10)
            if "error" in status_response.json():
                if retry_count < max_retries:
                    time.sleep(5)  # Wait 5 seconds before retrying
                    return self.generate_image(prompt, retry_count + 1, max_retries)
                else:
                    raise ValueError("Model is currently unavailable. Please try again later.")

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = response.text if response.text else "Unknown error"
                if retry_count < max_retries:
                    time.sleep(5)
                    return self.generate_image(prompt, retry_count + 1, max_retries)
                else:
                    raise ValueError(f"Image generation failed after {max_retries} attempts: {error_msg}")
                    
            return response.content
            
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            if retry_count < max_retries:
                time.sleep(5)
                return self.generate_image(prompt, retry_count + 1, max_retries)
            raise

    def generate_both(self, prompt: str):
        try:
            if not self.rate_limiter.can_proceed():
                wait_time = self.rate_limiter.wait_time()
                raise ValueError(f"Rate limit exceeded. Please wait {wait_time:.1f} seconds.")

            # Generate text first as it's more reliable
            text = self.generate_text(prompt)
            
            # Then try image generation with retries
            image = self.generate_image(prompt)
            
            return text, image
        except Exception as e:
            logger.error(f"Combined generation error: {str(e)}")
            raise

# Initialize generator
generator = AIGenerator()

@infosight_ai.route('/')
def index():
    return render_template('infosight_ai.html')

@infosight_ai.route('/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        text = generator.generate_text(data['prompt'])
        return jsonify({'text': text})
    except Exception as e:
        logger.error(f"Text generation endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@infosight_ai.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400

        image_bytes = generator.generate_image(data['prompt'])
        if not image_bytes:
            return jsonify({'error': 'Image generation failed'}), 500

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return jsonify({'image_url': f"data:image/png;base64,{image_base64}"})
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 429
    except Exception as e:
        logger.error(f"Image generation endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@infosight_ai.route('/generate-both', methods=['POST'])
def generate_both():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400

        text, image_bytes = generator.generate_both(data['prompt'])
        response = {'text': text}
        
        if image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            response['image_url'] = f"data:image/png;base64,{image_base64}"

        return jsonify(response)
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 429
    except Exception as e:
        logger.error(f"Combined generation endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500