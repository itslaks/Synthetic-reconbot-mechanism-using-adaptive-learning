from flask import Blueprint, request, jsonify, render_template
from flask_cors import CORS
import requests
import base64
import google.generativeai as genai
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from collections import deque
import threading

# Create blueprint
infosight_ai = Blueprint('infosight_ai', __name__, template_folder='templates')
logger = logging.getLogger(__name__)
CORS(infosight_ai)

# API Configuration
GEMINI_API_KEY = ' '  # Use your own Gemini API key
HF_API_TOKEN = ' '  # Use your own Hugging Face API key

# Configure AI model
genai.configure(api_key=GEMINI_API_KEY)

class RateLimiter:
    """Controls request rate to avoid hitting API limits."""
    def __init__(self, max_requests=10, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()

    def can_proceed(self):
        """Checks if a request can be processed."""
        now = datetime.now()
        with self.lock:
            while self.requests and self.requests[0] < now - timedelta(seconds=self.time_window):
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    def wait_time(self):
        """Returns the wait time before making another request."""
        if not self.requests:
            return 0
        now = datetime.now()
        oldest_request = self.requests[0]
        return max(0, (oldest_request + timedelta(seconds=self.time_window) - now).total_seconds())

class AIGenerator:
    def __init__(self):
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Updated to better models
        self.primary_image_model = "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL for higher quality
        self.fallback_image_model = "SG161222/Realistic_Vision_V5.1_noVAE"  # Fallback to another high-quality model
        
        self.rate_limiter = RateLimiter(max_requests=15, time_window=60)

    def format_text_content(self, text: str) -> str:
        """Cleans up generated text by removing special characters."""
        text = re.sub(r'\*+', '', text)
        sections = []
        current_section = []

        for line in text.strip().split('\n'):
            line = re.sub(r'[#_~]', '', line).strip()
            if line:
                current_section.append(line)
            elif current_section:
                sections.append('\n'.join(current_section))
                current_section = []

        if current_section:
            sections.append('\n'.join(current_section))

        return '\n\n'.join(sections)

    def generate_text(self, prompt: str) -> str:
        """Generates text using Google Gemini AI."""
        try:
            if not self.rate_limiter.can_proceed():
                wait_time = self.rate_limiter.wait_time()
                raise ValueError(f"Rate limit exceeded. Please wait {wait_time:.1f} seconds.")

            enhanced_prompt = f"""
            Provide accurate, well-structured information about {prompt}, including:
            1. Clear introduction
            2. Key features and details
            3. Interesting insights
            4. Practical applications
            """
            response = self.gemini_model.generate_content(enhanced_prompt)
            if not response.text:
                raise ValueError("No text generated from Gemini AI.")
            
            return self.format_text_content(response.text)
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}")
            return f"Error generating text: {str(e)}"

    def generate_image(self, prompt: str) -> bytes:
        """Generates image using Hugging Face image models."""
        try:
            if not self.rate_limiter.can_proceed():
                wait_time = self.rate_limiter.wait_time()
                logger.warning(f"Rate limit exceeded for image generation. Wait time: {wait_time:.1f} seconds.")
                return None

            enhanced_prompt = (
                f"Ultra-detailed, professional photorealistic image of {prompt}. "
                "8K resolution, perfect lighting, sharp focus, intricate details, "
                "hyperrealistic texture, cinematic composition, professional photography."
            )
            
            negative_prompt = (
                "blurry, distorted, low quality, low resolution, draft, ugly, unrealistic, "
                "text, watermark, signature, bad proportions, deformed, unrealistic shadows"
            )
            
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            
            # First try the primary high-quality model
            try:
                api_url = f"https://api-inference.huggingface.co/models/{self.primary_image_model}"
                
                payload = {
                    "inputs": enhanced_prompt,
                    "parameters": {
                        "negative_prompt": negative_prompt,
                        "num_inference_steps": 50,
                        "guidance_scale": 7.5,
                        "width": 1024,
                        "height": 1024
                    }
                }

                logger.info(f"Requesting image from primary model: {self.primary_image_model}")
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)

                if response.status_code == 200:
                    logger.info("Primary model successfully generated image")
                    return response.content
                else:
                    logger.warning(f"Primary model failed, trying fallback. Status: {response.status_code}, Error: {response.text}")
            except Exception as e:
                logger.warning(f"Primary model request failed with exception: {str(e)}")
            
            # Fallback to secondary model
            try:
                api_url = f"https://api-inference.huggingface.co/models/{self.fallback_image_model}"
                payload = {
                    "inputs": enhanced_prompt,
                    "parameters": {
                        "negative_prompt": negative_prompt,
                        "num_inference_steps": 40,
                        "guidance_scale": 7.0
                    }
                }
                
                logger.info(f"Requesting image from fallback model: {self.fallback_image_model}")
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    logger.info("Fallback model successfully generated image")
                    return response.content
                else:
                    error_message = "Unknown error"
                    try:
                        error_message = response.json().get("error", "Unknown error")
                    except:
                        error_message = response.text
                    logger.error(f"Fallback model failed. Status: {response.status_code}, Error: {error_message}")
                    return None
            except Exception as e:
                logger.error(f"Fallback model request failed with exception: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Image generation request error: {str(e)}")
            return None

    def generate_both(self, prompt: str):
        """Generates both text and image simultaneously."""
        if not self.rate_limiter.can_proceed():
            wait_time = self.rate_limiter.wait_time()
            return f"Rate limit exceeded. Please wait {wait_time:.1f} seconds.", None

        logger.info(f"Starting parallel generation for prompt: {prompt}")
        
        text = None
        image = None
        
        try:
            # Use ThreadPoolExecutor for parallel execution
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                text_future = executor.submit(self.generate_text, prompt)
                image_future = executor.submit(self.generate_image, prompt)
                
                # Wait for both to complete with timeout
                text = text_future.result(timeout=60)
                image = image_future.result(timeout=60)
                
                logger.info(f"Text generation completed: {'Success' if text else 'Failed'}")
                logger.info(f"Image generation completed: {'Success' if image else 'Failed'}")
        except Exception as e:
            logger.error(f"Exception in generate_both: {str(e)}")
            if not text:
                text = f"Error generating text: {str(e)}"
        
        return text, image

# Initialize the generator instance
generator = AIGenerator()

@infosight_ai.route('/')
def index():
    """Renders the main UI page."""
    return render_template('infosight_ai.html')

@infosight_ai.route('/generate-text', methods=['POST'])
def generate_text_endpoint():
    """API endpoint to generate text."""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        text = generator.generate_text(data['prompt'])
        return jsonify({'text': text})
    except Exception as e:
        logger.error(f"Error in generate-text endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@infosight_ai.route('/generate-image', methods=['POST'])
def generate_image_endpoint():
    """API endpoint to generate image."""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        image_bytes = generator.generate_image(data['prompt'])
        
        if image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return jsonify({'image_url': f"data:image/png;base64,{image_base64}"})
        else:
            return jsonify({'error': 'Image generation failed'}), 500
    except Exception as e:
        logger.error(f"Error in generate-image endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@infosight_ai.route('/generate-both', methods=['POST'])
def generate_both_endpoint():
    """API endpoint to generate both text and image."""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        prompt = data['prompt']
        logger.info(f"Received generate-both request for prompt: {prompt}")
        
        text, image_bytes = generator.generate_both(prompt)
        
        response = {'text': text}
        
        if image_bytes:
            try:
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                response['image_url'] = f"data:image/png;base64,{image_base64}"
                logger.info("Successfully encoded image to base64")
            except Exception as e:
                logger.error(f"Failed to encode image: {str(e)}")
                response['image_error'] = f"Image encoding failed: {str(e)}"
        else:
            logger.warning("No image bytes returned from generate_both")
            response['image_error'] = 'Image generation failed'

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in generate-both endpoint: {str(e)}")
        return jsonify({'error': str(e), 'location': 'generate_both_route'}), 500
