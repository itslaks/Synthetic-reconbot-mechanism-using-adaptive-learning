from flask import Blueprint, request, jsonify, render_template, current_app, send_file
from flask_cors import CORS
from transformers import BlipForConditionalGeneration, BlipProcessor, logging
import torch
from PIL import Image, ExifTags, ImageEnhance
from PIL.ExifTags import TAGS, GPSTAGS
import io
import time
import imagehash
import traceback
import warnings
import cv2
import numpy as np
import binascii
from sklearn.cluster import KMeans
import hashlib
import os
import json
from collections import Counter
import requests
from deepface import DeepFace
from ratelimit import limits, sleep_and_retry
import redis
from werkzeug.utils import secure_filename
import magic
import piexif
import pywt
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

# Blueprint setup
snapspeak_ai = Blueprint('snapspeak_ai', __name__, template_folder='templates')
CORS(snapspeak_ai)

# Redis setup for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)
CACHE_EXPIRATION = 3600  # 1 hour

# Rate limiting setup
CALLS_PER_MINUTE = 30

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.85
COLOR_CLUSTER_COUNT = 5
IMAGE_RESIZE_DIMENSION = 150
DARK_PIXEL_THRESHOLD = 10

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

class ImageAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def assess_image_quality(self, image):
        """Assess image quality metrics"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Calculate blur detection
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate noise estimation
            noise_sigma = np.std(gray)
            
            # Calculate contrast
            contrast = ImageEnhance.Contrast(image).enhance(1.0)
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            quality_score = min(100, (laplacian_var / 500) * 100)
            
            return {
                'sharpness': float(laplacian_var),
                'noise_level': float(noise_sigma),
                'contrast_score': float(np.std(contrast)),
                'brightness_score': float(brightness),
                'is_blurry': laplacian_var < 100,
                'quality_score': float(quality_score),
                'assessment': {
                    'sharpness': 'Good' if laplacian_var > 100 else 'Poor',
                    'noise': 'Low' if noise_sigma < 30 else 'High',
                    'brightness': 'Good' if 40 < brightness < 200 else 'Poor'
                }
            }
        except Exception as e:
            print(f"Error in quality assessment: {str(e)}")
            return None

    def detect_tampering(self, image):
        """Detect potential image tampering"""
        try:
            # Convert image to numpy array
            img_array = np.array(image)
            
            # Error Level Analysis (ELA)
            temp_filename = 'temp.jpg'
            image.save(temp_filename, quality=95)
            ela_image = Image.open(temp_filename)
            os.remove(temp_filename)
            
            # Calculate difference
            ela_array = np.array(ela_image)
            diff = np.abs(img_array - ela_array)
            
            # Analyze differences
            threshold = 40
            tampering_score = np.mean(diff)
            
            # Additional noise analysis
            noise_pattern = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            noise_diff = np.abs(img_array - noise_pattern)
            noise_score = np.mean(noise_diff)
            
            return {
                'tampering_detected': tampering_score > threshold,
                'tampering_score': float(tampering_score),
                'noise_analysis_score': float(noise_score),
                'analysis_method': ['Error Level Analysis', 'Noise Pattern Analysis'],
                'confidence': min(100, (tampering_score / threshold) * 100),
                'risk_level': 'High' if tampering_score > threshold * 1.5 else 'Medium' if tampering_score > threshold else 'Low'
            }
        except Exception as e:
            print(f"Error in tampering detection: {str(e)}")
            return None

    def analyze_facial_features(self, image):
        """Enhanced facial feature analysis"""
        try:
            result = DeepFace.analyze(
                np.array(image),
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False
            )
            
            if isinstance(result, list):
                result = result[0]
            
            # Additional facial landmark detection
            face_locations = self.detect_facial_landmarks(image)
            
            return {
                'age': result.get('age'),
                'gender': result.get('gender'),
                'dominant_race': result.get('dominant_race'),
                'dominant_emotion': result.get('dominant_emotion'),
                'emotion_probabilities': result.get('emotion'),
                'facial_landmarks': face_locations,
                'confidence_scores': {
                    'age_estimation': min(100, abs(100 - abs(result.get('age', 0) - 25) / 50 * 100)),
                    'gender_recognition': result.get('gender_probability', 0) * 100,
                    'emotion_recognition': max(result.get('emotion', {}).values()) * 100
                }
            }
        except Exception as e:
            print(f"Error in facial feature analysis: {str(e)}")
            return None

    def detect_facial_landmarks(self, image):
        """Detect facial landmarks using dlib"""
        try:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            landmarks = []
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                face_landmarks = {
                    'face_position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'eyes': [{'x': int(ex), 'y': int(ey), 'width': int(ew), 'height': int(eh)} 
                            for (ex, ey, ew, eh) in eyes]
                }
                landmarks.append(face_landmarks)
            
            return landmarks
        except Exception as e:
            print(f"Error in facial landmark detection: {str(e)}")
            return []

    def detect_watermark(self, image):
        """Detect digital watermarks"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Apply DWT
            coeffs = pywt.dwt2(gray, 'haar')
            
            # Analyze coefficients
            threshold = np.mean(np.abs(coeffs[0])) * 2
            watermark_detected = np.sum(np.abs(coeffs[0]) > threshold) > 100
            
            # Additional analysis
            edge_detection = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edge_detection > 0) / (gray.shape[0] * gray.shape[1])
            
            return {
                'watermark_detected': watermark_detected,
                'confidence': float(min(100, (np.sum(np.abs(coeffs[0]) > threshold) / 100) * 100)),
                'analysis_method': ['DWT Analysis', 'Edge Detection'],
                'edge_density': float(edge_density),
                'detection_details': {
                    'coefficient_strength': float(np.mean(np.abs(coeffs[0]))),
                    'edge_complexity': 'High' if edge_density > 0.1 else 'Low'
                }
            }
        except Exception as e:
            print(f"Error in watermark detection: {str(e)}")
            return None

def metadata_analysis(image):
    """Extract and analyze image metadata"""
    try:
        exif_data = {
            'Format': image.format,
            'Mode': image.mode,
            'Size': f"{image.width}x{image.height}",
            'Bits_Per_Channel': getattr(image, 'bits', 8),
            'Creation_Time': None,
            'Camera_Info': {},
            'GPS_Info': {},
            'Software_Info': {}
        }
        
        # Extract EXIF data
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'GPSInfo':
                    gps_data = {}
                    for gps_tag in value:
                        sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                        gps_data[sub_tag] = value[gps_tag]
                    exif_data['GPS_Info'] = gps_data
                elif tag == 'DateTime':
                    exif_data['Creation_Time'] = value
                elif tag in ['Make', 'Model', 'ExposureTime', 'FNumber', 'ISOSpeedRatings']:
                    exif_data['Camera_Info'][tag] = value
                elif tag in ['Software', 'ProcessingSoftware']:
                    exif_data['Software_Info'][tag] = value
                else:
                    exif_data[tag] = value

        # Additional image analysis
        if image.mode == 'RGB':
            r, g, b = image.split()
            exif_data['Color_Stats'] = {
                'Red_Mean': float(np.mean(r)),
                'Green_Mean': float(np.mean(g)),
                'Blue_Mean': float(np.mean(b)),
                'Color_Depth': image.bits,
                'Unique_Colors': len(set(image.getdata()))
            }

        return exif_data
    except Exception as e:
        print(f"Error in metadata analysis: {str(e)}")
        return {}

def color_analysis(image):
    """Analyze dominant colors using K-means clustering"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image for faster processing
        thumb = image.copy()
        thumb.thumbnail((IMAGE_RESIZE_DIMENSION, IMAGE_RESIZE_DIMENSION))
        
        # Convert to numpy array
        pixels = np.array(thumb).reshape(-1, 3)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=COLOR_CLUSTER_COUNT, random_state=0, n_init=10)
        kmeans.fit(pixels)
        
        # Get color information
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        # Calculate percentages and create color information
        total_pixels = sum(counts)
        color_info = []
        
        # Sort colors by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        for idx in sorted_indices:
            r, g, b = map(int, colors[idx])
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            percentage = (counts[idx] / total_pixels) * 100
            
            # Calculate brightness and saturation
            brightness = (r + g + b) / 3
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            saturation = 0 if max_val == 0 else (max_val - min_val) / max_val * 100
            
            color_info.append({
                'hex': hex_color,
                'rgb': f'rgb({r},{g},{b})',
                'percentage': round(percentage, 1),
                'brightness': round(brightness, 1),
                'saturation': round(saturation, 1),
                'is_grayscale': abs(r - g) < 5 and abs(g - b) < 5 and abs(r - b) < 5,
                'color_category': categorize_color(r, g, b)
            })
        
        return color_info
    except Exception as e:
        print(f"Error in color analysis: {str(e)}")
        return []

def categorize_color(r, g, b):
    """Categorize RGB color into basic color name"""
    if max(r, g, b) < 32:
        return "Black"
    if min(r, g, b) > 223:
        return "White"
    
    colors = {
        "Red": r > max(g, b) + 32,
        "Green": g > max(r, b) + 32,
        "Blue": b > max(r, g) + 32,
        "Yellow": r > 200 and g > 200 and b < 100,
        "Purple": r > 128 and b > 128 and g < 128,
        "Orange": r > 200 and g > 100 and b < 100,
        "Brown": r > g and g > b and r < 200,
        "Gray": abs(r - g) < 32 and abs(g - b) < 32 and abs(r - b) < 32
    }
    
    for color_name, condition in colors.items():
        if condition:
            return color_name
    return "Mixed"

def detect_steganography(image):
    """Advanced steganography detection"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        pixels = np.array(image)
        
        # LSB Analysis
        lsb = pixels & 1
        lsb_entropy = cv2.calcHist([lsb.ravel()], [0], None, [2], [0, 2])
        lsb_entropy = float(sum(-p * np.log2(p + 1e-10) for p in lsb_entropy))
        
        # Chi-Square Analysis
        chi_square = np.sum((lsb - np.mean(lsb)) ** 2 / (np.mean(lsb) + 1e-10))
        
        # Hidden Pixel Analysis
        # Hidden Pixel Analysis
        gray_image = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        hidden_pixel_count = int(np.sum(gray_image < DARK_PIXEL_THRESHOLD))
        
        # Pattern Analysis
        patterns = {
            'repeated_patterns': detect_repeated_patterns(lsb),
            'suspicious_regions': detect_suspicious_regions(gray_image)
        }
        
        # Calculate confidence scores
        lsb_threshold = 0.97
        chi_threshold = 1000
        pattern_threshold = 0.5
        
        confidence_scores = {
            'lsb_analysis': min((lsb_entropy / lsb_threshold) * 100, 100),
            'chi_square': min((chi_square / chi_threshold) * 100, 100),
            'pattern_analysis': patterns['repeated_patterns'] * 100
        }
        
        overall_confidence = np.mean(list(confidence_scores.values()))
        
        return {
            'detected': overall_confidence > 70,
            'confidence': float(overall_confidence),
            'methods': [
                'LSB Analysis',
                'Chi-Square Analysis',
                'Pattern Detection'
            ] if overall_confidence > 70 else [],
            'hidden_pixels': hidden_pixel_count,
            'hidden_pixel_threshold': DARK_PIXEL_THRESHOLD,
            'analysis_details': {
                'lsb_entropy': float(lsb_entropy),
                'chi_square_value': float(chi_square),
                'pattern_strength': patterns['repeated_patterns'],
                'suspicious_regions': patterns['suspicious_regions']
            },
            'confidence_scores': confidence_scores
        }
    except Exception as e:
        print(f"Error in steganography detection: {str(e)}")
        return {
            'detected': False,
            'confidence': 0,
            'methods': [],
            'hidden_pixels': 0,
            'error': str(e)
        }

def detect_repeated_patterns(lsb_data):
    """Detect repeated patterns in LSB data"""
    try:
        # Convert to 1D array
        data = lsb_data.ravel()
        
        # Look for repeating sequences
        sequence_length = 8
        sequences = {}
        
        for i in range(len(data) - sequence_length):
            seq = tuple(data[i:i+sequence_length])
            sequences[seq] = sequences.get(seq, 0) + 1
        
        # Calculate pattern strength
        max_repetitions = max(sequences.values())
        pattern_strength = min(1.0, max_repetitions / (len(data) / sequence_length))
        
        return float(pattern_strength)
    except Exception as e:
        print(f"Error in pattern detection: {str(e)}")
        return 0.0

def detect_suspicious_regions(gray_image):
    """Detect suspicious regions in the image"""
    try:
        # Apply threshold
        _, binary = cv2.threshold(gray_image, DARK_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        suspicious_regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            if area > 100:  # Minimum size threshold
                suspicious_regions.append({
                    'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'area': int(area),
                    'centroid': {'x': int(centroids[i][0]), 'y': int(centroids[i][1])}
                })
        
        return suspicious_regions
    except Exception as e:
        print(f"Error in suspicious region detection: {str(e)}")
        return []

@torch.no_grad()
def generate_caption(image):
    """Generate image caption using BLIP model"""
    try:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(
            pixel_values, 
            max_length=50,
            num_beams=5,
            length_penalty=1.0,
            repetition_penalty=1.2,
            temperature=0.7
        )
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Post-process caption
        caption = caption.capitalize()
        if not caption.endswith('.'):
            caption += '.'
            
        return caption
    except Exception as e:
        print(f"Error in caption generation: {str(e)}")
        return "Error generating caption"

def generate_image_digest(image_bytes):
    """Generate SHA-256 hash of the image"""
    try:
        return hashlib.sha256(image_bytes).hexdigest()
    except Exception as e:
        print(f"Error generating image digest: {str(e)}")
        return None

def image_hash(image):
    """Generate perceptual hash using multiple algorithms"""
    try:
        return {
            'average_hash': str(imagehash.average_hash(image)),
            'perceptual_hash': str(imagehash.phash(image)),
            'difference_hash': str(imagehash.dhash(image)),
            'wavelet_hash': str(imagehash.whash(image))
        }
    except Exception as e:
        print(f"Error generating image hash: {str(e)}")
        return None

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
def rate_limited_analyze(image_bytes):
    """Rate-limited image analysis"""
    # Generate cache key
    cache_key = hashlib.md5(image_bytes).hexdigest()
    
    # Check cache
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    # Perform analysis
    analyzer = ImageAnalyzer()
    image = Image.open(io.BytesIO(image_bytes))
    
    result = {
        'basic_info': {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'filename': secure_filename(request.files['file'].filename)
        },
        'quality_metrics': analyzer.assess_image_quality(image),
        'facial_analysis': analyzer.analyze_facial_features(image),
        'tampering_detection': analyzer.detect_tampering(image),
        'watermark_detection': analyzer.detect_watermark(image),
        'caption': generate_caption(image),
        'metadata': metadata_analysis(image),
        'image_hash': image_hash(image),
        'sha256_digest': generate_image_digest(image_bytes),
        'dominant_colors': color_analysis(image),
        'faces': enhanced_face_detection(image),
        'steganography': detect_steganography(image),
        'analysis_timestamp': datetime.utcnow().isoformat()
    }
    
    # Cache result
    redis_client.setex(cache_key, CACHE_EXPIRATION, json.dumps(result))
    
    return result

@snapsepak_ai.route('/')
def index:
   render_template(snapspeak.html)

@snapspeak_ai.route('/api/analyze/', methods=['POST'])
def analyze_image():
    """Main image analysis endpoint"""
    try:
        start_time = time.time()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No selected file'}), 400
        
        # Validate file type and size
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        image_bytes = file.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large'}), 400
        
        # Validate file content
        mime = magic.Magic(mime=True)
        file_type = mime.from_buffer(image_bytes)
        if not file_type.startswith('image/'):
            return jsonify({'error': 'Invalid file content'}), 400
        
        # Perform analysis
        result = rate_limited_analyze(image_bytes)
        result['processing_time'] = float(time.time() - start_time)
        
        return jsonify(result)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in analyze_image: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            'error': str(e),
            'traceback': error_trace if current_app.debug else 'Enable debug mode for traceback'
        }), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(snapspeak_ai, url_prefix='/snapspeak_ai')
    app.run(debug=True)
