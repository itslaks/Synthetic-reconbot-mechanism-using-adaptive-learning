from flask import Blueprint, request, jsonify, render_template, current_app
from flask_cors import CORS
from transformers import BlipForConditionalGeneration, BlipProcessor, logging
import torch
from PIL import Image, ExifTags
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
from collections import Counter
import requests

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

# Blueprint setup
snapspeak_ai = Blueprint('snapspeak_ai', __name__, template_folder='templates')
CORS(snapspeak_ai)

# Global configurations
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.85
COLOR_CLUSTER_COUNT = 5
IMAGE_RESIZE_DIMENSION = 150
DARK_PIXEL_THRESHOLD = 10

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

# Initialize face detection
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace not available")

def get_labeled_gps(gps_info):
    """Convert GPS information to readable format"""
    labeled = {}
    for key, value in gps_info.items():
        if key in GPSTAGS:
            tag_name = GPSTAGS[key]
            if isinstance(value, tuple) and len(value) > 0:
                if tag_name in ['GPSLatitude', 'GPSLongitude']:
                    try:
                        degrees = float(value[0][0]) / float(value[0][1])
                        minutes = float(value[1][0]) / float(value[1][1])
                        seconds = float(value[2][0]) / float(value[2][1])
                        labeled[tag_name] = f"{degrees:.6f}° {minutes:.4f}' {seconds:.2f}\""
                    except:
                        labeled[tag_name] = value
                else:
                    labeled[tag_name] = value
            else:
                labeled[tag_name] = value
    return labeled

def format_binary_data(data):
    """Format binary data into readable format"""
    if isinstance(data, bytes):
        try:
            decoded = data.decode('utf-8')
            return decoded if all(32 <= ord(c) <= 126 for c in decoded) else f"HEX: {binascii.hexlify(data).decode('ascii')}"
        except UnicodeDecodeError:
            return f"HEX: {binascii.hexlify(data).decode('ascii')}"
    return str(data)

def metadata_analysis(image):
    """Extract and analyze image metadata"""
    try:
        exif_data = {
            'Format': image.format,
            'Mode': image.mode,
            'Size': f"{image.width}x{image.height}",
            'Bits_Per_Channel': getattr(image, 'bits', 8)
        }
        
        # Extract EXIF data
        info = image.getexif()
        if info:
            for tag_id, value in info.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'GPSInfo':
                    gps_data = get_labeled_gps(value)
                    exif_data.update({f'GPS_{k}': v for k, v in gps_data.items()})
                else:
                    exif_data[tag] = value

        # Get ICC Profile information
        if 'icc_profile' in image.info:
            exif_data['ICC_Profile'] = format_binary_data(image.info['icc_profile'])

        # Extract quantization tables if available
        if hasattr(image, "quantization"):
            exif_data['Quantization_Tables'] = image.quantization

        # Add additional image info
        for key, value in image.info.items():
            if key not in ['exif', 'icc_profile']:
                exif_data[key] = value

        # Calculate image statistics
        if image.mode == 'RGB':
            r, g, b = image.split()
            exif_data.update({
                'Entropy_R': f"{r.entropy():.2f}",
                'Entropy_G': f"{g.entropy():.2f}",
                'Entropy_B': f"{b.entropy():.2f}",
                'Color_Range': str(image.getextrema())
            })

        return {k: format_binary_data(v) for k, v in exif_data.items()}
    except Exception as e:
        print(f"Error in metadata analysis: {str(e)}")
        return {}

def enhanced_face_detection(image):
    """Enhanced face detection using DeepFace or OpenCV fallback"""
    try:
        np_image = np.array(image)
        
        if len(np_image.shape) == 2:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
        elif np_image.shape[2] == 4:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)

        face_locations = []

        if DEEPFACE_AVAILABLE:
            try:
                faces = DeepFace.extract_faces(
                    np_image,
                    detector_backend='retinaface',
                    enforce_detection=False,
                    align=True
                )

                for face in faces:
                    confidence = face.get('confidence', 0)
                    if confidence > FACE_DETECTION_CONFIDENCE_THRESHOLD:
                        facial_area = face['facial_area']
                        face_locations.append({
                            'x': int(facial_area['x']),
                            'y': int(facial_area['y']),
                            'width': int(facial_area['w']),
                            'height': int(facial_area['h']),
                            'confidence': float(confidence),
                            'detector': 'retinaface'
                        })
            except Exception as e:
                print(f"DeepFace detection failed, falling back to OpenCV: {str(e)}")
                DEEPFACE_AVAILABLE = False

        if not DEEPFACE_AVAILABLE:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                face_locations.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': 0.8,
                    'detector': 'opencv'
                })

        return {
            'count': len(face_locations),
            'locations': face_locations,
            'success': True
        }

    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        return {
            'count': 0,
            'locations': [],
            'success': False,
            'error': str(e)
        }

def color_analysis(image):
    """Analyze dominant colors using K-means clustering"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        thumb = image.copy()
        thumb.thumbnail((IMAGE_RESIZE_DIMENSION, IMAGE_RESIZE_DIMENSION))
        
        pixels = np.array(thumb).reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=COLOR_CLUSTER_COUNT, random_state=0, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        total_pixels = sum(counts)
        color_info = []
        
        sorted_indices = np.argsort(counts)[::-1]
        
        for idx in sorted_indices:
            r, g, b = map(int, colors[idx])
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            percentage = (counts[idx] / total_pixels) * 100
            
            color_info.append({
                'hex': hex_color,
                'rgb': f'rgb({r},{g},{b})',
                'percentage': round(percentage, 1)
            })
        
        return color_info
    except Exception as e:
        print(f"Error in color analysis: {str(e)}")
        return []

def detect_steganography(image):
    """Advanced steganography detection"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        pixels = np.array(image)
        lsb = pixels & 1
        
        # Calculate entropy of LSB
        lsb_entropy = cv2.calcHist([lsb.ravel()], [0], None, [2], [0, 2])
        lsb_entropy = float(sum(-p * np.log2(p + 1e-10) for p in lsb_entropy))
        
        # Analyze hidden pixels
        gray_image = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        hidden_pixel_count = int(np.sum(gray_image < DARK_PIXEL_THRESHOLD))
        
        threshold = 0.97
        confidence = min((lsb_entropy / threshold) * 100, 100)
        
        return {
            'detected': lsb_entropy > threshold,
            'confidence': float(confidence),
            'methods': ['LSB Analysis'] if lsb_entropy > threshold else [],
            'hidden_pixels': hidden_pixel_count,
            'hidden_pixel_threshold': DARK_PIXEL_THRESHOLD
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

@torch.no_grad()
def generate_caption(image):
    """Generate image caption using BLIP model"""
    try:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
        return processor.decode(output_ids[0], skip_special_tokens=True)
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
    """Generate perceptual hash"""
    return str(imagehash.average_hash(image))

@snapspeak_ai.route('/')
def index():
    return render_template('snapspeak.html')

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
        
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Perform all analyses
        analysis_results = {
            'caption': generate_caption(image),
            'metadata': metadata_analysis(image),
            'image_hash': image_hash(image),
            'sha256_digest': generate_image_digest(image_bytes),
            'dominant_colors': color_analysis(image),
            'faces': enhanced_face_detection(image),
            'steganography': detect_steganography(image),
            'processing_time': float(time.time() - start_time)
        }
        
        return jsonify(analysis_results)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in analyze_image: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            'error': str(e),
            'traceback': error_trace if current_app.debug else 'Enable debug mode for traceback'
        }), 500

if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(snapspeak_ai, url_prefix='/snapspeak_ai')
    app.run(debug=True)
