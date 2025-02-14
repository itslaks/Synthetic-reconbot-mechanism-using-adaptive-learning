from flask import Flask, Blueprint, render_template, request, jsonify
from PIL import Image, ImageStat
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import io
import numpy as np

trueshot_ai = Blueprint('trueshot_ai', __name__, url_prefix='/trueshot_ai')

class ImageClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._setup_model()
        self.classes = ['AI-generated', 'Real']
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _setup_model(self):
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 2)
        )
        model.load_state_dict(torch.load('best_model9.pth', map_location=self.device))
        model.eval()
        return model.to(self.device)

    def _analyze_image_properties(self, image):
        # Convert to RGB if needed
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
            
        # Basic properties
        width, height = image.size
        gray_image = image.convert('L')
        noise_level = float(np.std(np.array(gray_image)))
        
        # Advanced analysis
        stat = ImageStat.Stat(image)
        mean_colors = stat.mean
        std_colors = stat.stddev
        
        # Color variance analysis
        color_variance = sum(std_colors) / 3
        
        return {
            'dimensions': (width, height),
            'noise_level': noise_level,
            'color_variance': color_variance,
            'mean_brightness': sum(mean_colors) / 3
        }

    def _get_detailed_reasoning(self, prediction, confidence, properties):
        reasons = []
        noise = properties['noise_level']
        color_var = properties['color_variance']
        brightness = properties['mean_brightness']
        
        if confidence < 0.65:
            reasons.append("Analysis is inconclusive due to mixed characteristics")
            reasons.append(f"Confidence level ({confidence*100:.1f}%) is below threshold for definitive classification")
            return reasons

        if prediction == 'AI-generated':
            if noise < 15:
                reasons.append("Unusually smooth textures and low noise patterns typical of AI generation")
            if color_var < 30:
                reasons.append("Highly consistent color distributions across the image")
            if confidence > 0.85:
                reasons.append("Strong indicators of AI generation patterns")
            reasons.append(f"Image noise level ({noise:.1f}) is lower than typical natural photos")
            
        else:  # Real
            if noise > 15:
                reasons.append("Natural noise patterns consistent with real photography")
            if color_var > 30:
                reasons.append("Natural variation in color distribution")
            if confidence > 0.85:
                reasons.append("Strong indicators of natural photo characteristics")
            reasons.append(f"Image noise level ({noise:.1f}) matches typical camera sensor patterns")

        return reasons

    def classify_image(self, image):
        try:
            # Analyze image properties
            properties = self._analyze_image_properties(image)
            
            # Prepare image for model
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                
            confidence_value = float(confidence.item())
            prediction = self.classes[predicted]
            
            # Get detailed reasoning
            reasoning = self._get_detailed_reasoning(prediction, confidence_value, properties)
            
            return {
                'prediction': prediction,
                'confidence': confidence_value,
                'reasoning': reasoning
            }
            
        except Exception as e:
            raise Exception(f"Error during image classification: {str(e)}")

@trueshot_ai.route('/')
def index():
    return render_template('trueshot.html')

@trueshot_ai.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No selected file'
            }), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Please upload JPG, JPEG or PNG'
            }), 400
        
        # Read and validate image
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({
                'status': 'error',
                'message': 'Empty file uploaded'
            }), 400
            
        # Open and process image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        # Get classification results
        classifier_instance = ImageClassifier()
        result = classifier_instance.classify_image(image)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(trueshot_ai)
    app.run(debug=True, port=5000)