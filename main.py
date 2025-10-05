import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the trained YOLO model
model_path = os.path.join('models', 'best.pt')
model = YOLO(model_path)

def _format_class_name(raw_name: str) -> str:
    """Make raw model class names human readable.

    Examples:
    - "VeryMildDemented" -> "Very Mild Demented"
    - "NonDemented" -> "Non Demented"
    - "mild_demented" -> "Mild Demented"
    """
    name = raw_name.replace('_', ' ')
    # Insert spaces before capital letters that follow lowercase letters
    formatted = []
    for ch in name:
        if formatted and ch.isupper() and formatted[-1].islower():
            formatted.append(' ')
        formatted.append(ch)
    return (''.join(formatted)).strip().title().replace('Non Demented', 'Non Demented')

# Build classes mapping directly from the loaded model to ensure correct indices
_model_names = getattr(model, 'names', None)
if isinstance(_model_names, dict):
    ALZHEIMER_CLASSES = {int(i): _format_class_name(n) for i, n in _model_names.items()}
elif isinstance(_model_names, (list, tuple)):
    ALZHEIMER_CLASSES = {int(i): _format_class_name(n) for i, n in enumerate(_model_names)}
else:
    # Fallback to a reasonable default if names are unavailable
    ALZHEIMER_CLASSES = {
        0: 'Very Mild Demented',
        1: 'Non Demented',
        2: 'Mild Demented',
        3: 'Moderate Demented'
    }

def process_image(image_bytes):
    """Process uploaded image and make prediction"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to numpy array for OpenCV
        image_np = np.array(image)
        
        # If image has 4 channels (RGBA), convert to RGB
        if image_np.shape[-1] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        elif len(image_np.shape) == 3 and image_np.shape[-1] == 3:
            # Convert RGB to BGR for OpenCV
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Run YOLO prediction
        results = model(image_np)
        
        # Extract prediction results
        predictions = []
        for result in results:
            if result.probs is not None:
                # Classification result
                confidence_scores = result.probs.data.cpu().numpy()
                predicted_class_idx = np.argmax(confidence_scores)
                confidence = float(confidence_scores[predicted_class_idx])
                
                prediction = {
                    'class_name': ALZHEIMER_CLASSES.get(predicted_class_idx, f'Class {predicted_class_idx}'),
                    'class_index': int(predicted_class_idx),
                    'confidence': confidence,
                    'all_confidences': {
                        ALZHEIMER_CLASSES.get(i, f'Class {i}'): float(conf) 
                        for i, conf in enumerate(confidence_scores)
                    }
                }
                predictions.append(prediction)
            
            # If detection results are available
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_idx = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    prediction = {
                        'class_name': ALZHEIMER_CLASSES.get(cls_idx, f'Class {cls_idx}'),
                        'class_index': cls_idx,
                        'confidence': confidence,
                        'bbox': box.xyxy.cpu().numpy().tolist()[0] if box.xyxy is not None else None
                    }
                    predictions.append(prediction)
        
        # If no predictions, use the highest confidence classification
        if not predictions and results:
            result = results[0]
            if hasattr(result, 'probs') and result.probs is not None:
                confidence_scores = result.probs.data.cpu().numpy()
                predicted_class_idx = np.argmax(confidence_scores)
                confidence = float(confidence_scores[predicted_class_idx])
                
                predictions.append({
                    'class_name': ALZHEIMER_CLASSES.get(predicted_class_idx, f'Class {predicted_class_idx}'),
                    'class_index': int(predicted_class_idx),
                    'confidence': confidence,
                    'all_confidences': {
                        ALZHEIMER_CLASSES.get(i, f'Class {i}'): float(conf) 
                        for i, conf in enumerate(confidence_scores)
                    }
                })
        
        return predictions
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Process image and get prediction
        predictions = process_image(image_bytes)
        
        if predictions is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        if not predictions:
            return jsonify({'error': 'No predictions could be made for this image'}), 400
        
        # Convert image to base64 for display
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'image': image_b64,
            'filename': file.filename
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/classes')
def get_classes():
    """Get available classification classes"""
    return jsonify({'classes': ALZHEIMER_CLASSES})

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Alzheimer's Classification Server...")
    print("Available classes:", ALZHEIMER_CLASSES)
    app.run(debug=True, host='0.0.0.0', port=5000)
