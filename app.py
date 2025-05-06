from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import io
import base64
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained model
def load_model():
    # Load ResNet50 model pre-trained on ImageNet
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model

model = load_model()

# Load class labels for ImageNet
def load_class_mapping():
    # This is a simplified mapping of food categories to calorie ranges
    # In a production application, you would use a more comprehensive database
    food_calories = {
        'apple': 95,
        'banana': 105,
        'orange': 62,
        'pizza': 285,
        'hamburger': 354,
        'hotdog': 151,
        'sandwich': 240,
        'donut': 253,
        'cake': 367,
        'ice cream': 137,
        'broccoli': 34,
        'carrot': 41,
        'tomato': 22,
        'salad': 152,
        'pasta': 131,
        'rice': 130,
        'bread': 79,
        'steak': 271,
        'chicken': 165,
        'fish': 136,
        'egg': 78,
        'cookie': 148,
        'french fries': 312,
        'burrito': 379,
    }
    
    # Map ImageNet class indices to food categories
    # This is a simplified mapping for demonstration
    imagenet_to_food = {
        948: 'fruit',  # banana
        950: 'orange',
        953: 'apple',
        927: 'pizza',
        933: 'hamburger',
        934: 'hotdog',
        925: 'sandwich',
        963: 'donut',
        924: 'cake',
        928: 'ice cream',
        937: 'broccoli',
        940: 'carrot',
        951: 'tomato',
        959: 'pasta',
        926: 'rice',
    }
    
    return food_calories, imagenet_to_food

food_calories, imagenet_to_food = load_class_mapping()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image_data):
    # Convert base64 to image
    if isinstance(image_data, str) and image_data.startswith('data:image'):
        # Remove the header part and decode
        image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))
    else:
        # For uploaded files
        image = Image.open(image_data)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        class_id = predicted.item()
    
    # Get class name and probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, 5)
    
    results = []
    total_calories = 0
    detected_foods = []
    
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        class_id = idx.item()
        probability = prob.item() * 100
        
        # Check if the class is in our food mapping
        if class_id in imagenet_to_food:
            food_name = imagenet_to_food[class_id]
            calorie_value = food_calories.get(food_name, 0)
            
            # Only consider if probability is above 15%
            if probability > 15:
                detected_foods.append({
                    'name': food_name,
                    'probability': probability,
                    'calories': calorie_value
                })
                total_calories += calorie_value
    
    return {
        'detected_foods': detected_foods,
        'total_calories': total_calories
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        try:
            result = process_image(file)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    
    elif 'image_data' in request.form:
        try:
            image_data = request.form['image_data']
            result = process_image(image_data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'No image data received'})

if __name__ == '__main__':
    app.run(debug=True)
