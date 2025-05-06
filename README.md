# FoodLens - AI Food Recognition & Calorie Estimation App

FoodLens is a web application that uses AI to recognize food items from uploaded or captured photos and provides calorie estimations for the detected items.

## Features

- Upload food images from your device
- Capture photos directly using your camera
- AI-powered food recognition using ResNet50
- Calorie estimation for detected food items
- Responsive design for mobile and desktop use

## Prerequisites

- Python 3.7+
- Flask
- PyTorch
- Pillow (PIL)
- A modern web browser with camera access (for photo capture feature)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/foodlens.git
   cd foodlens
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install flask torch torchvision Pillow
   ```

## Project Structure

```
foodlens/
├── app.py                  # Flask application
├── static/
│   ├── css/
│   │   └── style.css       # Styling for the web application
│   ├── js/
│   │   └── script.js       # Client-side JavaScript functionality
│   └── uploads/            # Directory for temporary image storage
├── templates/
│   └── index.html          # Main HTML template
└── README.md               # This file

## How It Works

### Backend

1. **Image Processing**: The application uses PyTorch with a pre-trained ResNet50 model to recognize food items in images.
2. **Classification**: The model identifies potential food items and their probabilities.
3. **Calorie Estimation**: For each detected food item, the app estimates calories based on a predefined database.

### Frontend

1. **User Interface**: A clean, responsive interface built with HTML, CSS, and JavaScript.
2. **Image Capture**: Users can upload images or use their device camera to capture food photos.
3. **Results Display**: Detected foods are displayed with their probability scores and calorie estimates.

## Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Upload a food image or use the camera to capture one, then click "Analyze Food".

## Limitations

- The food recognition is based on ImageNet classes, which has a limited set of food categories.
- Calorie estimates are approximations based on standard serving sizes.
- The app may struggle with mixed dishes or foods that aren't clearly visible.

## Extending the Application

To improve the application, consider:

1. **Training a custom food recognition model** with a more comprehensive food dataset.
2. **Expanding the food-calorie database** with more accurate and detailed information.
3. **Implementing portion size estimation** to provide more accurate calorie counts.
4. **Adding user accounts** to track food intake over time.
5. **Integrating nutritional information** beyond just calories (protein, carbs, fats, etc.).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
