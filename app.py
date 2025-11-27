from flask import Flask, render_template, request, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load your trained VGG19 model (replace with correct model path)
model = load_model('deepfake_detection4_vgg19_with_plots.h5') 
# Define the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for VGG19
    return img_array

# Function to make prediction
def make_prediction(img_path):
    img_array = preprocess_image(img_path)
    logging.debug(f"Image preprocessed: {img_array.shape}")

    # Make prediction
    predictions = model.predict(img_array)  # Make prediction
    logging.debug(f"Prediction values: {predictions}")

    # Check if the prediction is above or below threshold (0.5)
    if predictions[0] > 0.5:
        return "Real"
    else:
        return "Fake"
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file and allowed_file(file.filename):
        # Ensure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])  # Create the directory if it doesn't exist

        filename = file.filename
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        # Make a prediction
        prediction = make_prediction(img_path)
        return jsonify({'prediction': prediction})  # Return prediction as JSON

    else:
        return jsonify({'error': 'Invalid file type or no file provided'}), 400

@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')  # Render the thank you page

@app.route('/exit')
def exit_app():
    # Redirect to the thankyou page after submitting
    return redirect(url_for('thankyou'))

if __name__ == '__main__':
    app.run(debug=True)
