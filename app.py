from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

app = Flask(__name__)

# Load your trained model
model = load_model('breast_cancer_model_xray.h5')

# Class names from your training
CLASS_NAMES = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
TARGET_SIZE = (224, 224)  # From your constants

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file temporarily
        img_path = 'temp_img.' + file.filename.rsplit('.', 1)[1].lower()
        file.save(img_path)
        
        try:
            # Preprocess EXACTLY like during training
            img = image.load_img(img_path, target_size=TARGET_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)  # MobileNetV2 specific preprocessing
            
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            result = {
                'prediction': int(predicted_class),
                'class_name': CLASS_NAMES[predicted_class],
                'confidence': float(confidence),
                'all_predictions': {
                    'Benign': float(predictions[0][0]),
                    'Malignant': float(predictions[0][1]),
                    'Normal': float(predictions[0][2])
                }
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)