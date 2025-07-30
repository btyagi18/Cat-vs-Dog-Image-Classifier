from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
import os
import io

app = Flask(__name__)

MODEL_PATH = 'saved_model/model.h5'
IMG_WIDTH = 100
IMG_HEIGHT = 100
CLASS_LABELS = {0: 'Cat', 1: 'Dog'}

CONFIDENCE_THRESHOLD = 0.70

classifier_model = None
try:
    classifier_model = load_model(MODEL_PATH)

    print("AI Model loaded successfully for web predictions!")
except Exception as e:
    print(f"ERROR: Failed to load the AI model from '{MODEL_PATH}'.")
    print(f"Please ensure '{MODEL_PATH}' exists and is not corrupted. Details: {e}")

def preprocess_image_for_web(img_data):
    try:
        img = image.load_img(io.BytesIO(img_data), target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Preprocessing error for uploaded image: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image_route():

    if classifier_model is None:
        return jsonify({'error': 'AI model is not ready. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No image file was found in the upload.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No image was selected.'}), 400

    if file:
        img_data = file.read()
        processed_img = preprocess_image_for_web(img_data)

        if processed_img is None:
            return jsonify({'error': 'Failed to process image. Make sure it\'s a standard image format (JPG, PNG).'}), 400

        try:
        
            prob_dog = classifier_model.predict(processed_img)[0][0]
            prob_cat = 1 - prob_dog

            final_label = ""
            final_confidence_display = ""

            # Determine the predicted class based on the highest probability
            if prob_dog >= prob_cat:
                predicted_class = 'Dog'
                confidence_score = prob_dog
            else:
                predicted_class = 'Cat'
                confidence_score = prob_cat

            print(f"DEBUG: Image uploaded: {file.filename}")
            print(f"DEBUG: Raw Probabilities: Cat={prob_cat*100:.2f}%, Dog={prob_dog*100:.2f}%")
            print(f"DEBUG: Model's Best Guess: {predicted_class}, Confidence: {confidence_score*100:.2f}%")
            print(f"DEBUG: Current CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD*100:.2f}%")
    
            if confidence_score < CONFIDENCE_THRESHOLD:
                final_label = "Neither Cat nor Dog"
                final_confidence_display = f"Low Confidence ({confidence_score*100:.2f}%)"
            else:
                final_label = predicted_class
                final_confidence_display = f"{confidence_score*100:.2f}%"

            return jsonify({
                'label': final_label,
                'confidence': final_confidence_display
            })

        except Exception as e:
            print(f"Error during AI model prediction: {e}")
            return jsonify({'error': 'An internal error occurred during prediction.'}), 500

    return jsonify({'error': 'An unexpected issue occurred with your upload.'}), 500

# Run the Flask Application
if __name__ == '__main__':
    app.run(debug=True, port=5000)