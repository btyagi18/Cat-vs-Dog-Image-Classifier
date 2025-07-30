from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os
import io

app = Flask(__name__)

MODEL_PATH = 'saved_model/model.h5'
IMG_WIDTH = 100
IMG_HEIGHT = 100
CLASS_LABELS = {0: 'Cat', 1: 'Dog'}

classifier_model = None
try:
    classifier_model = load_model(MODEL_PATH)
    classifier_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("✨ AI Model loaded successfully for web predictions!")
except Exception as e:
    print(f"🚨 ERROR: Failed to load the AI model from '{MODEL_PATH}'.")
    print(f"Please ensure '{MODEL_PATH}' exists and is not corrupted. Details: {e}")

def preprocess_image_for_web(img_data):
    """
    Takes raw image data, loads it, resizes it, and normalizes pixel values.
    """
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
    """Renders the main HTML page for our classifier."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image_route():
    """
    Receives an uploaded image, preprocesses it, and uses the AI model
    to predict if it's a cat or a dog. Returns the prediction as JSON.
    """
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
            raw_prediction = classifier_model.predict(processed_img)[0][0]

            if raw_prediction >= 0.5:
                predicted_label = CLASS_LABELS[1]
                confidence = raw_prediction
            else:
                predicted_label = CLASS_LABELS[0]
                confidence = 1 - raw_prediction

            return jsonify({
                'label': predicted_label,
                'confidence': f"{confidence*100:.2f}%"
            })

        except Exception as e:
            print(f"Error during AI model prediction: {e}")
            return jsonify({'error': 'An internal error occurred during prediction.'}), 500

    return jsonify({'error': 'An unexpected issue occurred with your upload.'}), 500

# --- Run the Flask Application ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)