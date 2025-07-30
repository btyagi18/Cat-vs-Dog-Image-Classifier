import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

MODEL_PATH = 'saved_model/model.h5'

IMG_WIDTH = 100
IMG_HEIGHT = 100
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT) 

CLASS_LABELS = {0: 'Cat', 1: 'Dog'} 

def load_trained_model(model_path):
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please ensure the model.h5 file is in the correct directory (e.g., 'saved_model/model.h5').")
        return None
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from '{model_path}'")
        print("Model Summary (for verification of input shape):")
        model.summary()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(img_path, target_size=TARGET_SIZE):

    if not os.path.exists(img_path):
        print(f"Error: Image file not found at '{img_path}'")
        return None

    try:
        print(f"Loading image '{os.path.basename(img_path)}' with target_size={target_size}...")
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        print(f"Image '{os.path.basename(img_path)}' preprocessed successfully. Shape: {img_array.shape}")

        return img_array
    
    except Exception as e:
        print(f"Error preprocessing image '{img_path}': {e}")
        return None

# Main prediction function
def predict_image(model, img_path):
    """
    Predicts the class of a single image using the loaded model.
    """
    processed_img = preprocess_image(img_path)
    if processed_img is None:
        return "Prediction failed: Image preprocessing error."

    try:
        prediction_probability = model.predict(processed_img)[0][0]

        if prediction_probability >= 0.5:
            predicted_class_index = 1 # Dog
        else:
            predicted_class_index = 0 # Cat

        predicted_label = CLASS_LABELS.get(predicted_class_index, "Unknown")

        print(f"\n--- Prediction Results for '{os.path.basename(img_path)}' ---")
        print(f"Raw Prediction Probability (Dog likelihood): {prediction_probability:.4f}")
        print(f"Predicted Class: {predicted_label}")
        print(f"Confidence: {prediction_probability*100:.2f}% (if Dog) or {(1-prediction_probability)*100:.2f}% (if Cat)")
        print("-------------------------------------------------")

        return predicted_label, prediction_probability

    except Exception as e:
        print(f"Error during prediction: {e}")
        
        if processed_img is not None:
            print(f"Input shape during error: {processed_img.shape}")
        return "Prediction failed: Model inference error."

if __name__ == "__main__":
    
    model = load_trained_model(MODEL_PATH)

    if model is not None:
        print("\nModel loaded successfully. Ready for predictions!")

        while True:
        
            image_to_predict_path = input("\nEnter the path to the image (e.g., dataset/PetImages/Cat/1.jpg) or 'q' to quit: ")

            if image_to_predict_path.lower() == 'q':
                print("Exiting prediction script.")
                break

            predict_image(model, image_to_predict_path)

    else:
        print("\nModel not loaded. Cannot proceed with predictions.")