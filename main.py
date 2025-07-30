from src.data_preprocessing import load_data, preprocess_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_history
from utils.helpers import show_sample_images

def main():
    print("[INFO] Loading and preprocessing data...")
    data = load_data("dataset/PetImages")
    X_train, X_test, y_train, y_test = preprocess_data(data)

    print("[INFO] Showing sample images...")
    show_sample_images(X_train, y_train, ['Cat', 'Dog'])

    print("[INFO] Building CNN model...")
    model = build_model(input_shape=X_train.shape[1:])

    print("[INFO] Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test)

    print("[INFO] Visualizing training history...")
    plot_history(history)

    print("[INFO] Evaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
