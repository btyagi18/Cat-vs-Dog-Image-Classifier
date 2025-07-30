from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_model(model, X_test, y_test):
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["Cat", "Dog"])
    print(report)

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()
