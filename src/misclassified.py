import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Constants
PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR = 'models'
PLOTS_DIR = os.path.join('outputs', 'plots')
METRICS_DIR = os.path.join('outputs', 'metrics')
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_data_and_models():
    print("Loading test data...")
    X_test_flat = np.load(os.path.join(PROCESSED_DIR, 'X_test_flat.npy'))
    X_test_cnn = np.load(os.path.join(PROCESSED_DIR, 'X_test_cnn.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

    print("Loading models...")
    ann_model = load_model(os.path.join(MODELS_DIR, 'ann_model.h5'))
    cnn_model = load_model(os.path.join(MODELS_DIR, 'cnn_multiscale.h5'))
    
    return X_test_flat, X_test_cnn, y_test, ann_model, cnn_model

def save_misclassified_results(model_name, x_data, x_images, y_true_classes, model):
    print(f"Finding misclassified samples for {model_name}...")
    
    # Get predictions
    y_pred = model.predict(x_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Find indices where prediction != true
    misclassified_idx = np.where(y_pred_classes != y_true_classes)[0]
    
    # 1. Plot first 10 misclassified samples
    plt.figure(figsize=(15, 6))
    for i in range(10):
        idx = misclassified_idx[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_images[idx])
        plt.axis('off')
        
        true_label = CLASS_NAMES[y_true_classes[idx]]
        pred_label = CLASS_NAMES[y_pred_classes[idx]]
        
        plt.title(f"True: {true_label}\nPred: {pred_label}", color='red', fontsize=10)
    
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_misclassified.png'))
    plt.close()
    
    # 2. Save indices and labels to JSON
    results = []
    for idx in misclassified_idx[:100]: # Save top 100 for reference
        results.append({
            "index": int(idx),
            "true_class": CLASS_NAMES[y_true_classes[idx]],
            "predicted_class": CLASS_NAMES[y_pred_classes[idx]]
        })
    
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, f'{model_name.lower()}_misclassified.json'), 'w') as f:
        json.dump(results, f, indent=4)

def run_analysis():
    # Load data and models
    X_test_flat, X_test_cnn, y_test, ann_model, cnn_model = load_data_and_models()
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Use CNN test set for image display (unflattened)
    X_test_images = X_test_cnn

    # Process ANN
    save_misclassified_results("ANN", X_test_flat, X_test_images, y_test_classes, ann_model)
    
    # Process CNN
    save_misclassified_results("CNN", X_test_cnn, X_test_images, y_test_classes, cnn_model)

    print(f"\nMisclassified plots saved to {PLOTS_DIR}")
    print(f"Misclassified metadata saved to {METRICS_DIR}")

if __name__ == "__main__":
    run_analysis()
