import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score

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

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

def plot_and_evaluate():
    # 1. Load data and models
    X_test_flat, X_test_cnn, y_test, ann_model, cnn_model = load_data_and_models()
    y_test_classes = np.argmax(y_test, axis=1)

    # 2. Get predictions
    print("Generating predictions...")
    ann_pred = np.argmax(ann_model.predict(X_test_flat), axis=1)
    cnn_pred = np.argmax(cnn_model.predict(X_test_cnn), axis=1)

    # 3. Generate confusion matrices
    ann_cm = confusion_matrix(y_test_classes, ann_pred)
    cnn_cm = confusion_matrix(y_test_classes, cnn_pred)

    # 4. Plot side-by-side heatmaps
    print("Plotting confusion matrices...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(ann_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax1.set_title('ANN Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax2.set_title('CNN Confusion Matrix')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices.png'))
    plt.close()

    # 5. Print classification reports
    print("\n" + "="*30)
    print("ANN CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_test_classes, ann_pred, target_names=CLASS_NAMES))

    print("\n" + "="*30)
    print("CNN CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_test_classes, cnn_pred, target_names=CLASS_NAMES))

    # 6. Save metrics to JSON
    metrics_dict = {
        "ann": get_metrics(y_test_classes, ann_pred),
        "cnn": get_metrics(y_test_classes, cnn_pred)
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, 'eval_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"\nEvaluation metrics saved to {METRICS_DIR}")
    print(f"Confusion matrices saved to {PLOTS_DIR}")

if __name__ == "__main__":
    plot_and_evaluate()
