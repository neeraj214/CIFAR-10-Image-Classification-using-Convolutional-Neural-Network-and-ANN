import os
import json
import matplotlib.pyplot as plt

# Constants
METRICS_DIR = os.path.join('outputs', 'metrics')
PLOTS_DIR = os.path.join('outputs', 'plots')

def load_history(filename):
    with open(os.path.join(METRICS_DIR, filename), 'r') as f:
        return json.load(f)

def plot_training_curves():
    # 1. Load histories
    print("Loading training histories...")
    ann_history = load_history('ann_history.json')
    cnn_history = load_history('cnn_history.json')

    # 2. Plot 2x2 subplot figure
    print("Generating training curves...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # [0,0] ANN Accuracy
    axes[0, 0].plot(ann_history['accuracy'], label='Train')
    axes[0, 0].plot(ann_history['val_accuracy'], label='Val')
    axes[0, 0].set_title('ANN: Training vs Validation Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # [0,1] CNN Accuracy
    axes[0, 1].plot(cnn_history['accuracy'], label='Train')
    axes[0, 1].plot(cnn_history['val_accuracy'], label='Val')
    axes[0, 1].set_title('CNN: Training vs Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # [1,0] ANN Loss
    axes[1, 0].plot(ann_history['loss'], label='Train')
    axes[1, 0].plot(ann_history['val_loss'], label='Val')
    axes[1, 0].set_title('ANN: Training vs Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # [1,1] CNN Loss
    axes[1, 1].plot(cnn_history['loss'], label='Train')
    axes[1, 1].plot(cnn_history['val_loss'], label='Val')
    axes[1, 1].set_title('CNN: Training vs Validation Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'))
    plt.close()

    # 3. Plot ANN vs CNN val_accuracy on same graph
    print("Generating validation accuracy comparison...")
    plt.figure(figsize=(10, 6))
    plt.plot(ann_history['val_accuracy'], label='ANN Val Accuracy', color='blue')
    plt.plot(cnn_history['val_accuracy'], label='CNN Val Accuracy', color='green')
    plt.title('Validation Accuracy: ANN vs CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'val_accuracy_compare.png'))
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    plot_training_curves()
