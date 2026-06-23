import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10

# Constants
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
OUTPUT_DIR = os.path.join('outputs', 'plots')

def explore_data():
    # 1. Load CIFAR-10
    print("Loading CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 2. Print shapes
    print(f"Train shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape}")

    # 3. Print pixel range before normalization
    print(f"Pixel range: min={x_train.min()}, max={x_train.max()}")

    # 4. Print class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    distribution = dict(zip(unique, counts))
    print("Class distribution (Train):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {distribution[i]}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. Plot 5x10 sample grid (5 samples per class, 10 classes)
    print("Generating sample grid...")
    plt.figure(figsize=(15, 8))
    for i in range(10):
        # Get indices of the first 5 samples for each class
        indices = np.where(y_train == i)[0][:5]
        for j, idx in enumerate(indices):
            plt.subplot(5, 10, j * 10 + i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x_train[idx])
            if j == 0:
                plt.title(CLASS_NAMES[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_grid.png'))
    plt.close()

    # 6. Plot class distribution bar chart
    print("Generating class distribution plot...")
    plt.figure(figsize=(12, 6))
    sns.barplot(x=CLASS_NAMES, y=counts)
    plt.title('CIFAR-10 Class Distribution (Training Set)')
    plt.xlabel('Class Name')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))
    plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    explore_data()
