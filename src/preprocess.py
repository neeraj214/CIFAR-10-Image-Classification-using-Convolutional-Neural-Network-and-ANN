import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Constants
PROCESSED_DIR = os.path.join('data', 'processed')

def preprocess_data():
    # 1. Load CIFAR-10 via keras
    print("Loading CIFAR-10 data...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # 2. Normalize: divide by 255.0 → float32 in [0,1]
    print("Normalizing pixel values...")
    X_train_cnn = X_train.astype('float32') / 255.0
    X_test_cnn = X_test.astype('float32') / 255.0

    # 3. One-hot encode labels: to_categorical(y, num_classes=10)
    print("One-hot encoding labels...")
    y_train_cat = to_categorical(y_train, num_classes=10)
    y_test_cat = to_categorical(y_test, num_classes=10)

    # 4. For ANN: flatten images → shape (N, 3072)
    print("Flattening images for ANN...")
    X_train_flat = X_train_cnn.reshape(X_train_cnn.shape[0], -1)
    X_test_flat = X_test_cnn.reshape(X_test_cnn.shape[0], -1)

    # Ensure output directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 5. Save to data/processed/
    print(f"Saving processed data to {PROCESSED_DIR}...")
    np.save(os.path.join(PROCESSED_DIR, 'X_train_flat.npy'), X_train_flat)
    np.save(os.path.join(PROCESSED_DIR, 'X_test_flat.npy'), X_test_flat)
    np.save(os.path.join(PROCESSED_DIR, 'X_train_cnn.npy'), X_train_cnn)
    np.save(os.path.join(PROCESSED_DIR, 'X_test_cnn.npy'), X_test_cnn)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train_cat)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test_cat)

    # 6. Print all saved shapes as confirmation
    print("\nPre-processing complete. Saved shapes:")
    print(f"  X_train_flat: {X_train_flat.shape}")
    print(f"  X_test_flat:  {X_test_flat.shape}")
    print(f"  X_train_cnn:  {X_train_cnn.shape}")
    print(f"  X_test_cnn:   {X_test_cnn.shape}")
    print(f"  y_train:      {y_train_cat.shape}")
    print(f"  y_test:       {y_test_cat.shape}")

if __name__ == "__main__":
    preprocess_data()
