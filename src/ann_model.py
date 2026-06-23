import os
import time
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Constants
PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR = 'models'
METRICS_DIR = os.path.join('outputs', 'metrics')

def load_data():
    print("Loading processed data...")
    X_train_flat = np.load(os.path.join(PROCESSED_DIR, 'X_train_flat.npy'))
    X_test_flat = np.load(os.path.join(PROCESSED_DIR, 'X_test_flat.npy'))
    y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
    return X_train_flat, X_test_flat, y_train, y_test

def build_ann(neurons=512, dropout=0.3, learning_rate=0.001):
    model = Sequential([
        Input(shape=(3072,)),
        Dense(neurons, activation='relu'),
        BatchNormalization(),
        Dropout(dropout),
        Dense(neurons//2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout),
        Dense(neurons//4, activation='relu'),
        BatchNormalization(),
        Dropout(dropout),
        Dense(10, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_and_evaluate():
    # 1. Load Data
    X_train_flat, X_test_flat, y_train, y_test = load_data()

    # 2. Build Model
    model = build_ann()
    model.summary()

    # 3. Setup Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    # 4. Train Model
    print("Starting ANN training...")
    start_time = time.time()
    history = model.fit(
        X_train_flat, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # 5. Evaluate Model
    loss, accuracy = model.evaluate(X_test_flat, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    y_pred = model.predict(X_test_flat)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes))

    # 6. Save Results
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Save model
    model.save(os.path.join(MODELS_DIR, 'ann_model.h5'))
    
    # Save history
    history_dict = history.history
    with open(os.path.join(METRICS_DIR, 'ann_history.json'), 'w') as f:
        json.dump(history_dict, f)

    # Save metadata
    meta_data = {
        "accuracy": float(accuracy),
        "loss": float(loss),
        "params": int(model.count_params()),
        "training_time": float(training_time)
    }
    with open(os.path.join(METRICS_DIR, 'ann_meta.json'), 'w') as f:
        json.dump(meta_data, f)

    print("Model and metrics saved successfully.")

if __name__ == "__main__":
    train_and_evaluate()
