import os
import json
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI(title="CIFAR-10 Prediction API")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
MODELS_DIR = os.path.join(os.getcwd(), 'models')
METRICS_DIR = os.path.join(os.getcwd(), 'outputs', 'metrics')

# Global model placeholders
models = {}

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    ann_path = os.path.join(MODELS_DIR, 'ann_model.h5')
    cnn_path = os.path.join(MODELS_DIR, 'cnn_multiscale.h5')
    
    if os.path.exists(ann_path):
        models['ann'] = load_model(ann_path)
        print("ANN model loaded successfully.")
    else:
        print(f"Warning: ANN model not found at {ann_path}")
        
    if os.path.exists(cnn_path):
        models['cnn'] = load_model(cnn_path)
        print("CNN model loaded successfully.")
    else:
        print(f"Warning: CNN model not found at {cnn_path}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "models": [name for name in models.keys()]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of an uploaded image using both ANN and CNN models"""
    if not models.get('ann') or not models.get('cnn'):
        raise HTTPException(status_code=503, detail="Models not loaded. Please run training scripts first.")

    try:
        # Read and process image
        request_object_content = await file.read()
        image = Image.open(io.BytesIO(request_object_content)).convert('RGB')
        image = image.resize((32, 32))
        img_array = np.array(image).astype('float32') / 255.0
        
        # Prepare inputs
        ann_input = img_array.flatten().reshape(1, 3072)
        cnn_input = img_array.reshape(1, 32, 32, 3)
        
        # Run ANN prediction
        ann_probs = models['ann'].predict(ann_input)[0]
        ann_class_idx = np.argmax(ann_probs)
        ann_conf = float(ann_probs[ann_class_idx])
        
        # Run CNN prediction
        cnn_probs = models['cnn'].predict(cnn_input)[0]
        cnn_class_idx = np.argmax(cnn_probs)
        cnn_conf = float(cnn_probs[cnn_class_idx])
        
        # Determine winner
        winner = "cnn" if cnn_conf >= ann_conf else "ann"
        
        return {
            "ann": {
                "class": CLASS_NAMES[ann_class_idx],
                "confidence": ann_conf,
                "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(ann_probs)}
            },
            "cnn": {
                "class": CLASS_NAMES[cnn_class_idx],
                "confidence": cnn_conf,
                "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(cnn_probs)}
            },
            "winner": winner
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/comparison")
async def get_comparison():
    """Get final model comparison metrics"""
    path = os.path.join(METRICS_DIR, 'final_comparison.json')
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Comparison metrics not found. Run final_comparison.py first.")
    
    with open(path, 'r') as f:
        return json.load(f)

@app.get("/misclassified/{model_name}")
async def get_misclassified(model_name: str):
    """Get misclassified samples for a specific model"""
    if model_name.lower() not in ["ann", "cnn"]:
        raise HTTPException(status_code=400, detail="Invalid model name. Use 'ann' or 'cnn'.")
        
    path = os.path.join(METRICS_DIR, f'{model_name.lower()}_misclassified.json')
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Misclassified metrics for {model_name} not found.")
    
    with open(path, 'r') as f:
        return json.load(f)
