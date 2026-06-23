# CIFAR-10 CNN vs ANN: Deep Learning Performance Comparison 🚀

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat&logo=tensorflow)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-61DAFB?style=flat&logo=react)](https://reactjs.org/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.3-38B2AC?style=flat&logo=tailwind-css)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive full-stack project comparing the efficiency, accuracy, and spatial awareness of **Artificial Neural Networks (ANN)** versus **Convolutional Neural Networks (CNN)** using the CIFAR-10 dataset.

## 🏗️ Model Architectures

### **Artificial Neural Network (ANN)**
A classic multi-layer perceptron (MLP) architecture focused on dense connectivity.
- **Structure**: `Input(3072)` → `Dense(512)` → `Dense(256)` → `Dense(128)` → `Output(10)`
- **Key Features**: Batch Normalization, Dropout (0.3), ReLU activation.

### **Convolutional Neural Network (CNN)**
A multiscale architecture designed to capture features at different spatial resolutions.
- **Structure**: `Conv(3×3)` → `Conv(5×5)` → `Conv(7×7)` → `Flatten` → `Dense(256)` → `Output(10)`
- **Key Features**: MaxPooling, Multiscale Kernels, Spatial Feature Extraction.

## 📊 Results Summary

| Model | Params | Accuracy | Training Time |
| :--- | :--- | :--- | :--- |
| **ANN** | ~1.7M | ... | ... |
| **CNN** | ~0.8M | ... | ... |

*Note: Results vary based on local training iterations. Run the evaluation scripts to update.*

## 🛠️ Tech Stack
- **Deep Learning**: TensorFlow, Keras
- **Backend**: FastAPI, Uvicorn, Python
- **Frontend**: React.js, Vite, Tailwind CSS, Recharts
- **DevOps**: Docker, Git

## 🚀 Getting Started

### **1. Prerequisites**
- Python 3.10+
- Node.js & npm

### **2. Installation**
```bash
# Clone the repository
git clone https://github.com/neeraj214/CIFAR-10-Image-Classification-using-Convolutional-Neural-Network-and-ANN.git
cd CIFAR-10-Image-Classification-using-Convolutional-Neural-Network-and-ANN

# Install Python dependencies
pip install -r requirements.txt
```

### **3. Execution Order**
Follow these steps to train and evaluate:
1. **Data Exploration**: `python src/data_exploration.py`
2. **Preprocessing**: `python src/preprocess.py`
3. **Train ANN**: `python src/ann_model.py`
4. **Train CNN**: `python src/cnn_multiscale.py`
5. **Final Metrics**: `python src/final_comparison.py`

### **4. Running the App**
- **Backend**: 
  ```bash
  cd backend
  uvicorn main:app --reload
  ```
- **Frontend**:
  ```bash
  cd frontend
  npm install
  npm run dev
  ```

## 🌐 Deployment
- **Backend**: Hosted on [Render](https://render.com/) via `backend/Dockerfile`.
- **Frontend**: Hosted on [Vercel](https://vercel.com/).

🔗 **Live Demo**: [Coming Soon](https://github.com/neeraj214/CIFAR-10-Image-Classification-using-Convolutional-Neural-Network-and-ANN)

## 👤 Author
**neeraj214**  
[GitHub](https://github.com/neeraj214) | [LinkedIn](https://www.linkedin.com/in/neeraj214/)

---
*Developed for deep learning performance analysis and educational purposes.*
