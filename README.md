# CIFAR-10 CNN vs ANN Deep Comparison 🚀

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat&logo=keras)](https://keras.io/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

This repository contains a comprehensive deep comparison between **Convolutional Neural Networks (CNN)** and **Artificial Neural Networks (ANN)** using the **CIFAR-10** image classification dataset.

## 📌 Project Overview
The goal of this project is to evaluate and compare the performance, accuracy, and efficiency of two different neural network architectures on the same image classification task. The comparison includes training curves, confusion matrices, and analysis of misclassified samples.

## 🏗️ Folder Structure
Refer to [project_structure.md](project_structure.md) for the complete directory layout.

- `src/`: Core Python scripts for data exploration, preprocessing, and training.
- `data/`: Local storage for raw and processed datasets (ignored by git).
- `models/`: Saved model architectures and weights.
- `outputs/`: Visualization plots and performance metrics.

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/neeraj214/CIFAR-10-Image-Classification-using-Convolutional-Neural-Network-and-ANN.git
   cd CIFAR-10-Image-Classification-using-Convolutional-Neural-Network-and-ANN
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Workflow
Follow these steps in order to reproduce the results:

1. **Data Exploration:**
   ```bash
   python src/data_exploration.py
   ```
   *Analyzes dataset distribution and generates sample grids.*

2. **Preprocessing:**
   ```bash
   python src/preprocess.py
   ```
   *Normalizes, encodes, and flattens data for model inputs.*

3. **Train ANN Model:**
   ```bash
   python src/ann_model.py
   ```
   *Trains a 3-hidden layer Artificial Neural Network.*

4. **Train CNN Model:**
   ```bash
   python src/cnn_multiscale.py
   ```
   *Trains a Multiscale Convolutional Neural Network (3x3, 5x5, 7x7 kernels).*

5. **Generate Evaluation Plots:**
   ```bash
   python src/plot_curves.py
   python src/plot_confusion.py
   ```
   *Generates training curves, validation comparisons, and confusion matrices.*

6. **Error Analysis:**
   ```bash
   python src/misclassified.py
   ```
   *Visualizes samples that the models failed to classify correctly.*

## 📊 Technologies Used
- **Deep Learning:** TensorFlow, Keras
- **Data Analysis:** NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---
Developed with ❤️ by [Neeraj](https://github.com/neeraj214)
