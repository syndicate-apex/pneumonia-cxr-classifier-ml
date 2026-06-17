# Pneumonia Chest X-Ray Classifier

A deep learning-based web application for pneumonia detection from chest X-ray images using EfficientNet-B0 and Grad-CAM visualization.

## Overview

This project classifies chest X-ray images as either pneumonia-positive or normal using a fine-tuned EfficientNet-B0 model. The application provides real-time predictions with confidence scores and visual explanations using Grad-CAM.

## Features

- Binary classification (Pneumonia / Normal)
- Confidence scores
- Grad-CAM visual explanations
- Streamlit web interface
- Dockerized deployment
- Model training and evaluation notebooks

## Dataset

- **Source**: Chest X-Ray Images (Pneumonia)
- **Structure**: Training, validation, and test splits with NORMAL and PNEUMONIA subdirectories

## Tech Stack

- Python
- PyTorch
- EfficientNet-B0
- Streamlit
- Grad-CAM
- Docker

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Or use conda:
   ```
   conda env create -f environment.yml
   conda activate pneumonia-env
   ```

## Usage

### Run Locally (Streamlit)
```bash
streamlit run gui/app.py
```

### Run with Docker
```bash
docker build -t pneumonia-app .
docker run -p 8501:8501 pneumonia-app
```

## Project Structure

```
pneumonia-cxr-classifier-ml/
├── dockerfile                      # Docker configuration
├── environment.yml                 # Conda environment file
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── chest_xray/                     # Dataset directory
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── model/
│   └── pneumonia_classifier.pth    # Trained model weights
├── gui/
│   └── app.py                      # Streamlit web application
├── notebooks/                      # Jupyter notebooks
│   ├── data_balancing.ipynb
│   ├── E1_model_train_Adam.ipynb
│   ├── E2_model_train_AdamW.ipynb
│   ├── E3_model_train_SGD.ipynb
│   └── E4_model_train_ResNet50.ipynb
└── src/                            # Source code and utilities
```

## Model Information

- **Architecture**: EfficientNet-B0
- **Input**: Chest X-ray images (224x224)
- **Output**: Pneumonia probability + Grad-CAM visualization
- **Framework**: PyTorch