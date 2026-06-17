# Pneumonia Chest X-Ray Classifier

A deep learning-based web application for pneumonia detection from chest X-ray images using EfficientNet-B0 and Grad-CAM.

## Features

- Pneumonia / Normal classification
- Confidence score
- Grad-CAM visualization
- Streamlit web interface
- Dockerized deployment

## Tech Stack

- Python
- PyTorch
- EfficientNet-B0
- Streamlit
- Grad-CAM
- Docker

## Run Locally

docker build -t pneumonia-app .
docker run -p 8501:8501 pneumonia-app