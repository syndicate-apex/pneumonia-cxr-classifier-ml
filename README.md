# 🫁 Pneumonia Detection System - Chest X-ray Classification

A deep learning-based application for automated pneumonia detection from chest X-ray images. This project uses state-of-the-art convolutional neural networks (CNNs) to classify chest X-rays as either Normal or Pneumonia, with an interactive web interface for easy predictions.

**Project Status:** Bio Nova Hackathon Entry  
**Last Updated:** June 15, 2026

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Model Training](#model-training)
- [GUI Usage](#gui-usage)
- [Technical Stack](#technical-stack)
- [Models Included](#models-included)
- [Results](#results)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [Medical Disclaimer](#medical-disclaimer)
- [License](#license)

---

## 🎯 Overview

Pneumonia is a serious respiratory infection affecting millions worldwide. This project implements a machine learning pipeline to assist in early detection of pneumonia from chest X-ray images. The system features:

- **Multiple trained models** with different architectures and optimizers
- **Interactive Streamlit GUI** for easy image upload and prediction
- **Grad-CAM visualization** to understand model decision-making
- **Real-time predictions** with confidence scores
- **Prediction history tracking** in the sidebar

> ⚠️ **Disclaimer:** This system is intended for educational and research purposes only. It should not be used for clinical diagnosis without professional medical review.

---

## ✨ Features

### Core Features
- 🖼️ **Image Upload**: Support for JPG, JPEG, and PNG chest X-ray images
- 🤖 **AI Prediction**: Binary classification (Normal vs Pneumonia)
- 📊 **Confidence Scores**: Detailed probability distribution for each class
- 🔍 **Grad-CAM Visualization**: Visual heatmaps showing model attention areas
- 📈 **Class Probabilities**: Real-time probability bars for both classes
- 💾 **Prediction History**: Track recent predictions with timestamps
- ⚙️ **Adjustable Settings**: Confidence threshold control and visualization toggles

### GUI Features
- **Responsive Design**: Works on desktop and tablet browsers
- **Dark/Light Mode Compatible**: Streamlit theme support
- **User-Friendly Interface**: Intuitive layout with clear sections
- **Real-time Results**: Instant predictions upon image upload
- **Image Information**: Display filename, dimensions, and upload timestamp
- **Model Metadata**: View model architecture and specifications

---

## 📁 Project Structure

```
pneumonia-cxr-classifier-ml/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment file
│
├── chest_xray/                        # Dataset directory
│   ├── train/
│   │   ├── NORMAL/                   # Normal X-ray training images
│   │   └── PNEUMONIA/                # Pneumonia X-ray training images
│   ├── val/
│   │   ├── NORMAL/                   # Normal X-ray validation images
│   │   └── PNEUMONIA/                # Pneumonia X-ray validation images
│   └── test/
│       ├── NORMAL/                   # Normal X-ray test images
│       └── PNEUMONIA/                # Pneumonia X-ray test images
│
├── gui/                              # Streamlit GUI application
│   └── app.py                        # Main GUI application
│
├── model/                            # Pre-trained model weights
│   ├── pneumonia_classifier.pth      # EfficientNet-B0 weights
│   └── E4_pneumonia_classifier_ResNet50.pth  # ResNet50 weights
│
├── src/                              # Source code utilities (empty)
│
└── Notebooks/                        # Jupyter notebooks for training
    ├── data_balancing.ipynb          # Data analysis and balancing
    ├── E1_model_train_Adam.ipynb     # Training with Adam optimizer
    ├── E2_model_train_AdamW.ipynb    # Training with AdamW optimizer
    ├── E3_model_train_SGD.ipynb      # Training with SGD optimizer
    └── E4_model_train_ResNet50.ipynb # ResNet50 architecture training
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Conda or pip (recommended: Conda for better dependency management)
- 4GB+ RAM (8GB recommended for model inference)
- GPU (optional, but recommended for faster training)

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd pneumonia-cxr-classifier-ml

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate pneumonia-env
```

### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd pneumonia-cxr-classifier-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Data

### Dataset Information
- **Source**: ChexPert/NIH Chest X-ray Dataset or similar medical imaging dataset
- **Classes**: 2 (Normal, Pneumonia)
- **Total Samples**: Organized in train/val/test splits
- **Image Format**: JPEG, PNG

### Directory Structure
```
chest_xray/
├── train/          # Training set
├── val/            # Validation set  
└── test/           # Test set
```

### Data Preparation
The project includes:
- **data_balancing.ipynb**: Analyze class distribution and handle imbalanced data

---

## 🧠 Model Training

### Available Training Notebooks

#### 1. **E1_model_train_Adam.ipynb**
- Architecture: EfficientNet-B0
- Optimizer: Adam
- Learning Rate: Standard (typically 0.001)
- Batch Size: Configurable
- Use Case: Baseline with adaptive learning rates

#### 2. **E2_model_train_AdamW.ipynb**
- Architecture: EfficientNet-B0
- Optimizer: AdamW (Adam with weight decay)
- Learning Rate: Standard
- Use Case: Better generalization through weight decay regularization

#### 3. **E3_model_train_SGD.ipynb**
- Architecture: EfficientNet-B0
- Optimizer: SGD (Stochastic Gradient Descent)
- Momentum: With momentum for faster convergence
- Use Case: Classical optimizer, often excellent for CNNs

#### 4. **E4_model_train_ResNet50.ipynb**
- Architecture: ResNet50 (Residual Network)
- Optimizer: Varies (check notebook)
- Depth: 50 layers with skip connections
- Use Case: Deeper network for more complex feature learning

### Training Steps

1. **Open a Jupyter notebook** (e.g., `E1_model_train_Adam.ipynb`)
2. **Configure hyperparameters** as needed:
   - Epochs
   - Batch size
   - Learning rate
   - Data augmentation settings
3. **Run all cells** to train the model
4. **Monitor training** with loss curves and accuracy metrics
5. **Evaluate** on validation and test sets
6. **Save weights** automatically to `model/` directory

### Key Training Features
- Data augmentation (rotation, flip, brightness, contrast)
- Early stopping to prevent overfitting
- Learning rate scheduling
- Validation metrics tracking
- Model checkpointing

---

## 🎨 GUI Usage

### Starting the Application

```bash
# Navigate to gui directory
cd gui

# Run Streamlit app
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### How to Use

1. **Upload Image**
   - Click "Browse files" button
   - Select a chest X-ray image (JPG, JPEG, or PNG)
   - Image will be processed automatically

2. **View Results**
   - Prediction: Normal or Pneumonia (with color coding)
   - Confidence score in percentage
   - Probability distribution for both classes

3. **Analyze Visualization**
   - Original X-ray image
   - Grad-CAM heatmap showing regions of interest
   - Heatmap highlights areas that influenced the prediction

4. **Check Details**
   - Filename
   - Image dimensions
   - Upload timestamp
   - Model information in sidebar

5. **Track History**
   - Last 5 predictions shown in sidebar
   - Click "Clear History" to reset
   - Each entry shows prediction, confidence, and time

### Sidebar Controls

- **Confidence Threshold**: Adjust minimum confidence for predictions
- **Show Grad-CAM**: Toggle visualization on/off
- **Model Info**: View architecture details
- **Prediction History**: Track recent predictions

---

## 🔧 Technical Stack

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | Latest | Deep learning framework |
| **torchvision** | Latest | Computer vision utilities |
| **Streamlit** | Latest | Web GUI framework |
| **Pillow (PIL)** | Latest | Image processing |
| **NumPy** | Latest | Numerical computing |
| **OpenCV** | Latest | Image processing |
| **Matplotlib** | Latest | Visualization |
| **pytorch-grad-cam** | Latest | Model interpretability |

### Development Tools
- **Jupyter Notebook**: For interactive development
- **Python 3.8+**: Programming language
- **Conda**: Environment management
- **Git**: Version control

---

## 🤖 Models Included

### EfficientNet-B0
- **Architecture**: Compound scaling of efficient networks
- **Input Size**: 224 × 224 pixels
- **Parameters**: ~5.3M
- **Advantages**: 
  - Balanced accuracy and speed
  - Good performance on medical imaging
  - Efficient for deployment
- **Pre-trained Model**: `model/pneumonia_classifier.pth`

### ResNet50
- **Architecture**: Residual Network with 50 layers
- **Input Size**: 224 × 224 pixels
- **Parameters**: ~23.5M
- **Advantages**:
  - Skip connections prevent vanishing gradients
  - Excellent for deeper learning
  - Well-studied architecture
- **Pre-trained Model**: `model/E4_pneumonia_classifier_ResNet50.pth`

---

## 📈 Results

### Model Performance
The models achieve competitive performance on the test set (624 test images):

#### **EfficientNet-B0** Classification Report
```
              precision    recall  f1-score   support

      NORMAL       0.86      0.84      0.85       234
   PNEUMONIA       0.91      0.92      0.91       390

    accuracy                           0.89       624
   macro avg       0.89      0.88      0.88       624
weighted avg       0.89      0.89      0.89       624
```

**Key Metrics:**
- **Overall Accuracy**: 89%
- **Sensitivity (Pneumonia Recall)**: 92% - Excellent at detecting pneumonia cases
- **Specificity (Normal Recall)**: 84% - Good at identifying normal cases
- **Pneumonia Precision**: 91% - High confidence in positive predictions
- **F1-Score**: 0.91 (Pneumonia), 0.85 (Normal)

#### **ResNet50 (E4)** Classification Report
```
              precision    recall  f1-score   support

      NORMAL       0.89      0.76      0.82       234
   PNEUMONIA       0.87      0.94      0.90       390

    accuracy                           0.87       624
   macro avg       0.88      0.85      0.86       624
weighted avg       0.87      0.87      0.87       624
```

**Key Metrics:**
- **Overall Accuracy**: 87%
- **Sensitivity (Pneumonia Recall)**: 94% - Highest pneumonia detection rate
- **Specificity (Normal Recall)**: 76% - Moderate at identifying normal cases
- **Pneumonia Precision**: 87% - Good confidence in positive predictions
- **F1-Score**: 0.90 (Pneumonia), 0.82 (Normal)

### Comparative Analysis

| Metric | EfficientNet-B0 | ResNet50 |
|--------|-----------------|----------|
| **Accuracy** | 89% | 87% |
| **Pneumonia Recall (Sensitivity)** | 92% | 94% |
| **Normal Recall (Specificity)** | 84% | 76% |
| **Pneumonia Precision** | 91% | 87% |
| **Model Size** | ~5.3M params | ~23.5M params |
| **Inference Speed** | Faster | Slower |

### Performance Insights

✅ **EfficientNet-B0 Advantages:**
- Better overall accuracy (89% vs 87%)
- Better specificity for normal cases (84% vs 76%)
- Smaller model size for faster inference
- Balanced performance across both classes

✅ **ResNet50 Advantages:**
- Highest pneumonia detection rate (94% sensitivity)
- Better precision on normal cases (89% vs 86%)
- Better for minimizing false negatives (critical in medical context)

### Test Set Statistics
- Total Test Samples: 624
- Normal Cases: 234 (37.5%)
- Pneumonia Cases: 390 (62.5%)

> **Clinical Note:** The high sensitivity (92-94%) for pneumonia detection is particularly important in medical applications, as missing a pneumonia case is more critical than a false positive.

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for different image formats
- [ ] Implement ensemble predictions
- [ ] Add batch processing capability
- [ ] Create REST API endpoint
- [ ] Add explainability metrics (LIME, SHAP)
- [ ] Implement uncertainty quantification
- [ ] Add more optimizer options
- [ ] Create performance comparison dashboard

---

## ⚠️ Medical Disclaimer

**IMPORTANT NOTICE:**

This pneumonia detection system is:
- ✅ Intended for **educational and research purposes only**
- ❌ **NOT** intended for clinical diagnosis
- ❌ **NOT** FDA-approved or medically certified
- ❌ **NOT** suitable for real-world clinical use without extensive validation

**You must:**
- Always consult qualified medical professionals for diagnosis
- Never rely solely on this system for medical decisions
- Understand the model's limitations and potential for false positives/negatives
- Use this only as a research/educational tool

**Liability:** The authors assume no responsibility for any consequences of using this system.

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

