# VOICE PSL - Pakistan Sign Language Recognition System

## Project Overview

A production-ready, real-time Pakistan Sign Language (PSL) recognition system using deep learning and MediaPipe hand tracking, deployed in the browser using TensorFlow.js.

## Features

- **High Accuracy**: 80%+ recognition accuracy using TCN+Transformer architecture
- **Low Latency**: <50ms inference time in browser
- **Real-time Processing**: Live webcam-based sign recognition
- **Robust Pipeline**: Comprehensive data augmentation and error handling
- **Browser-based**: No server required for inference

## Technology Stack

- **Training**: Python, TensorFlow/Keras, MediaPipe
- **Frontend**: React.js + TypeScript
- **Hand Tracking**: MediaPipe Hands
- **Deployment**: TensorFlow.js with WebGL acceleration

## Project Structure

```
voice-psl/
├── data/                          # Dataset directory
│   ├── raw_videos/               # Original sign language videos
│   ├── extracted_landmarks/      # Preprocessed landmark data
│   └── augmented/                # Augmented training data
├── models/                        # Model training and inference
│   ├── training/                 # Training scripts
│   │   ├── train.py             # Main training script
│   │   ├── model_architecture.py # Model definitions
│   │   ├── data_loader.py       # Data loading utilities
│   │   ├── augmentation.py      # Data augmentation pipeline
│   │   └── evaluator.py         # Model evaluation
│   ├── inference/                # Inference utilities
│   │   └── tfjs_model/          # TensorFlow.js model
│   └── saved_models/             # Trained model checkpoints
├── frontend/                      # React application
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── inference/           # Inference engine
│   │   └── utils/               # Utility functions
│   └── public/
├── backend/                       # Optional API server
│   ├── api/
│   └── services/
├── notebooks/                     # Jupyter notebooks for experiments
└── config/                        # Configuration files
```

## Setup Instructions

### 1. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your raw videos in `data/raw_videos/` organized by word:

```
data/raw_videos/
├── word1/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── word2/
│   └── ...
```

### 3. Extract Landmarks

```bash
python models/training/extract_landmarks.py
```

### 4. Train Model

```bash
python models/training/train.py
```

### 5. Convert to TensorFlow.js

```bash
python models/training/convert_to_tfjs.py
```

### 6. Run Frontend

```bash
cd frontend
npm install
npm start
```

## Model Architecture

- **Temporal Convolutional Network (TCN)**: Captures temporal patterns
- **Transformer Encoder**: Models long-range dependencies
- **Feature Engineering**: Hand geometry + normalized landmarks
- **Regularization**: Dropout, label smoothing, data augmentation

## Performance Targets

- **Accuracy**: >80% on test set
- **Top-3 Accuracy**: >95%
- **Inference Latency**: <50ms in browser
- **Model Size**: <5MB after quantization

## Training Strategy

- K-fold cross-validation (5 folds)
- Extensive data augmentation (10x expansion)
- Transfer learning with MoViNet
- Ensemble methods for final deployment

## Citation

If you use this system in your research, please cite:

```
@project{voice-psl-2025,
  title={VOICE PSL: Real-time Pakistan Sign Language Recognition},
  author={[Your Name]},
  year={2025}
}
```

## License

MIT License
