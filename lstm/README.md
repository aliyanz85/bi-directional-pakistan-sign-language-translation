# 🤟 Sign Language Recognition System

A professional-grade sign language to text conversion system using MediaPipe and LSTM neural networks.

## 🎯 Project Overview

This system recognizes sign language gestures in real-time using:

- **MediaPipe Holistic** for landmark detection (543 points: pose, face, hands)
- **Bidirectional LSTM** neural network for temporal sequence modeling
- **Real-time webcam inference** with smooth predictions

Currently trained on: **good** and **funny** signs

## 📁 Project Structure

```
MediaPipe/
├── 1_extract_data.py          # Extract and preprocess video data
├── 2_train_model.py            # Train LSTM model with best practices
├── 3_realtime_detection.py    # Real-time webcam detection
├── process_data.py             # Legacy full dataset processor
├── requirements.txt            # Python dependencies
├── videosDataset/              # Raw video data
│   ├── good/                   # 8 videos
│   └── funny/                  # 8 videos
├── processed_data/             # Generated after step 1
│   ├── dataset.npz             # Train/val/test splits
│   └── classes.json            # Class labels
├── models/                     # Generated after step 2
│   ├── sign_language_model_final.h5
│   └── model_info.json
└── logs/                       # Training logs and history
```

## 🚀 Quick Start

### 1️⃣ Install Dependencies

```powershell
# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Install requirements
pip install -r requirements.txt
```

### 2️⃣ Extract and Preprocess Data

```powershell
python 1_extract_data.py
```

**What it does:**

- Processes videos for 'good' and 'funny' signs
- Extracts 543 MediaPipe landmarks per frame
- Normalizes data (position & scale invariant)
- Standardizes to 30 frames per video
- Creates train (70%), validation (15%), test (15%) splits
- Saves to `processed_data/dataset.npz`

**Expected output:** 16 sequences (8 good + 8 funny)

### 3️⃣ Train the Model

```powershell
python 2_train_model.py
```

**What it does:**

- Builds Bidirectional LSTM architecture
- Implements best practices:
  - Early stopping
  - Learning rate reduction
  - Model checkpointing
  - Dropout & batch normalization
  - L2 regularization
- Trains for up to 100 epochs
- Saves best model to `models/`
- Generates training plots

**Expected training time:** 5-10 minutes

### 4️⃣ Run Real-time Detection

```powershell
python 3_realtime_detection.py
```

**What it does:**

- Opens webcam feed
- Detects landmarks in real-time
- Predicts sign language gestures
- Shows confidence scores
- Displays smooth predictions

**Controls:**

- Perform gestures in front of camera
- Press `q` to quit

## 📊 Model Architecture

```
Input: (30 frames, 1662 features)
    ↓
Bidirectional LSTM (128 units) + BatchNorm + Dropout(0.3)
    ↓
Bidirectional LSTM (64 units) + BatchNorm + Dropout(0.3)
    ↓
LSTM (32 units) + BatchNorm + Dropout(0.2)
    ↓
Dense (64 units, ReLU) + Dropout(0.2)
    ↓
Dense (32 units, ReLU)
    ↓
Output: Softmax (2 classes)
```

**Total Parameters:** ~2.5M (varies with exact architecture)

## 🎨 Features

### Data Processing

✅ Position-invariant normalization (relative to nose)  
✅ Scale-invariant normalization (relative to shoulder width)  
✅ Handles variable-length videos (sampling/padding)  
✅ Stratified train/val/test split

### Model Training

✅ Bidirectional LSTM for better temporal understanding  
✅ Early stopping to prevent overfitting  
✅ Learning rate scheduling  
✅ Model checkpointing (saves best model)  
✅ TensorBoard logging  
✅ Training visualization plots

### Real-time Detection

✅ Smooth predictions (rolling average)  
✅ Confidence thresholding  
✅ Visual feedback (landmarks + UI)  
✅ FPS counter  
✅ Mirror-flipped camera for natural interaction

## 📈 Performance

With only 16 training samples:

- **Expected accuracy:** 80-100% (small dataset, simple binary classification)
- **Real-time FPS:** 20-30 (depending on hardware)
- **Inference latency:** <50ms per prediction

## 🔧 Configuration

Edit hyperparameters in each script:

**`1_extract_data.py`:**

```python
SELECTED_CLASSES = ['good', 'funny']  # Add more classes
SEQ_LENGTH = 30  # Frames per video
```

**`2_train_model.py`:**

```python
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15  # Early stopping
```

**`3_realtime_detection.py`:**

```python
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to show
SMOOTHING_WINDOW = 5  # Frames to average
```

## 📝 Adding More Signs

To expand to more sign language words:

1. Add videos to `videosDataset/<word>/`
2. Update `SELECTED_CLASSES` in `1_extract_data.py`
3. Re-run the complete pipeline (steps 2-4)

## 🐛 Troubleshooting

**Issue:** Low accuracy  
**Solution:** Add more training videos (aim for 20+ per class)

**Issue:** Webcam not working  
**Solution:** Check camera permissions and index in `cv2.VideoCapture(0)`

**Issue:** "No module named tensorflow"  
**Solution:** Install requirements: `pip install -r requirements.txt`

**Issue:** Model predicts wrong class  
**Solution:** Increase `CONFIDENCE_THRESHOLD` or add more training data

## 📚 Technical Details

### Feature Vector (1662 dimensions per frame)

- **Pose:** 33 landmarks × 4 values (x, y, z, visibility) = 132
- **Face:** 468 landmarks × 3 values (x, y, z) = 1404
- **Left Hand:** 21 landmarks × 3 values (x, y, z) = 63
- **Right Hand:** 21 landmarks × 3 values (x, y, z) = 63

### Normalization Formula

```
normalized_landmark = (landmark - nose_position) / shoulder_width
```

This makes the model robust to:

- Different camera distances
- Different body sizes
- Different positions in frame

## 🎓 Best Practices Implemented

1. **Data split:** Stratified 70/15/15 train/val/test
2. **Regularization:** Dropout + L2 + BatchNorm
3. **Optimization:** Adam + LR scheduling
4. **Monitoring:** Early stopping + checkpointing
5. **Reproducibility:** Random seed setting
6. **Documentation:** Comprehensive logging

## 🚧 Future Improvements

- [ ] Data augmentation (time warping, noise injection)
- [ ] Attention mechanisms for better temporal modeling
- [ ] Multi-word sentence recognition
- [ ] Transfer learning from larger datasets
- [ ] Model quantization for mobile deployment

## 📄 License

This project is for educational purposes (FYP).

## 👨‍💻 Author

Created for Final Year Project (FYP) - Sign Language Recognition System

---

**Note:** This is a professional implementation following ML best practices. The system is designed to be scalable to more sign language words.
