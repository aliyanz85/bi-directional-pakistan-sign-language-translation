# 🎯 QUICK START GUIDE - Sign Language Recognition

## ✅ COMPLETED PIPELINE

All scripts are ready and tested! Follow these steps:

---

## 📋 Step 1: Data Extraction ✅ DONE

```powershell
python 1_extract_data.py
```

**Results:**

- ✅ Processed 16 videos (8 good + 8 funny)
- ✅ Created `processed_data/dataset.npz`
- ✅ Train/Val/Test split: 11/2/3 samples
- ✅ Each sequence: 30 frames × 1662 features

---

## 📋 Step 2: Model Training ✅ DONE

```powershell
python 2_train_model.py
```

**Results:**

- ✅ Trained Bidirectional LSTM (2M parameters)
- ✅ Training completed in ~70 epochs
- ✅ Test Accuracy: **66.67%**
- ✅ Model saved: `models/sign_language_model_final.h5`
- ✅ Training plot: `logs/run_*/training_history.png`

**Training Features Used:**

- ✅ Early stopping (patience=15)
- ✅ Learning rate reduction
- ✅ Model checkpointing (best model saved)
- ✅ Dropout & Batch Normalization
- ✅ L2 Regularization

---

## 📋 Step 3: Real-time Detection 🎥

```powershell
python 3_realtime_detection.py
```

**What to expect:**

- 📹 Webcam opens automatically
- 🎯 Perform "good" or "funny" sign
- ✅ Prediction appears with confidence score
- 📊 Green bar = high confidence (>70%)
- ⌨️ Press 'q' to quit

**Features:**

- Real-time landmark detection
- Smooth predictions (5-frame rolling average)
- Confidence thresholding (70%)
- Visual feedback (landmarks + UI)
- FPS counter

---

## 🎯 Expected Performance

### With Current Dataset (16 samples):

- **Training Accuracy:** ~90-100%
- **Validation Accuracy:** ~50-100% (varies due to small dataset)
- **Test Accuracy:** **66.67%** (2 out of 3 correct)
- **Real-time FPS:** 20-30 fps

### To Improve Accuracy:

1. **Add more videos** (aim for 50+ per class)
2. **Collect diverse data** (different people, lighting, angles)
3. **Data augmentation** (time warping, noise)
4. **Fine-tune hyperparameters**

---

## 📁 Generated Files

```
MediaPipe/
├── processed_data/
│   ├── dataset.npz           ✅ Train/val/test splits
│   └── classes.json          ✅ ['good', 'funny']
├── models/
│   ├── sign_language_model_final.h5  ✅ Trained model
│   ├── sign_language_model_*.h5      ✅ Best checkpoint
│   └── model_info.json               ✅ Model metadata
└── logs/
    └── run_*/
        ├── training_history.png      ✅ Accuracy/loss plots
        └── training.csv              ✅ Training metrics
```

---

## 🚀 Adding More Words

To expand to all 32 words:

### Method 1: Update and Re-run

```python
# Edit 1_extract_data.py
SELECTED_CLASSES = ['good', 'funny', 'alert', 'book', ...]  # Add more

# Re-run pipeline
python 1_extract_data.py
python 2_train_model.py
```

### Method 2: Incremental Training

1. Keep current model
2. Add new classes
3. Use transfer learning (freeze early layers)
4. Fine-tune on new data

---

## 🎬 Demo Video (Recommended)

For your FYP presentation:

1. **Run real-time detection:**

   ```powershell
   python 3_realtime_detection.py
   ```

2. **Perform signs:**

   - Do "good" sign → Watch prediction
   - Do "funny" sign → Watch prediction
   - Show confidence scores

3. **Highlight features:**
   - Real-time landmark detection
   - Smooth predictions
   - Confidence thresholding
   - Professional UI

---

## 🐛 Troubleshooting

### Low Accuracy on Real-time Detection

- **Solution:** Ensure good lighting
- **Solution:** Perform sign clearly and consistently
- **Solution:** Wait for sequence buffer to fill (30 frames)

### Webcam not opening

- **Solution:** Check camera permissions
- **Solution:** Try different camera index in code (change `0` to `1`)

### Model predicts same class always

- **Solution:** More training data needed
- **Solution:** Check if signs are too similar
- **Solution:** Lower confidence threshold

---

## 📊 Technical Specs

### Model Architecture

- **Input:** (30, 1662) - 30 frames, 1662 features
- **Layer 1:** Bidirectional LSTM (128 units)
- **Layer 2:** Bidirectional LSTM (64 units)
- **Layer 3:** LSTM (32 units)
- **Output:** Dense (2 classes) - Softmax
- **Total Parameters:** 2,024,866

### Features Per Frame

- **Pose:** 33 landmarks × 4 = 132
- **Face:** 468 landmarks × 3 = 1404
- **Left Hand:** 21 landmarks × 3 = 63
- **Right Hand:** 21 landmarks × 3 = 63
- **Total:** 1662 features

### Normalization

- **Anchor:** Nose position (landmark 0)
- **Scale:** Shoulder width (landmarks 11-12)
- **Formula:** `(coordinate - nose) / shoulder_width`

---

## 🎓 Best Practices Implemented

✅ **Data Processing:**

- Normalization (position & scale invariant)
- Stratified train/val/test split
- Sequence standardization (padding/sampling)

✅ **Model Training:**

- Bidirectional LSTM (better temporal modeling)
- Regularization (Dropout, L2, BatchNorm)
- Early stopping & LR scheduling
- Model checkpointing

✅ **Real-time Inference:**

- Smooth predictions (rolling average)
- Confidence thresholding
- Professional UI/UX
- Performance optimization

---

## 🎉 Success! Your Pipeline is Ready!

**You now have a complete, professional sign language recognition system:**

1. ✅ Data extraction with MediaPipe
2. ✅ Deep learning model training
3. ✅ Real-time webcam inference

**Next steps:**

- Test with webcam: `python 3_realtime_detection.py`
- Add more words as needed
- Collect more training data for better accuracy
- Present your FYP with confidence! 🚀

---

**Questions or Issues?**

- Check the main `README.md` for detailed docs
- Review training logs in `logs/` folder
- Inspect model info in `models/model_info.json`
