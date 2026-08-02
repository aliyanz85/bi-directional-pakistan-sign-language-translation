# PSL Recognition Setup Guide

## 🎯 Feature Overview

This implementation provides **real-time Pakistan Sign Language (PSL) to Text & Speech** conversion using:

- **Frontend**: React + MediaPipe.js for hand landmark detection
- **Backend**: FastAPI + TensorFlow for PSL sign recognition
- **Model**: TCN + Transformer (80.95% accuracy, 32 PSL words)

---

## 📦 What Was Implemented

### Backend (`voice-backend/`)

1. **`app/services/psl_inference.py`**
   - Loads the trained PSL model on startup
   - Applies normalization to input sequences
   - Runs inference and returns predictions with confidence scores

2. **`app/api/psl.py`**
   - `POST /api/psl/recognize` - Main recognition endpoint
   - `GET /api/psl/model-info` - Model metadata
   - `GET /api/psl/health` - Service health check

3. **`app/main.py`** (Updated)
   - Loads PSL model on application startup
   - Includes PSL router at `/api/psl/*`

4. **`requirements.txt`** (Updated)
   - Added `tensorflow>=2.10.0,<2.16.0`
   - Added `numpy>=1.23.0,<2.0.0`

### Frontend (`voice-frontend/`)

1. **`src/api/pslApi.js`**
   - API service layer for PSL recognition
   - Validates sequences before sending to backend

2. **`src/hooks/useMediaPipe.js`**
   - Integrates MediaPipe Hands for landmark detection
   - Extracts 188-dimensional feature vectors from hand landmarks
   - Draws landmarks on canvas overlay

3. **`src/hooks/usePSLRecognition.js`**
   - Manages 60-frame sequence buffer
   - Auto-triggers recognition when buffer is full
   - Builds sentences from recognized words
   - Implements cooldown to prevent over-calling API

4. **`src/pages/PSLtoText.jsx`** (Refactored)
   - Live camera feed with hand landmark visualization
   - Real-time PSL recognition display
   - Sentence builder with text-to-speech
   - Progress indicators and statistics

5. **`package.json`** (Updated)
   - Added `@mediapipe/hands`
   - Added `@mediapipe/camera_utils`
   - Added `@mediapipe/drawing_utils`

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9+ (for backend)
- Node.js 16+ (for frontend)
- Webcam (for PSL recognition)

### Step 1: Install Backend Dependencies

```bash
cd voice-backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: TensorFlow installation may take several minutes.

### Step 2: Verify Model Files Exist

Ensure these files exist:
```
transformer/models/saved_models/tcn_transformer_20251203_125022/
├── best_model.h5                     # Model weights (~2.4 MB)
├── normalization_params.json         # Feature normalization stats
└── training_summary.json             # Model metadata

transformer/data/extracted_landmarks/
└── metadata.json                     # Class labels
```

If these files are missing, the backend will fail to start.

### Step 3: Start Backend Server

```bash
cd voice-backend
uvicorn app.main:app --reload --port 8000
```

**Expected startup logs:**
```
INFO: Starting D-VOICE backend...
INFO: Loading PSL recognition model...
INFO: Loading model from: <path>/best_model.h5
INFO: Model loaded successfully. Input shape: (None, 60, 188)
INFO: Loaded 32 class labels: alert, book, careful, ...
INFO: PSL model initialization complete!
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Verify backend is running:**

Open http://localhost:8000 - You should see:
```json
{
  "message": "D-VOICE Animation Service API",
  "version": "1.0.0",
  "endpoints": {
    "animations": "/api/animations",
    "psl_recognition": "/api/psl/recognize",
    "psl_health": "/api/psl/health",
    "psl_model_info": "/api/psl/model-info"
  }
}
```

Test PSL health endpoint:
```bash
curl http://localhost:8000/api/psl/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "PSL Recognition",
  "model_loaded": true,
  "num_classes": 32
}
```

### Step 4: Install Frontend Dependencies

```bash
cd voice-frontend
npm install
```

This will install MediaPipe and all required packages.

### Step 5: Configure Environment Variables

Ensure `voice-frontend/.env` contains:
```env
VITE_API_BASE_URL=http://localhost:8000
```

### Step 6: Start Frontend Development Server

```bash
cd voice-frontend
npm run dev
```

**Expected output:**
```
VITE v5.0.8  ready in 523 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
➜  press h to show help
```

### Step 7: Test PSL Recognition

1. Open http://localhost:5173/psl-to-text
2. Click **"Start Camera"** button
3. Allow webcam access when prompted
4. You should see:
   - Live video feed (mirrored)
   - Green hand landmarks overlayed when hands are detected
   - Buffer progress bar (0-60 frames)
   - FPS counter

5. Perform a PSL sign (e.g., "ready", "yes", "good")
6. Hold the sign steady for 2 seconds
7. Wait for buffer to fill (60 frames)
8. Recognition will trigger automatically
9. Predicted word appears in "Current Sign" panel
10. Word is added to sentence if confidence > 60%

### Step 8: Test Text-to-Speech

1. Build a sentence by performing multiple signs
2. Click the **"Speak"** button
3. Browser should speak the sentence aloud

---

## 🔧 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (localhost:5173)                 │
│                                                             │
│  ┌────────────────┐         ┌──────────────────┐          │
│  │  PSLtoText.jsx │◄────────┤  useMediaPipe    │          │
│  │                │         │  (MediaPipe.js)  │          │
│  │                │         └──────────────────┘          │
│  │                │                   │                    │
│  │                │         ┌──────────────────┐          │
│  │                │◄────────┤ usePSLRecognition│          │
│  └────────┬───────┘         └─────────┬────────┘          │
│           │                           │                    │
│           │ Camera Stream             │ 188-dim features   │
│           │                           │ (60 frames)        │
│           ▼                           ▼                    │
│  ┌───────────────┐         ┌──────────────────┐          │
│  │ Video Element │         │  Sequence Buffer  │          │
│  │ + Canvas      │         │  [60 x 188]       │          │
│  └───────────────┘         └─────────┬────────┘          │
│                                      │                    │
│                                      │ POST /psl/recognize│
└──────────────────────────────────────┼────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend (localhost:8000)                       │
│                                                             │
│  ┌──────────────┐         ┌──────────────────┐            │
│  │  psl.py      │────────►│ psl_inference.py │            │
│  │  (FastAPI)   │         │                  │            │
│  └──────────────┘         └─────────┬────────┘            │
│                                     │                      │
│                                     ▼                      │
│                          ┌──────────────────┐             │
│                          │  TensorFlow      │             │
│                          │  TCN+Transformer │             │
│                          │  Model (32 words)│             │
│                          └─────────┬────────┘             │
│                                    │                      │
│                                    ▼                      │
│                          ┌──────────────────┐             │
│                          │  Predictions     │             │
│                          │  + Confidence    │             │
│                          └──────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow

### 1. **Camera Frame Capture** (30 FPS)

```
Webcam → Video Element → MediaPipe Hands
```

### 2. **Landmark Extraction** (Per Frame)

```
MediaPipe Hands
  ↓
Detects up to 2 hands
  ↓
21 landmarks per hand (x, y, z coordinates)
  ↓
Feature Extraction (useMediaPipe.js)
  ↓
188-dimensional vector:
  - 63 features: Wrist-relative coordinates (hand 1)
  - 29 features: Geometric features (distances, angles)
  - 2 features: Hand label (left/right one-hot)
  - [Repeat for hand 2]
  ↓
Add to sequence buffer
```

### 3. **Sequence Buffering** (Rolling Window)

```
Sequence Buffer (usePSLRecognition.js)
  ↓
Maintains last 60 frames
  ↓
When buffer full (60/60):
  ↓
Auto-trigger recognition
```

### 4. **Backend Recognition**

```
POST /api/psl/recognize
  ↓
Validate: sequence shape = (60, 188)
  ↓
Normalize: (x - mean) / std
  ↓
Add batch dimension: (1, 60, 188)
  ↓
TensorFlow model.predict()
  ↓
Softmax probabilities (32 classes)
  ↓
Argmax → predicted class ID
  ↓
Map to label (e.g., "ready")
  ↓
Return JSON:
{
  "label": "ready",
  "class_id": 27,
  "confidence": 0.93,
  "top_predictions": [...]
}
```

### 5. **Frontend Display**

```
Prediction received
  ↓
Update current prediction display
  ↓
If confidence > threshold (0.6):
  ↓
Add word to sentence
  ↓
Update sentence display
  ↓
User can click "Speak" for TTS
```

---

## 🎨 UI Components

### Camera Feed Section
- Live video with mirrored display
- Hand landmark overlay (green/red for left/right)
- Hands detected counter
- FPS counter
- Buffer progress bar (0-60 frames)
- "Recognizing..." indicator during API call

### Current Sign Panel
- Large predicted word display
- Confidence percentage
- Top 3 predictions with progress bars

### Sentence Builder
- Text area showing accumulated words
- Word count and character count
- **Speak** button (text-to-speech)
- **Undo** button (remove last word)
- **Clear** button (reset sentence)

### Live Statistics
- Buffer status (X/60 frames)
- Words recognized count
- Hands detected count

---

## 🧪 Testing the System

### Test 1: Backend Health Check

```bash
curl http://localhost:8000/api/psl/health
```

**Expected**:
```json
{
  "status": "healthy",
  "service": "PSL Recognition",
  "model_loaded": true,
  "num_classes": 32
}
```

### Test 2: Model Info

```bash
curl http://localhost:8000/api/psl/model-info
```

**Expected**:
```json
{
  "loaded": true,
  "input_shape": "(None, 60, 188)",
  "output_shape": "(None, 32)",
  "num_classes": 32,
  "classes": ["alert", "book", "careful", ...],
  "normalization_features": 188
}
```

### Test 3: Manual Recognition Request

Create a test file `test_psl.py`:

```python
import requests
import numpy as np

# Create dummy sequence (60 frames × 188 features)
sequence = np.random.randn(60, 188).tolist()

# Send to backend
response = requests.post(
    'http://localhost:8000/api/psl/recognize',
    json={'sequence': sequence}
)

print(response.json())
```

Run:
```bash
python test_psl.py
```

**Expected**: Prediction response with label, confidence, and top predictions.

### Test 4: Frontend End-to-End

1. **Navigate** to http://localhost:5173/psl-to-text
2. **Start Camera**
3. **Verify**:
   - ✅ Video feed displays
   - ✅ Hand landmarks appear when hands visible
   - ✅ Buffer fills to 60/60
   - ✅ Recognition triggers automatically
   - ✅ Prediction appears in "Current Sign"
   - ✅ Word added to sentence if confidence > 60%
4. **Test TTS**: Click "Speak" button
5. **Test Controls**:
   - Undo last word
   - Clear sentence
   - Reset buffer

---

## 🐛 Troubleshooting

### Backend Issues

#### Model fails to load

**Error**: `FileNotFoundError: Model file not found at: ...`

**Solution**:
- Verify model files exist in `transformer/models/saved_models/tcn_transformer_20251203_125022/`
- Check file permissions
- Ensure you're running from project root

#### TensorFlow import error

**Error**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
pip install tensorflow>=2.10.0
```

For GPU support:
```bash
pip install tensorflow-gpu>=2.10.0
```

#### CORS errors

**Error**: `Access to fetch at 'http://localhost:8000' ... blocked by CORS policy`

**Solution**:
- Verify `CORS_ORIGINS` in `voice-backend/app/core/config.py` includes `http://localhost:5173`
- Restart backend server

### Frontend Issues

#### MediaPipe fails to load

**Error**: `Cannot find module '@mediapipe/hands'`

**Solution**:
```bash
cd voice-frontend
npm install @mediapipe/hands @mediapipe/camera_utils
```

#### Camera permission denied

**Error**: User denies camera access

**Solution**:
- Ensure browser has camera permissions
- Try a different browser (Chrome recommended)
- Check if another application is using the camera

#### No hands detected

**Issue**: Green landmarks don't appear

**Solution**:
- Ensure good lighting
- Hold hands clearly in frame
- Try moving closer to camera
- Check camera quality/resolution

#### Recognition not triggering

**Issue**: Buffer fills but no prediction

**Solution**:
- Check browser console for errors
- Verify backend is running (http://localhost:8000/health)
- Check network tab for failed requests
- Ensure cooldown has passed (1.5 seconds)

#### Low confidence predictions

**Issue**: Confidence < 60%, words not added to sentence

**Solution**:
- Hold sign steadier for full 2 seconds
- Ensure sign matches one of the 32 available words
- Improve lighting conditions
- Position hands clearly in frame
- Check if model recognizes the specific sign (some signs may have lower accuracy)

---

## 📝 API Reference

### POST /api/psl/recognize

Recognize PSL sign from a sequence of landmark features.

**Request Body**:
```json
{
  "sequence": [
    [0.1, 0.2, ..., 0.5],  // Frame 1: 188 features
    [0.1, 0.2, ..., 0.5],  // Frame 2: 188 features
    ...                    // (60 frames total)
  ]
}
```

**Response** (200 OK):
```json
{
  "label": "ready",
  "class_id": 27,
  "confidence": 0.93,
  "top_predictions": [
    {"label": "ready", "class_id": 27, "confidence": 0.93},
    {"label": "quick", "class_id": 25, "confidence": 0.04},
    {"label": "yes", "class_id": 31, "confidence": 0.02}
  ]
}
```

**Error Responses**:

- `400 Bad Request`: Invalid input shape
  ```json
  {"detail": "Invalid input shape. Expected (60, 188), got (30, 188)"}
  ```

- `500 Internal Server Error`: Model inference failed
  ```json
  {"detail": "Model inference failed: [error details]"}
  ```

---

## 🎓 Available PSL Signs (32)

| Category | Words |
|----------|-------|
| **Communication** | alert, no, ready, yes |
| **Quality Adjectives** | careful, cheap, crazy, dangerous, decent, dumb, extreme, fantastic, far, foreign, funny, good, healthy, heavy, important, intelligent, interesting, late, new, noisy, quick, secure, smart |
| **Emotions** | excited, fearful, peaceful |
| **Objects** | book |
| **Quantity** | less |

---

## 🔮 Next Steps / Future Enhancements

### Backend
- Add WebSocket support for continuous streaming
- Implement user authentication and session management
- Add logging and analytics
- Deploy to production server
- Optimize model inference speed (TensorRT, ONNX)

### Frontend
- Add option to manually trigger recognition
- Support for continuous phrase recognition
- Save/export sentences to file
- Visual feedback during sign performance
- Add tutorial/demo mode
- Mobile-responsive improvements
- Offline mode with cached model

### Model
- Increase vocabulary (currently 32 words)
- Add sentence-level recognition (grammar)
- Support for dynamic gestures
- Multi-language support (Urdu subtitles)

---

## 📞 Support

**Issues**:
- Backend not starting: Check model files and Python dependencies
- Frontend not loading: Run `npm install` and check `.env` file
- Recognition not working: Verify camera permissions and backend health

**Logs to check**:
- Backend: Terminal where `uvicorn` is running
- Frontend: Browser console (F12 → Console tab)
- Network: Browser DevTools → Network tab

---

## ✅ Success Checklist

- [ ] Backend starts without errors
- [ ] Backend health check returns "healthy"
- [ ] Frontend loads at localhost:5173
- [ ] Camera feed displays when "Start Camera" clicked
- [ ] Hand landmarks appear (green/red) when hands visible
- [ ] Buffer fills to 60/60 frames
- [ ] Recognition triggers automatically
- [ ] Predicted word appears with confidence score
- [ ] Words added to sentence when confidence > 60%
- [ ] Text-to-speech works when "Speak" clicked
- [ ] Sentence controls work (undo, clear)

---

**🎉 Congratulations! Your PSL to Text & Speech system is now fully operational!**

Built with ❤️ for the D-VOICE FYP Project
