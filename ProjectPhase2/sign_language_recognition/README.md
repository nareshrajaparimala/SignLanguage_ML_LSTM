# Sign Language Recognition with Finger Bend Analysis

Advanced hand gesture recognition system using MediaPipe and CNN+LSTM neural networks with comprehensive finger bend analysis.

## Features

- **Real-time Hand Detection**: MediaPipe-powered hand landmark detection
- **Finger Bend Analysis**: Calculates finger angles and bend ratios
- **CNN+LSTM Model**: Deep learning for gesture sequence recognition
- **Gesture Management**: Capture, train, and predict custom gestures
- **Web Interface**: React frontend for easy interaction

## Quick Start

### 1. Start Backend Server
```bash
cd camera_backend
source mediapipe_env/bin/activate
python advanced_mediapipe_server.py
```

### 2. Start Frontend
```bash
cd frontend
npm run dev
```

### 3. Access Application
Open http://localhost:5173 in your browser

## Usage

1. **Capture Gestures**:
   - Enter gesture name and meaning
   - Click "Start Capturing Gesture"
   - Capture 30 frames by clicking "Capture Frame"
   - Click "Save Gesture"

2. **Train Model**:
   - Capture at least 2 different gestures
   - Click "Train CNN+LSTM Model"
   - Model trains with 2-100 samples

3. **Predict Gestures**:
   - Click "Start Prediction"
   - Perform gesture in front of camera
   - View prediction results

## Technical Details

- **Backend**: FastAPI with MediaPipe and TensorFlow
- **Frontend**: React with Vite
- **Model**: CNN+LSTM for temporal sequence learning
- **Features**: 148-dimensional feature vectors including landmarks, angles, and bend ratios
- **Port**: Backend runs on 5002, Frontend on 5173

## Requirements

- Python 3.12+
- Node.js 16+
- Webcam
- MediaPipe, TensorFlow, FastAPI, React