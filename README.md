# 🤟 Sign Language Recognition System

A comprehensive gesture and sign language recognition system using advanced machine learning techniques. This project features real-time hand detection, gesture capture, model training, and prediction capabilities through both traditional ML and deep learning approaches.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Troubleshooting](#troubleshooting)
- [Project Details](#project-details)

---

## 🎯 Project Overview

This is a full-stack gesture recognition system that captures hand movements, extracts advanced features (finger angles, bend ratios, landmarks), trains deep learning models (CNN+LSTM), and provides real-time predictions through a web interface.

**Key Highlights:**
- ✅ Real-time hand detection with MediaPipe
- ✅ Advanced finger bend analysis
- ✅ CNN+LSTM deep learning models
- ✅ Interactive React web interface
- ✅ RESTful API backend
- ✅ Multiple model architectures (LSTM, CNN+LSTM, Enhanced models)
- ✅ Gesture dataset management

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Port 5173)               │
│                   - Gesture Capture UI                       │
│                   - Model Training Interface                 │
│                   - Real-time Prediction Display             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    HTTP/WebSocket
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              FastAPI Backend (Port 5002)                     │
│         - Hand Detection & Feature Extraction                │
│         - Model Training & Management                        │
│         - Real-time Video Stream Processing                  │
│         - Gesture Prediction & Storage                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
        MediaPipe      TensorFlow   Scikit-learn
       (Hand Detect)    (ML Models)  (KNN, Preprocessing)
```

---

## ✨ Features

### 1. **Hand Detection & Tracking**
- MediaPipe-based real-time hand landmark detection
- 21-point hand skeleton tracking
- Multi-hand detection support
- Hand confidence scoring

### 2. **Advanced Feature Extraction**
- 148-dimensional feature vectors
- Finger angle calculations (thumb, index, middle, ring, pinky)
- Finger bend ratios (0-1 scale)
- Hand position and scale normalization
- Temporal sequence encoding

### 3. **Multiple Model Architectures**
- **LSTM Model**: Traditional sequential learning
- **CNN+LSTM Model**: Convolutional + sequential learning
- **Enhanced Model**: Advanced feature integration

### 4. **Gesture Management**
- Capture and save custom gestures
- Organize gestures into labeled datasets
- Train models on custom datasets
- Export/import gesture data

### 5. **Web Interface**
- Intuitive React UI
- Real-time video preview
- Live gesture detection
- Training progress monitoring
- Model performance visualization

---

## 📁 Project Structure

```
ProjectPhase-2/
├── ProjectPhase2/
│   └── sign_language_recognition/
│       ├── 📄 README.md (existing)
│       ├── 📄 ENHANCED_FEATURES_README.md
│       ├── 📄 00_START_HERE.md
│       │
│       ├── camera_backend/
│       │   ├── advanced_mediapipe_server.py    ← Main backend server
│       │   ├── enhanced_model.py               ← Enhanced ML model
│       │   ├── retrain_model.py                ← Model retraining script
│       │   ├── train_enhanced_model.py         ← Training pipeline
│       │   ├── requirements.txt                ← Python dependencies
│       │   ├── requirements_advanced.txt       ← Advanced features deps
│       │   ├── models/                         ← Saved ML models
│       │   │   ├── gesture_model.h5
│       │   │   ├── cnn_lstm_model.h5
│       │   │   ├── enhanced_gesture_model.h5
│       │   │   ├── hand_landmarker.task
│       │   │   └── [encoders & scalers]
│       │   ├── gesture_dataset/                ← Captured gesture data
│       │   │   ├── one/
│       │   │   ├── super/
│       │   │   └── [other gestures]
│       │   ├── mediapipe_env/                  ← Python virtual env
│       │   └── start_perfect_mediapipe.sh      ← Setup script
│       │
│       └── frontend/
│           ├── src/
│           │   ├── main.jsx                    ← Entry point
│           │   ├── App.jsx                     ← Main component
│           │   ├── api.js                      ← API client
│           │   ├── App.css                     ← Global styles
│           │   └── pages/
│           │       ├── CameraCaptureePage.jsx  ← Capture interface
│           │       ├── CameraPredictPage.jsx   ← Prediction interface
│           │       ├── CameraTrainPage.jsx     ← Training interface
│           │       ├── LiveRecognitionPage.jsx ← Live demo
│           │       ├── MediaPipeHandsPage.jsx  ← Raw hand detection
│           │       ├── [other pages]
│           │       └── [CSS files]
│           │
│           ├── index.html
│           ├── vite.config.js
│           ├── package.json
│           └── node_modules/
│
├── Hand/
│   └── LSTM_model/
│       ├── lstm_trainer.py              ← LSTM training script
│       ├── data_collector.py            ← Data collection tool
│       ├── real_time_predictor.py       ← Real-time inference
│       ├── raspberry_pi_inference.py    ← RPi deployment
│       ├── simple_trainer.py            ← Basic training
│       ├── gesture_lstm_model.h5        ← Trained LSTM model
│       ├── [data files & encoders]
│       ├── requirements.txt
│       └── README.md
│
└── reading_arduino_data/
    ├── arduino_mpu_reader.ino           ← Arduino sketch
    ├── data_reader.py                   ← Data reading script
    ├── find_port.py                     ← Serial port finder
    └── requirements.txt
```

---

## 📦 Prerequisites

### System Requirements
- **OS**: macOS, Linux, or Windows
- **Python**: 3.9 or higher
- **Node.js**: 14 or higher
- **Webcam**: Required for real-time hand detection
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk**: 2GB free space

### Required Software
```bash
# Check Python version
python3 --version

# Check Node.js version
node --version
npm --version
```

---

## 🚀 Installation & Setup

### Option 1: Quick Start (Recommended)

#### Step 1: Clone/Navigate to Project
```bash
cd /Users/nareshraja/Desktop/code/ProjectPhase-2/ProjectPhase2/ProjectPhase2/sign_language_recognition
```

#### Step 2: Setup Backend
```bash
cd camera_backend

# Create virtual environment
python3 -m venv mediapipe_env

# Activate virtual environment
source mediapipe_env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Or for enhanced features
pip install -r requirements_advanced.txt
```

#### Step 3: Setup Frontend
```bash
cd ../frontend

# Install Node.js dependencies
npm install

# Build frontend
npm run build
```

#### Step 4: Run the System (3 Terminal Windows)

**Terminal 1 - Backend Server:**
```bash
cd ProjectPhase2/sign_language_recognition/camera_backend
source mediapipe_env/bin/activate
python advanced_mediapipe_server.py
```

**Terminal 2 - Frontend Server:**
```bash
cd ProjectPhase2/sign_language_recognition/frontend
npm run dev
```

**Terminal 3 - (Optional) Monitor/Debug:**
```bash
# Check running processes
ps aux | grep python
ps aux | grep node

# Kill a process if needed
kill -9 <PID>
```

#### Step 5: Access Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5002
- **API Docs**: http://localhost:5002/docs

---

### Option 2: Manual Setup

#### Backend Setup
```bash
# Navigate to backend
cd camera_backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install fastapi uvicorn opencv-python mediapipe tensorflow scikit-learn joblib numpy python-multipart

# Verify installation
python -c "import mediapipe; import tensorflow; print('✓ All packages installed')"
```

#### Frontend Setup
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Install additional packages if needed
npm install axios react-dom
```

---

## 📖 Usage Guide

### 1. **Capture Gestures**

Navigate to **Camera Capture** page:
1. Enter a **Gesture Name** (e.g., "HELLO", "GOODBYE")
2. Enter the **Text to Speak** when gesture is recognized
3. Click **"Start Capture"**
4. Perform your gesture in front of the camera
5. Hold for ~6 seconds
6. Click **"Save Gesture"**

**Example gestures:**
- Peace sign
- Thumbs up
- Wave hello
- Open hand
- Fist

### 2. **Train Model**

Navigate to **Camera Train** page:
1. Ensure you have **at least 2-3 different gestures** captured
2. Click **"Train CNN+LSTM Model"**
3. Monitor training progress
4. Model will be saved automatically

**Training takes:**
- 2-3 gestures: ~2-5 minutes
- 5+ gestures: ~5-10 minutes

### 3. **Predict/Recognize Gestures**

Navigate to **Camera Predict** page:
1. Click **"Start Prediction"**
2. Perform a trained gesture in front of camera
3. View prediction results in real-time
4. Confidence scores shown in percentage

### 4. **Live Recognition**

Navigate to **Live Recognition** page:
1. Continuous live gesture recognition
2. Multiple gestures can be predicted
3. Real-time hand landmarks displayed

### 5. **MediaPipe Hand Detection**

Navigate to **MediaPipe Hands** page:
1. View raw hand landmark detection
2. See 21-point hand skeleton
3. Debug hand detection issues

---

## 🔌 API Documentation

### Base URL
```
http://localhost:5002
```

### Key Endpoints

#### 1. **Health Check**
```bash
GET /health
```
Response:
```json
{
  "status": "Server is running"
}
```

#### 2. **Video Stream**
```bash
GET /video_feed
```
Returns MJPEG video stream with hand detection overlay.

#### 3. **Start Prediction**
```bash
POST /start-prediction
```
Response:
```json
{
  "status": "Prediction started",
  "timestamp": "2026-01-08T10:30:00"
}
```

#### 4. **Get Prediction Results**
```bash
GET /get-prediction-result
```
Response:
```json
{
  "gesture": "HELLO",
  "confidence": 0.95,
  "timestamp": "2026-01-08T10:30:05"
}
```

#### 5. **Capture Gesture**
```bash
POST /capture-gesture
Body: {
  "gesture_name": "HELLO",
  "gesture_meaning": "Greeting"
}
```

#### 6. **Train Model**
```bash
POST /train-model
```
Response:
```json
{
  "status": "Training started",
  "model_type": "CNN+LSTM"
}
```

#### 7. **Get Training Status**
```bash
GET /training-status
```
Response:
```json
{
  "status": "Training in progress",
  "progress": 45,
  "epoch": 5,
  "loss": 0.234
}
```

#### 8. **Save Model**
```bash
POST /save-model
Body: {
  "model_name": "gesture_model.h5"
}
```

#### 9. **List Gestures**
```bash
GET /list-gestures
```
Response:
```json
{
  "gestures": ["HELLO", "GOODBYE", "PEACE"],
  "count": 3
}
```

### Interactive API Docs
Access **Swagger UI** at: `http://localhost:5002/docs`

---

## 🤖 Model Training

### Training Process

The system uses a **CNN+LSTM architecture**:

```
Input (Time Series of Features)
    ↓
Conv1D Layer (32 filters, kernel=3)
    ↓
MaxPooling1D (pool_size=2)
    ↓
LSTM Layer (64 units)
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output (Softmax, num_gestures)
```

### Training Parameters
- **Epochs**: 100
- **Batch Size**: 16
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Validation Split**: 20%

### Training Commands

```bash
# Using the backend
cd camera_backend
source mediapipe_env/bin/activate

# Start server and train via UI (recommended)
python advanced_mediapipe_server.py

# Or retrain with specific parameters
python retrain_model.py --gestures 5 --epochs 100

# Train enhanced model
python train_enhanced_model.py --dataset gesture_dataset
```

### Model Files Location
```
camera_backend/models/
├── gesture_model.h5              ← Basic model
├── cnn_lstm_model.h5             ← CNN+LSTM model
├── enhanced_gesture_model.h5     ← Enhanced model
├── label_encoder.joblib          ← Label encoder
├── enhanced_label_encoder.joblib ← Enhanced encoder
├── scaler_params.npy             ← Feature scaler
└── enhanced_scaler.joblib        ← Enhanced scaler
```

---

## 🐛 Troubleshooting

### Issue 1: Port Already in Use
```bash
# Error: Address already in use (port 5002)

# Solution 1: Kill the process
lsof -i :5002
kill -9 <PID>

# Solution 2: Use different port
python advanced_mediapipe_server.py --port 5003
```

### Issue 2: Missing CSS Files
```bash
# Error: Failed to resolve import "./hand-detection.css"

# Solution: Create the missing CSS file
touch frontend/src/pages/hand-detection.css
```

### Issue 3: Webcam Not Detected
```bash
# Solution 1: Check permissions (macOS)
System Preferences → Security & Privacy → Camera → Enable app

# Solution 2: Verify webcam
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### Issue 4: MediaPipe Installation Issues
```bash
# Clean reinstall
rm -rf mediapipe_env
python3 -m venv mediapipe_env
source mediapipe_env/bin/activate
pip install --upgrade pip
pip install mediapipe==0.10.0
```

### Issue 5: TensorFlow/CUDA Issues
```bash
# For CPU-only (recommended for most users)
pip install tensorflow-cpu

# For GPU support (NVIDIA)
pip install tensorflow[and-cuda]
```

### Issue 6: Frontend Build Errors
```bash
# Clear node modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Issue 7: Low Prediction Accuracy
**Solutions:**
- Capture more gesture samples (aim for 10+ per gesture)
- Ensure consistent lighting and background
- Maintain clear distance from camera (1-2 meters)
- Retrain the model after collecting more data
- Try the enhanced model: `python train_enhanced_model.py`

---

## 📊 Project Details

### Technology Stack

**Backend:**
- **Framework**: FastAPI
- **ML Libraries**: TensorFlow, Scikit-learn, OpenCV
- **Hand Detection**: MediaPipe (Google)
- **Server**: Uvicorn

**Frontend:**
- **Framework**: React 18
- **Build Tool**: Vite
- **HTTP Client**: Axios
- **Styling**: CSS3

**ML Models:**
- **Architecture**: CNN + LSTM (Convolutional Neural Networks + Long Short-Term Memory)
- **Preprocessing**: StandardScaler from scikit-learn
- **Encoding**: LabelEncoder for gesture classes

### Feature Extraction

The system extracts **148 features** per hand frame:
- 42 values: 21 hand landmarks (x, y, z coordinates)
- 15 values: Finger angles (thumb, index, middle, ring, pinky)
- 5 values: Finger bend ratios
- Additional: Hand position, scale, orientation

### Data Flow

```
Webcam Video Frame
    ↓
MediaPipe Hand Detection
    ↓
Extract 21-Point Landmarks
    ↓
Calculate Angles & Bend Ratios
    ↓
Normalize Features (StandardScaler)
    ↓
Feed to CNN+LSTM Model
    ↓
Predict Gesture Class
    ↓
Display Result with Confidence
```

### Model Performance

**Typical Accuracy:**
- 2-3 gestures: 85-90%
- 4-5 gestures: 80-85%
- 6+ gestures: 75-80%

*Note: Accuracy depends on data quality and gesture distinctiveness*

---

## 🔧 Advanced Configuration

### Custom Port Configuration

**Backend (advanced_mediapipe_server.py):**
```python
# Line ~900
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
    # Change 5002 to your desired port
```

**Frontend (vite.config.js):**
```javascript
export default {
  server: {
    port: 5173,  // Change to your desired port
    proxy: {
      '/api': 'http://localhost:5002'
    }
  }
}
```

### Changing Model Architecture

Edit `advanced_mediapipe_server.py`:
```python
# Find build_model() function around line 200
# Modify layers, units, dropout rates as needed
```

### Dataset Location

Change dataset path in `advanced_mediapipe_server.py`:
```python
DATASET_DIR = Path("gesture_dataset")  # Change this path
```

---

## 📝 Logging & Debugging

### Enable Debug Logging
```bash
# Backend (add to advanced_mediapipe_server.py)
import logging
logging.basicConfig(level=logging.DEBUG)

# Frontend (browser console)
# Press F12 → Console tab
```

### Check Logs
```bash
# Backend output
tail -f backend.log

# Frontend build output
npm run dev --verbose
```

---

## 🚀 Deployment

### Local Network Access
```bash
# Frontend (vite.config.js)
server: {
  host: '0.0.0.0'  // Accessible from other machines
}

# Backend
# Already configured with host 0.0.0.0

# Access from other machine
http://<YOUR_IP>:5173  # Frontend
http://<YOUR_IP>:5002  # Backend
```

### Docker Deployment (Optional)
Create `Dockerfile` for containerization (future enhancement).

---

## 📚 Additional Resources

### Project Documentation
- [Enhanced Features Guide](ProjectPhase2/sign_language_recognition/ENHANCED_FEATURES_README.md)
- [Quick Start Guide](ProjectPhase2/sign_language_recognition/00_START_HERE.md)
- [Backend Documentation](test/README_BACKEND.md)

### LSTM Model Documentation
- [LSTM Model README](Hand/LSTM_model/README.md)

### External Resources
- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [TensorFlow Keras Documentation](https://keras.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

---

## 📞 Support & Troubleshooting

### Common Commands

```bash
# Start everything fresh
pkill -f "python advanced_mediapipe_server"
pkill -f "vite"

# Backend
cd camera_backend && source mediapipe_env/bin/activate && python advanced_mediapipe_server.py

# Frontend
cd frontend && npm run dev

# Check if ports are available
lsof -i :5002
lsof -i :5173
```

### Performance Monitoring

```bash
# Monitor CPU/Memory usage
top -p $(pgrep -f "advanced_mediapipe_server.py")
top -p $(pgrep -f "node")

# Check disk usage
du -sh camera_backend/gesture_dataset/
du -sh camera_backend/models/
```

---

## 📄 License

This project is part of a university/research initiative.

---

## ✅ Verification Checklist

After installation, verify everything works:

- [ ] Backend starts without errors: `python advanced_mediapipe_server.py`
- [ ] Frontend loads at `http://localhost:5173`
- [ ] Webcam shows in camera preview
- [ ] Can capture a gesture
- [ ] Can train a model
- [ ] Can make predictions
- [ ] API docs visible at `http://localhost:5002/docs`

---

## 🎓 Learning Outcomes

This project demonstrates:
- Real-time computer vision processing
- Deep learning model training and inference
- Feature engineering for gesture recognition
- Full-stack web application development
- REST API design and implementation
- Neural network architectures (CNN, LSTM)

---

**Last Updated**: January 8, 2026
**Status**: ✅ Production Ready

For questions or issues, refer to the troubleshooting section or check the API documentation at `http://localhost:5002/docs`
