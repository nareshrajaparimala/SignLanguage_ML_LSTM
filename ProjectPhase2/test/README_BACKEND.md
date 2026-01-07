# 🤖 Raspberry Pi Sign Language Recognition - Python Backend

Core ML backend for real-time gesture recognition. Runs on Raspberry Pi, integrates Arduino sensors with KNN classifier and FastAPI REST API.

## 🎯 Quick Start

### 1. Setup

```bash
cd /home/naresh/test

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run FastAPI Server

```bash
# Start the server (connects to Arduino on /dev/ttyUSB0)
uvicorn api_server:app --host 0.0.0.0 --port 5000 --reload
```

Server will be accessible at:
- **FastAPI Docs**: http://localhost:5000/docs
- **API Base**: http://localhost:5000

### 3. Connect Frontend

The Express.js proxy server (in `sign_language_recognition/backend/`) proxies all requests to this FastAPI backend.

## 📁 Python Files

### `api_server.py` (NEW - Main Backend)
FastAPI server with frame buffering, model training, and prediction.

**Key Features:**
- Real-time serial reader thread
- 30-frame gesture buffering
- KNN model training and prediction
- Statistical feature extraction (mean + std dev)
- CORS enabled for MERN frontend

**Endpoints:**
- `GET /capture-frame` - Get latest sensor frame
- `POST /buffer-frame` - Add frame to buffer
- `GET /buffer-status` - Check buffer progress
- `POST /save-label` - Save buffered frames as gesture
- `POST /train-model` - Train KNN classifier
- `GET /list-labels` - List trained gestures
- `POST /predict-live` - Predict gesture from frames
- `GET /status` - System status

### `serial_reader.py`
Standalone serial reader for debugging Arduino connection.

```bash
python3 serial_reader.py --port /dev/ttyUSB0 --baud 9600
```

### `trainer_and_model.py`
ML model utilities (used by api_server.py internally).

**Features:**
- Load training data from dataset/
- Extract statistical features (22 per gesture)
- Train/predict with KNN classifier
- Save/load trained models

### `server.py`
Alternative Flask backend (legacy, use api_server.py for new projects).

### `collect_flex.py`
Original serial collection script (for reference).

## 📊 Data Structure

```
/home/naresh/test/
├── api_server.py           # FastAPI main server
├── serial_reader.py        # Arduino reader utility
├── trainer_and_model.py    # ML model logic
├── requirements.txt        # Python dependencies
├── dataset/                # Gesture data (created at runtime)
│   ├── HELLO/
│   │   └── frames.json     # Array of 30 frames
│   ├── GOODBYE/
│   │   └── frames.json
│   └── ... (more gestures)
├── models/                 # Trained models (created at runtime)
│   └── model.pkl          # Scikit-learn KNN model + scaler
└── collect_flex.py        # Legacy serial script
```

## 🔌 Arduino Data Format

Expected format from Arduino (one line per ~30ms):

```
F1:694 F2:1023 F3:266 F4:539 F5:426 | AX:-4792 AY:720 AZ:16156 | GX:-177 GY:59 GZ:12
```

- **F1-F5**: Flex sensor analog readings (0-1023)
- **AX, AY, AZ**: Accelerometer (raw int16)
- **GX, GY, GZ**: Gyroscope (raw int16)

## 🧠 ML Pipeline

### Feature Extraction

Per 30-frame gesture:
```
Raw Features (per frame): F1, F2, F3, F4, F5, AX, AY, AZ, GX, GY, GZ = 11 values
Statistical Features: Mean(11) + StdDev(11) = 22 values per gesture
```

### Training

```python
# Load all gestures
for gesture in dataset/:
    frames = load(gesture/frames.json)  # 30 frames
    features = extract_statistics(frames)  # 22 features
    X.append(features)
    y.append(gesture_label)

# Train KNN with k=3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_scaled, y)
model.save('models/model.pkl')
```

### Prediction

```python
# Capture 30 frames
frames = capture_30_frames()
features = extract_statistics(frames)  # 22 features

# Predict
label = model.predict(features_scaled)
confidence = 1.0 / (1.0 + average_distance_to_neighbors)
```

## 🔌 Serial Port Configuration

Default: `/dev/ttyUSB0` (CH340 Arduino)

To change:
```python
# In api_server.py
SERIAL_PORT = "/dev/ttyACM0"  # Arduino Original
SERIAL_PORT = "/dev/ttyUSB0"  # Arduino with CH340
```

Check available ports:
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

## 📡 API Usage Examples

### Capture Single Frame
```bash
curl http://localhost:5000/capture-frame
# Returns: {"F1": 694, "F2": 1023, ..., "GZ": 12}
```

### Buffer Frames (for 30-frame capture)
```bash
curl -X POST http://localhost:5000/buffer-frame \
  -H "Content-Type: application/json" \
  -d '{"F1": 694, "F2": 1023, ...}'
```

### Save Gesture Label
```bash
curl -X POST "http://localhost:5000/save-label?label=HELLO"
# Saves 30 buffered frames to dataset/HELLO/frames.json
```

### Train Model
```bash
curl -X POST "http://localhost:5000/train-model?k=3"
# Trains KNN on all gestures in dataset/
# Returns: {"status": "trained", "gestures": [...]}
```

### Predict Gesture
```bash
curl -X POST http://localhost:5000/predict-live \
  -H "Content-Type: application/json" \
  -d '{
    "frames": [
      {"F1": 694, "F2": 1023, ..., "GZ": 12},
      ...
      (30 frames total)
    ]
  }'
# Returns: {"label": "HELLO", "confidence": 0.95}
```

### List Trained Gestures
```bash
curl http://localhost:5000/list-labels
# Returns: {"labels": ["HELLO", "GOODBYE"], "count": 2}
```

### System Status
```bash
curl http://localhost:5000/status
# Returns: {
#   "serial_connected": true,
#   "model_trained": true,
#   "gestures": ["HELLO", "GOODBYE"],
#   "buffer_size": 0,
#   "dataset_path": "dataset"
# }
```

## 🛠️ Troubleshooting

### Arduino Not Connected

```bash
# Check if Arduino appears
ls /dev/ttyUSB* /dev/ttyACM*

# Check permissions
sudo usermod -a -G dialout $USER
# Log out and back in

# Test with serial reader
python3 serial_reader.py --port /dev/ttyUSB0

# Watch /status endpoint
curl http://localhost:5000/status
# Should show "serial_connected": true
```

### Model Training Fails

```bash
# Check gestures exist
ls -la dataset/

# Ensure at least 2 gestures with 30 frames each
python3 -c "
import json, os
for g in os.listdir('dataset'):
    f = json.load(open(f'dataset/{g}/frames.json'))
    print(f'{g}: {len(f)} frames')
"

# View model file
ls -la models/model.pkl
```

### Low Prediction Accuracy

**Solutions:**
1. Collect more samples (5-10 per gesture, not just 1)
2. Ensure consistent sensor calibration
3. Try different k values: k=1 (sensitive), k=5 (robust)
4. Check Arduino data format matches expected pattern
5. Retrain model after collecting new samples

```bash
# Retrain
curl -X POST http://localhost:5000/train-model?k=5
```

### CORS Errors

CORS is already enabled in `api_server.py` for all origins.

If still issues, ensure:
- FastAPI running on port 5000
- Express proxy running on port 3001
- Frontend configured with correct API URLs

## 📚 Advanced Usage

### Monitor Serial Data in Real-Time
```bash
python3 serial_reader.py --port /dev/ttyUSB0
```

### Extract Features Programmatically
```python
from api_server import GestureModel

model = GestureModel()
frames = [...]  # 30 frames
features = model.extract_features_from_batch(frames)
print(features.shape)  # Should be (22,)
```

### Use Different KNN Parameters
```bash
# Use k=5 for more robust predictions
curl -X POST "http://localhost:5000/train-model?k=5"

# Use k=1 for exact matching
curl -X POST "http://localhost:5000/train-model?k=1"
```

### Deploy on Remote Network

```bash
# On Raspberry Pi, ensure binding to 0.0.0.0
uvicorn api_server:app --host 0.0.0.0 --port 5000

# From another machine on network
curl http://RASPBERRY_PI_IP:5000/status
```

## 📝 Dependencies

See `requirements.txt`:
- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **pyserial**: Arduino communication
- **numpy**: Numerical operations
- **pandas**: Data handling
- **scikit-learn**: KNN classifier
- **joblib**: Model serialization

## 🔒 Security Notes

- **No authentication**: Suitable for LAN only
- **Open CORS**: For development/learning
- **Local network**: Run on 0.0.0.0:5000 for network access

For production:
1. Add API key authentication
2. Use HTTPS/SSL
3. Implement rate limiting
4. Add input validation
5. Run behind reverse proxy (nginx, Apache)

## 📖 API Documentation

Interactive docs available at:
- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

## 🔄 Integration with Frontend

The MERN frontend (in `/home/naresh/sign_language_recognition/`) uses Express proxy to:
1. Route `/api/*` requests to this FastAPI backend
2. Add middleware/auth if needed
3. Handle frontend-specific logic

Frontend calls Express, Express proxies to FastAPI:
```
React → Express (3001) → FastAPI (5000) → Arduino
```

## 📊 Performance

- **Capture**: 30 frames in ~1 second (30ms per frame)
- **Training**: <100ms (KNN is lazy learner)
- **Prediction**: <50ms per gesture
- **Memory**: 20-50 MB (varies with dataset size)

## 🎓 Learning Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [PySerial](https://pyserial.readthedocs.io/)
- [Raspberry Pi Projects](https://www.raspberrypi.com/documentation/)

---

**Next Steps:**
1. Ensure Arduino connected and sending data
2. Run `uvicorn api_server:app --host 0.0.0.0 --port 5000`
3. Start capturing gestures via frontend or API
4. Train model when you have 2+ gestures
5. Run predictions!
