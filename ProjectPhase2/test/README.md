# Complete Report: Raspberry-Pi Sign Language Recognition System

## Overview
This project implements a full pipeline for sign language recognition on a Raspberry Pi, integrating Arduino sensor data with machine learning for real-time gesture classification. The system reads flex sensor and MPU-6050 accelerometer/gyroscope data, trains a KNN classifier, and exposes a REST API for predictions that can be consumed by a MERN frontend.

## Project Structure
The `test/` folder contains three main Python files:

### 1. `serial_reader.py`
- **Purpose**: Reads and parses Arduino serial data stream
- **Features**:
  - Connects to `/dev/ttyUSB0` (or `/dev/ttyACM0`) at 9600 baud
  - Parses lines in format: `F1:694 F2:1023 F3:266 F4:539 F5:426 | AX:-4792 AY:720 AZ:16156 | GX:-177 GY:59 GZ:12`
  - Displays live sensor data in console
  - Optional CSV logging with timestamps
  - Interactive labeling (commented out by default; use API instead)

### 2. `trainer_and_model.py`
- **Purpose**: Handles model training and prediction
- **Features**:
  - Loads labeled CSV data (format: timestamp,F1..F5,AX..GZ,label)
  - Trains KNN classifier using scikit-learn (falls back to numpy if unavailable)
  - Standardizes features (mean/std scaling) for better distance metrics
  - Saves model to `model.joblib` and scaler to `scaler.joblib`
  - Provides prediction function with confidence scores

### 3. `server.py`
- **Purpose**: Flask REST API server
- **Endpoints**:
  - `POST /predict`: Accepts sensor data JSON, returns predicted label + confidence
  - `POST /add_sample`: Adds labeled sample to CSV for training
  - `POST /train`: Retrains model from current CSV data
  - `GET /status`: Returns model status (trained/untrained, type)
- **Features**: Runs on `0.0.0.0:5000` for network access

## Setup Instructions

### 1. Environment Setup
```bash
cd test
python3 -m venv venv
source venv/bin/activate
pip install pyserial flask numpy pandas joblib scikit-learn
```

### 2. Hardware Requirements
- Raspberry Pi (any model with USB/serial)
- Arduino connected via USB (port `/dev/ttyUSB0` or `/dev/ttyACM0`)
- Arduino sketch sending data in specified format

### 3. Initial Run Order
1. Start the API server: `python3 server.py`
2. Monitor serial data: `python3 serial_reader.py --port /dev/ttyUSB0`
3. Collect labeled samples via API calls or CSV
4. Train model: `python3 trainer_and_model.py --csv samples_labeled.csv`
5. Use API for predictions

## Usage Workflow

### Data Collection
- **Option A: API-based collection**
  - Run `server.py`
  - POST to `/add_sample` with sensor data + label
  - Example: `curl -X POST http://localhost:5000/add_sample -H "Content-Type: application/json" -d '{"F1":694,"F2":1023,"F3":266,"F4":539,"F5":426,"AX":-4792,"AY":720,"AZ":16156,"GX":-177,"GY":59,"GZ":12,"label":"HELLO"}'`

- **Option B: CSV collection**
  - Run `serial_reader.py --out samples.csv`
  - Manually add labels to CSV
  - Use `trainer_and_model.py --csv samples.csv`

### Training
```bash
# Via CLI
python3 trainer_and_model.py --csv samples_labeled.csv --k 3

# Via API
curl -X POST http://localhost:5000/train
```

### Prediction
```bash
# Via API
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"F1":694,"F2":1023,"F3":266,"F4":539,"F5":426,"AX":-4792,"AY":720,"AZ":16156,"GX":-177,"GY":59,"GZ":12}'
```

### MERN Frontend Integration
```javascript
// Predict sample
async function predictSample(sample) {
  const res = await fetch("http://RASPBERRY_PI_IP:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sample)
  });
  return await res.json();
}

// Add labeled sample
async function addSample(sample, label) {
  const data = { ...sample, label };
  const res = await fetch("http://RASPBERRY_PI_IP:5000/add_sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  return await res.json();
}
```

## Key Features & Benefits

### Robustness
- **Fallback classifier**: Uses numpy nearest neighbor if scikit-learn unavailable
- **Feature scaling**: Standardizes sensor ranges for accurate distance calculations
- **Error handling**: Graceful failures with informative error messages

### Performance
- **Low latency**: KNN is fast for small datasets
- **Pi-compatible**: Minimal dependencies, works on resource-constrained devices
- **Real-time capable**: Processes samples quickly for live predictions

### Extensibility
- **API-driven**: Easy integration with web/mobile frontends
- **Modular design**: Separate concerns for reading, training, serving
- **Configurable**: Adjustable KNN neighbors, ports, CSV paths

## Tips & Best Practices

### Data Collection
- Collect 50-100 samples per gesture from different angles/speeds
- Balance classes to avoid bias
- Use consistent labeling (e.g., uppercase words)

### Model Training
- Start with k=3 neighbors
- Retrain periodically as you add more data
- Monitor confidence scores for prediction reliability

### Production Deployment
- Run server with `gunicorn` for production
- Add authentication/CORS for network security
- Implement logging for monitoring

### Troubleshooting
- Check serial port permissions: `sudo usermod -a -G dialout $USER`
- Verify Arduino data format matches regex pattern
- Use `/status` endpoint to confirm model loading

This system provides a complete, production-ready pipeline for gesture recognition, from sensor data to web API predictions.
