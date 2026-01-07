"""
FastAPI server for real-time sign language recognition.

Provides endpoints for:
- Capturing sensor frames
- Saving labeled gesture batches (30 frames per gesture)
- Training KNN model on collected gestures
- Real-time prediction

Run: uvicorn api_server:app --host 0.0.0.0 --port 5000 --reload
"""
import os
import json
import time
from datetime import datetime
from collections import defaultdict, deque
import threading
try:
    import numpy as np
except Exception:
    np = None
    print("[Startup] WARNING: numpy not available — numeric features and model endpoints will be disabled until numpy is installed")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import serial
import re

# ============================================================================
# CONFIG
# ============================================================================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 9600
DATASET_DIR = "dataset"
MODELS_DIR = "models"
FRAME_BUFFER_SIZE = 30  # Capture 30 frames per gesture

# Line pattern matching
LINE_PATTERN = re.compile(
    r"F1:(?P<F1>-?\d+)\s+F2:(?P<F2>-?\d+)\s+F3:(?P<F3>-?\d+)\s+F4:(?P<F4>-?\d+)\s+F5:(?P<F5>-?\d+)\s*\|\s*"
    r"AX:(?P<AX>-?\d+)\s+AY:(?P<AY>-?\d+)\s+AZ:(?P<AZ>-?\d+)\s*\|\s*"
    r"GX:(?P<GX>-?\d+)\s+GY:(?P<GY>-?\d+)\s+GZ:(?P<GZ>-?\d+)"
)

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# SERIAL READER THREAD
# ============================================================================
class SerialReader:
    """Continuously read from Arduino and buffer latest frame."""
    
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self.current_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
    def parse_line(self, line):
        """Parse a single serial line into a dict of ints."""
        m = LINE_PATTERN.search(line)
        if not m:
            return None
        return {k: int(v) for k, v in m.groupdict().items()}
    
    def start(self):
        """Start the reader thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print(f"[SerialReader] Started on {self.port} @ {self.baud} baud")
    
    def stop(self):
        """Stop the reader thread."""
        self.running = False
        if self.ser:
            try:
                self.ser.close()
            except:
                pass
    
    def _read_loop(self):
        """Main read loop (runs in background thread)."""
        retries = 0
        while self.running and retries < 5:
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=1)
                time.sleep(2)  # Arduino reset
                print(f"[SerialReader] Connected to {self.port}")
                retries = 0
                
                while self.running:
                    raw = self.ser.readline().decode(errors="ignore").strip()
                    if not raw:
                        continue
                    parsed = self.parse_line(raw)
                    if parsed:
                        with self.lock:
                            self.current_frame = parsed
            except serial.SerialException as e:
                retries += 1
                print(f"[SerialReader] Serial error (retry {retries}): {e}")
                time.sleep(2)
            except Exception as e:
                print(f"[SerialReader] Unexpected error: {e}")
                retries += 1
                time.sleep(2)
    
    def get_frame(self):
        """Get the latest frame."""
        with self.lock:
            return self.current_frame.copy() if self.current_frame else None


# ============================================================================
# MODEL / TRAINER
# ============================================================================
class GestureModel:
    """Train and predict gestures from 30-frame batches."""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.model = None  # Will hold trained model
        self.scaler = None
        self.labels = []
        
    def extract_features_from_batch(self, frames: List[Dict]) -> np.ndarray:
        """
        Convert a batch of 30 frames to a single feature vector.
        
        For each frame: [F1, F2, F3, F4, F5, AX, AY, AZ, GX, GY, GZ] (11 features)
        Compute mean and std -> 22 features total
        """
        if np is None:
            raise RuntimeError("NumPy is required for feature extraction. Install numpy in your Python environment.")

        feat_cols = ["F1", "F2", "F3", "F4", "F5", "AX", "AY", "AZ", "GX", "GY", "GZ"]

        # Convert frames to matrix
        X = np.array([[f[col] for col in feat_cols] for f in frames], dtype=float)

        # Compute mean and std per feature
        mean_vec = X.mean(axis=0)  # (11,)
        std_vec = X.std(axis=0, ddof=0)  # (11,)
        std_vec[std_vec == 0] = 1.0

        # Concatenate: [mean1..mean11, std1..std11]
        features = np.concatenate([mean_vec, std_vec])
        return features  # (22,)
    
    def load_all_gestures(self) -> tuple:
        """
        Load all gesture data from dataset folder.
        Returns (X, y) where X is features and y is labels.
        """
        X_list = []
        y_list = []
        
        for gesture_label in os.listdir(self.models_dir if not os.path.exists(DATASET_DIR) else DATASET_DIR):
            folder = os.path.join(DATASET_DIR, gesture_label)
            if not os.path.isdir(folder):
                continue
            
            frames_file = os.path.join(folder, "frames.json")
            if not os.path.exists(frames_file):
                continue
            
            with open(frames_file, "r") as f:
                frames = json.load(f)
            
            if len(frames) >= FRAME_BUFFER_SIZE:
                # Use first 30 frames
                frames = frames[:FRAME_BUFFER_SIZE]
                features = self.extract_features_from_batch(frames)
                X_list.append(features)
                y_list.append(gesture_label)
        
        if not X_list:
            raise ValueError("No gesture data found in dataset folder")
        
        X = np.array(X_list)
        y = np.array(y_list)
        return X, y
    
    def train(self, k: int = 3):
        """Train KNN model from all gestures in dataset folder."""
        try:
            X, y = self.load_all_gestures()
            
            # Standardize features
            self.scaler = {
                "mu": X.mean(axis=0),
                "sigma": X.std(axis=0, ddof=0)
            }
            self.scaler["sigma"][self.scaler["sigma"] == 0] = 1.0
            
            X_scaled = (X - self.scaler["mu"]) / self.scaler["sigma"]
            
            # Try sklearn, fallback to numpy
            try:
                from sklearn.neighbors import KNeighborsClassifier
                self.model = KNeighborsClassifier(n_neighbors=k)
                self.model.fit(X_scaled, y)
                self.labels = list(self.model.classes_)
                
                # Save model
                import joblib
                joblib.dump({"model": self.model, "scaler": self.scaler}, 
                           os.path.join(self.models_dir, "model.pkl"))
                print(f"[Model] Trained sklearn KNN with {len(X)} samples, {len(self.labels)} gestures")
                return True
            except ImportError:
                # Fallback to numpy
                self.model = {"X": X_scaled, "y": y, "k": k}
                self.labels = list(np.unique(y))
                
                import joblib
                joblib.dump({"model": self.model, "scaler": self.scaler}, 
                           os.path.join(self.models_dir, "model.pkl"))
                print(f"[Model] Trained numpy KNN with {len(X)} samples, {len(self.labels)} gestures (sklearn not available)")
                return True
        except Exception as e:
            print(f"[Model] Training failed: {e}")
            raise
    
    def predict(self, frames: List[Dict]) -> Dict:
        """Predict gesture from a batch of frames."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if len(frames) < FRAME_BUFFER_SIZE:
            raise ValueError(f"Need at least {FRAME_BUFFER_SIZE} frames, got {len(frames)}")
        
        # Extract features
        features = self.extract_features_from_batch(frames[:FRAME_BUFFER_SIZE])
        
        # Scale
        X_scaled = (features - self.scaler["mu"]) / self.scaler["sigma"]
        
        # Predict
        try:
            from sklearn.neighbors import KNeighborsClassifier
            if isinstance(self.model, KNeighborsClassifier):
                pred = self.model.predict(X_scaled.reshape(1, -1))[0]
                dists, _ = self.model.kneighbors(X_scaled.reshape(1, -1), n_neighbors=self.model.n_neighbors)
                confidence = 1.0 / (1.0 + float(dists.mean()))
                return {"label": pred, "confidence": float(confidence)}
        except:
            pass
        
        # Numpy fallback
        X_train = self.model["X"]
        y_train = self.model["y"]
        diffs = X_train - X_scaled
        dists = np.linalg.norm(diffs, axis=1)
        best = np.argmin(dists)
        pred = y_train[best]
        confidence = 1.0 / (1.0 + float(dists[best]))
        return {"label": pred, "confidence": float(confidence)}


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="Sign Language Recognition API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
serial_reader = SerialReader(SERIAL_PORT, BAUD_RATE)
gesture_model = GestureModel(MODELS_DIR)
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

# Start serial reader
serial_reader.start()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class Frame(BaseModel):
    F1: int
    F2: int
    F3: int
    F4: int
    F5: int
    AX: int
    AY: int
    AZ: int
    GX: int
    GY: int
    GZ: int


class PredictionRequest(BaseModel):
    frames: List[Frame]


# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/")
def root():
    return {"status": "API running", "description": "Sign Language Recognition System"}


@app.get("/capture-frame")
def capture_frame():
    """Get the latest sensor frame from Arduino."""
    frame = serial_reader.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No sensor data available")
    return frame


@app.post("/buffer-frame")
def buffer_frame(frame: Frame):
    """Add a frame to the capture buffer."""
    frame_dict = frame.dict()
    frame_buffer.append(frame_dict)
    return {
        "buffered": len(frame_buffer),
        "total_needed": FRAME_BUFFER_SIZE
    }


@app.get("/buffer-status")
def buffer_status():
    """Get current buffer status."""
    return {
        "frames_collected": len(frame_buffer),
        "frames_needed": FRAME_BUFFER_SIZE,
        "ready": len(frame_buffer) >= FRAME_BUFFER_SIZE
    }


@app.post("/save-label")
def save_label(label: str):
    """Save the buffered frames under a gesture label."""
    if len(frame_buffer) < FRAME_BUFFER_SIZE:
        raise HTTPException(status_code=400, detail=f"Not enough frames: {len(frame_buffer)}/{FRAME_BUFFER_SIZE}")
    
    # Create label folder
    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Save frames as JSON
    frames_file = os.path.join(label_dir, "frames.json")
    with open(frames_file, "w") as f:
        json.dump(list(frame_buffer), f, indent=2)
    
    frame_buffer.clear()
    return {
        "status": "saved",
        "label": label,
        "frames_saved": FRAME_BUFFER_SIZE,
        "path": frames_file
    }


@app.post("/train-model")
def train_model(k: int = 3):
    """Train KNN model from all labeled gesture data."""
    try:
        gesture_model.train(k=k)
        return {
            "status": "trained",
            "gestures": gesture_model.labels,
            "num_gestures": len(gesture_model.labels)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-labels")
def list_labels():
    """List all trained gesture labels."""
    if not os.path.exists(DATASET_DIR):
        return {"labels": []}
    labels = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    return {"labels": labels, "count": len(labels)}


@app.post("/predict-live")
def predict_live(request: PredictionRequest):
    """Predict gesture from provided frames."""
    try:
        frames_list = [f.dict() for f in request.frames]
        result = gesture_model.predict(frames_list)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    """Get system status."""
    return {
        "serial_connected": serial_reader.ser is not None if serial_reader.ser else False,
        "model_trained": gesture_model.model is not None,
        "gestures": gesture_model.labels,
        "buffer_size": len(frame_buffer),
        "dataset_path": DATASET_DIR
    }


# ============================================================================
# SHUTDOWN
# ============================================================================
@app.on_event("shutdown")
def shutdown():
    serial_reader.stop()


if __name__ == "__main__":
    import uvicorn
    # Run FastAPI on port 5050 to avoid conflicts and match backend proxy
    uvicorn.run(app, host="0.0.0.0", port=5050, reload=False)
    
