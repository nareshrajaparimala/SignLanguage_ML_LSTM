#!/usr/bin/env python3
"""
Advanced MediaPipe Hand Detection with Finger Bend Analysis and CNN+LSTM Model
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import base64
import math
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Initialize FastAPI
app = FastAPI(title="Advanced Sign Language Recognition API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variables
detector = None
camera = None
last_timestamp_ms = 0

DATASET_DIR = Path("gesture_dataset")
MODEL_DIR = Path("models")
DATASET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Gesture capture state
capture_state = {
    "active": False,
    "gesture_name": None,
    "gesture_text": None,
    "frames": [],
    "target_frames": 30
}

class GestureRequest(BaseModel):
    name: str
    text: str
    target_frames: int = 30

class PredictionRequest(BaseModel):
    frames: int = 30

def calculate_finger_angles(landmarks):
    """Calculate finger bend angles from hand landmarks"""
    
    def angle_between_points(p1, p2, p3):
        """Calculate angle between three points"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    finger_angles = {}
    
    # Thumb angle (CMC-MCP-IP)
    if len(landmarks) > 4:
        finger_angles['thumb'] = angle_between_points(
            landmarks[1], landmarks[2], landmarks[3]
        )
    
    # Index finger angle (MCP-PIP-DIP)
    if len(landmarks) > 8:
        finger_angles['index'] = angle_between_points(
            landmarks[5], landmarks[6], landmarks[7]
        )
    
    # Middle finger angle
    if len(landmarks) > 12:
        finger_angles['middle'] = angle_between_points(
            landmarks[9], landmarks[10], landmarks[11]
        )
    
    # Ring finger angle
    if len(landmarks) > 16:
        finger_angles['ring'] = angle_between_points(
            landmarks[13], landmarks[14], landmarks[15]
        )
    
    # Pinky angle
    if len(landmarks) > 20:
        finger_angles['pinky'] = angle_between_points(
            landmarks[17], landmarks[18], landmarks[19]
        )
    
    return finger_angles

def calculate_hand_rotation(landmarks):
    """Calculate wrist rotation (roll, pitch, yaw) and hand direction vector"""
    if len(landmarks) < 21:
        return None
    
    # Key points for rotation calculation
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]
    
    # Calculate hand plane normal vector
    v1 = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
    v2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
    normal = np.cross(v1, v2)
    
    # Normalize
    if np.linalg.norm(normal) > 0:
        normal = normal / np.linalg.norm(normal)
    
    # Calculate roll (rotation around z-axis)
    hand_vector = np.array([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, 0])
    if np.linalg.norm(hand_vector) > 0:
        hand_vector = hand_vector / np.linalg.norm(hand_vector)
        roll = np.arctan2(hand_vector[1], hand_vector[0]) * 180 / np.pi
    else:
        roll = 0
    
    # Calculate pitch (up/down tilt)
    pitch = np.arcsin(np.clip(normal[1], -1, 1)) * 180 / np.pi
    
    # Calculate yaw (left/right rotation)
    yaw = np.arctan2(normal[0], normal[2]) * 180 / np.pi
    
    # Hand direction vector (from wrist to middle finger)
    direction = np.array([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, middle_mcp.z - wrist.z])
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
    
    return {
        'roll': roll,
        'pitch': pitch, 
        'yaw': yaw,
        'direction_vector': direction.tolist(),
        'palm_normal': normal.tolist()
    }

def calculate_hand_velocity(current_landmarks, previous_landmarks, time_delta=0.1):
    """Calculate hand movement velocity and direction"""
    if not previous_landmarks or len(current_landmarks) < 21 or len(previous_landmarks) < 21:
        return None
    
    # Calculate wrist velocity
    wrist_curr = current_landmarks[0]
    wrist_prev = previous_landmarks[0]
    
    velocity = np.array([
        (wrist_curr.x - wrist_prev.x) / time_delta,
        (wrist_curr.y - wrist_prev.y) / time_delta,
        (wrist_curr.z - wrist_prev.z) / time_delta
    ])
    
    speed = np.linalg.norm(velocity)
    direction = velocity / speed if speed > 0 else np.array([0, 0, 0])
    
    return {
        'velocity': velocity.tolist(),
        'speed': speed,
        'direction': direction.tolist()
    }

def detect_finger_states(finger_angles):
    """Detect if fingers are bent or extended based on angles"""
    finger_states = {}
    
    # Thresholds for bent/extended (in degrees)
    bend_threshold = 160  # Less than this is considered bent
    
    for finger, angle in finger_angles.items():
        finger_states[finger] = {
            'angle': angle,
            'state': 'extended' if angle > bend_threshold else 'bent',
            'bend_ratio': max(0, (bend_threshold - angle) / bend_threshold)
        }
    
    return finger_states

# Global variable to store previous landmarks for velocity calculation
previous_landmarks = None

def extract_comprehensive_features(frame):
    """Extract comprehensive features including landmarks, angles, rotation, velocity, and handedness"""
    global previous_landmarks, detector, last_timestamp_ms
    
    if detector is None:
        print("❌ Detector not initialized")
        return [0.0] * 176, []
    
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Calculate timestamp (simulated for video mode)
    timestamp_ms = int(datetime.now().timestamp() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms
    
    # Detect
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    features = []
    handedness_info = []
    
    if detection_result.hand_landmarks:
        # Use first hand for velocity calculation if available
        current_landmarks = detection_result.hand_landmarks[0]
        
        for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Get handedness information
            handedness = "Unknown"
            handedness_confidence = 0.0
            
            if detection_result.handedness and hand_idx < len(detection_result.handedness):
                # Note: tasks API returns list of categories for each hand
                category = detection_result.handedness[hand_idx][0]
                handedness = category.category_name
                handedness_confidence = category.score
            
            handedness_info.append({
                "hand": handedness,
                "confidence": handedness_confidence
            })
            
            # Basic landmark coordinates (21 points * 3 = 63 features)
            landmark_coords = []
            for landmark in hand_landmarks:
                landmark_coords.extend([landmark.x, landmark.y, landmark.z])
            
            # Calculate finger angles
            finger_angles = calculate_finger_angles(hand_landmarks)
            
            # Detect finger states
            finger_states = detect_finger_states(finger_angles)
            
            # Extract angle features (5 fingers)
            angle_features = [finger_angles.get(finger, 0) for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']]
            
            # Extract bend ratio features (5 fingers)
            bend_features = [finger_states.get(finger, {}).get('bend_ratio', 0) for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']]
            
            # Hand rotation features (NEW)
            rotation_data = calculate_hand_rotation(hand_landmarks)
            if rotation_data:
                rotation_features = [
                    rotation_data['roll'],
                    rotation_data['pitch'],
                    rotation_data['yaw']
                ] + rotation_data['direction_vector'] + rotation_data['palm_normal']
            else:
                rotation_features = [0.0] * 9  # 3 rotations + 3 direction + 3 normal
            
            # Hand velocity features (NEW)
            velocity_data = calculate_hand_velocity(current_landmarks, previous_landmarks)
            if velocity_data:
                velocity_features = [
                    velocity_data['speed']
                ] + velocity_data['velocity'] + velocity_data['direction']
            else:
                velocity_features = [0.0] * 7  # 1 speed + 3 velocity + 3 direction
            
            # Hand shape and angle features (NEW)
            wrist = hand_landmarks[0]
            middle_mcp = hand_landmarks[9]
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]
            
            # Hand size
            hand_size = math.sqrt((middle_mcp.x - wrist.x)**2 + (middle_mcp.y - wrist.y)**2)
            
            # Hand angle relative to camera (NEW)
            hand_angle = math.atan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x) * 180 / math.pi
            
            # Thumb-index distance (important for many gestures)
            thumb_index_dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            
            # Hand orientation angle (palm facing direction)
            palm_angle = math.atan2(rotation_data['palm_normal'][1], rotation_data['palm_normal'][0]) * 180 / math.pi if rotation_data else 0
            
            # Wrist angle (important for orientation)
            wrist_angle = math.atan2(wrist.y - middle_mcp.y, wrist.x - middle_mcp.x) * 180 / math.pi
            
            # Handedness features (NEW)
            handedness_features = [
                1.0 if handedness == "Left" else 0.0,  # Left hand indicator
                1.0 if handedness == "Right" else 0.0,  # Right hand indicator
                handedness_confidence  # Handedness confidence
            ]
            
            shape_features = [hand_size, hand_angle, thumb_index_dist, palm_angle, wrist_angle]
            
            # Combine all features for this hand (85 + 3 = 88 features per hand)
            hand_features = (landmark_coords + angle_features + bend_features + 
                           rotation_features + velocity_features + shape_features + handedness_features)
            features.extend(hand_features)
        
        # Store current landmarks for next frame
        previous_landmarks = current_landmarks
    
    # Pad or truncate to fixed size (2 hands * 88 features = 176)
    target_size = 176
    while len(features) < target_size:
        features.append(0.0)
    
    # Return features and handedness info
    return features[:target_size], handedness_info

def create_cnn_lstm_model(input_shape, num_classes):
    """Create simplified CNN+LSTM model for small datasets"""
    model = Sequential([
        # Simplified CNN layers
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.2),
        
        # Simplified LSTM layers
        LSTM(64, return_sequences=True, dropout=0.2),
        LSTM(32, dropout=0.2),
        
        # Simplified dense layers
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Use lower learning rate for better convergence
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

@app.on_event("startup")
async def startup_event():
    global camera, detector
    try:
        # Initialize Camera
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            print("✅ Camera initialized successfully")
        else:
            print("⚠️ Warning: Camera not available")
            
        # Initialize Hand Landmarker
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        detector = HandLandmarker.create_from_options(options)
        print("✅ Hand Landmarker initialized successfully")
            
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        camera = None
        detector = None

@app.on_event("shutdown")
async def shutdown_event():
    global camera
    if camera:
        camera.release()

@app.get("/")
async def root():
    return {"message": "Advanced Sign Language Recognition API", "status": "running"}

@app.get("/api/status")
async def api_status():
    return {"status": "running", "message": "API is operational"}

@app.get("/api/list-gestures")
async def api_list_gestures():
    return await list_gestures()

@app.post("/api/start-gesture-capture")
async def api_start_gesture_capture(request: GestureRequest):
    return await start_gesture_capture(request)

@app.post("/api/capture-frame")
async def api_capture_frame():
    return await capture_frame()

@app.post("/api/save-gesture")
async def api_save_gesture():
    return await save_gesture()

@app.delete("/api/delete-gesture/{gesture_name}")
async def api_delete_gesture(gesture_name: str):
    return await delete_gesture(gesture_name)

@app.post("/api/predict-gesture")
async def api_predict_gesture(request: PredictionRequest):
    return await predict_gesture(request)

@app.post("/api/predict-live")
async def api_predict_live(request: PredictionRequest):
    return await predict_gesture(request)

@app.get("/api/hand-detection")
async def api_hand_detection():
    return await get_hand_detection()

@app.get("/api/finger-bend-data")
async def get_finger_bend_data():
    """Get detailed finger bend data for all gestures"""
    bend_data = []
    
    for gesture_dir in DATASET_DIR.iterdir():
        if not gesture_dir.is_dir():
            continue
        
        gesture_name = gesture_dir.name
        gesture_text = gesture_name
        
        # Read gesture text
        text_file = gesture_dir / "text.txt"
        if text_file.exists():
            with open(text_file, 'r') as f:
                gesture_text = f.read().strip()
        
        # Process each sequence
        for sequence_file in gesture_dir.glob("sequence_*.json"):
            try:
                with open(sequence_file, 'r') as f:
                    data = json.load(f)
                    
                if "frames" in data and data["frames"]:
                    # Analyze finger bends for each frame
                    frame_analysis = []
                    for frame_idx, frame_features in enumerate(data["frames"]):
                        # Extract finger angles from features (positions 63-67)
                        finger_angles = {
                            "thumb": frame_features[63] if len(frame_features) > 63 else 0,
                            "index": frame_features[64] if len(frame_features) > 64 else 0,
                            "middle": frame_features[65] if len(frame_features) > 65 else 0,
                            "ring": frame_features[66] if len(frame_features) > 66 else 0,
                            "pinky": frame_features[67] if len(frame_features) > 67 else 0
                        }
                        
                        # Extract bend ratios (positions 68-72)
                        bend_ratios = {
                            "thumb": frame_features[68] if len(frame_features) > 68 else 0,
                            "index": frame_features[69] if len(frame_features) > 69 else 0,
                            "middle": frame_features[70] if len(frame_features) > 70 else 0,
                            "ring": frame_features[71] if len(frame_features) > 71 else 0,
                            "pinky": frame_features[72] if len(frame_features) > 72 else 0
                        }
                        
                        # Determine finger states
                        finger_states = {}
                        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                            angle = finger_angles[finger]
                            finger_states[finger] = {
                                "angle": angle,
                                "bend_ratio": bend_ratios[finger],
                                "state": "extended" if angle > 160 else "bent"
                            }
                        
                        frame_analysis.append({
                            "frame": frame_idx,
                            "finger_angles": finger_angles,
                            "finger_states": finger_states
                        })
                    
                    bend_data.append({
                        "gesture_name": gesture_name,
                        "gesture_text": gesture_text,
                        "sequence_file": sequence_file.name,
                        "total_frames": len(data["frames"]),
                        "frame_analysis": frame_analysis
                    })
            except Exception:
                continue
    
    return {"finger_bend_data": bend_data}

@app.get("/hand-detection")
async def get_hand_detection():
    """Get real-time hand detection with finger bend analysis"""
    global camera, detector, last_timestamp_ms
    
    if not camera or not camera.isOpened():
        raise HTTPException(status_code=500, detail="Camera not available")
    
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    ret, frame = camera.read()
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to capture frame")
    
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Calculate timestamp
    timestamp_ms = int(datetime.now().timestamp() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms
    
    # Detect
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    annotated_frame = frame.copy()
    detailed_landmarks = []
    
    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Draw landmarks manually
            
            # Connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8), # Index
                (0, 9), (9, 10), (10, 11), (11, 12), # Middle
                (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
                (5, 9), (9, 13), (13, 17) # Knuckles
            ]
            
            for start_idx, end_idx in connections:
                start_point = hand_landmarks[start_idx]
                end_point = hand_landmarks[end_idx]
                x1 = int(start_point.x * width)
                y1 = int(start_point.y * height)
                x2 = int(end_point.x * width)
                y2 = int(end_point.y * height)
                cv2.line(annotated_frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
            
            # Keypoints
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(annotated_frame, (x, y), 4, (255, 0, 0), -1)
            
            # Calculate finger angles and states
            finger_angles = calculate_finger_angles(hand_landmarks)
            finger_states = detect_finger_states(finger_angles)
            
            # Calculate hand rotation and direction
            rotation_data = calculate_hand_rotation(hand_landmarks)
            
            # Calculate hand velocity
            velocity_data = calculate_hand_velocity(
                hand_landmarks, 
                previous_landmarks if idx == 0 else None
            )
            
            # Get handedness
            handedness = "Unknown"
            confidence = 0.0
            if detection_result.handedness and idx < len(detection_result.handedness):
                category = detection_result.handedness[idx][0]
                handedness = category.category_name
                confidence = category.score
            
            # Prepare landmark data
            landmarks_data = []
            for landmark_idx, landmark in enumerate(hand_landmarks):
                landmarks_data.append({
                    "id": landmark_idx,
                    "name": get_landmark_name(landmark_idx),
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                })
            
            # Add finger state visualization
            for finger, state in finger_states.items():
                color = (0, 255, 0) if state['state'] == 'extended' else (0, 0, 255)
                finger_tip_ids = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
                
                if finger in finger_tip_ids:
                    tip_landmark = hand_landmarks[finger_tip_ids[finger]]
                    x, y = int(tip_landmark.x * width), int(tip_landmark.y * height)
                    cv2.circle(annotated_frame, (x, y), 8, color, -1)
                    cv2.putText(annotated_frame, f"{finger}: {state['state']}", 
                               (x-30, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            detailed_landmarks.append({
                "handedness": handedness,
                "confidence": confidence,
                "landmarks": landmarks_data,
                "finger_angles": finger_angles,
                "finger_states": finger_states,
                "rotation": rotation_data,
                "velocity": velocity_data
            })
    
    # Add info overlay
    cv2.putText(annotated_frame, f'Hands: {len(detailed_landmarks)}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, 'Green=Extended, Red=Bent', (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "hand_count": len(detailed_landmarks),
        "annotated_frame": f"data:image/jpeg;base64,{frame_base64}",
        "detailed_landmarks": detailed_landmarks,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/start-gesture-capture")
async def start_gesture_capture(request: GestureRequest):
    """Start capturing frames for a new gesture"""
    global capture_state
    
    capture_state = {
        "active": True,
        "gesture_name": request.name,
        "gesture_text": request.text,
        "frames": [],
        "target_frames": request.target_frames
    }
    
    # Create gesture directory
    gesture_dir = DATASET_DIR / request.name
    gesture_dir.mkdir(exist_ok=True)
    
    # Save gesture text
    with open(gesture_dir / "text.txt", 'w') as f:
        f.write(request.text)
    
    return {
        "message": f"Started capturing gesture: {request.name}",
        "target_frames": request.target_frames
    }

@app.post("/capture-frame")
async def capture_frame():
    """Capture a single frame for the active gesture"""
    global camera, capture_state
    
    if not capture_state["active"]:
        raise HTTPException(status_code=400, detail="No active capture session")
    
    if not camera or not camera.isOpened():
        raise HTTPException(status_code=500, detail="Camera not available")
    
    ret, frame = camera.read()
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to capture frame")
    
    frame = cv2.flip(frame, 1)
    features, handedness_info = extract_comprehensive_features(frame)
    capture_state["frames"].append(features)
    
    # Store handedness info for this capture session
    if "handedness_data" not in capture_state:
        capture_state["handedness_data"] = []
    capture_state["handedness_data"].append(handedness_info)
    
    frames_captured = len(capture_state["frames"])
    is_complete = frames_captured >= capture_state["target_frames"]
    
    return {
        "frames_captured": frames_captured,
        "target_frames": capture_state["target_frames"],
        "remaining": max(0, capture_state["target_frames"] - frames_captured),
        "complete": is_complete
    }

@app.post("/save-gesture")
async def save_gesture():
    """Save the captured gesture sequence"""
    global capture_state
    
    if not capture_state["active"]:
        raise HTTPException(status_code=400, detail="No active capture session")
    
    if len(capture_state["frames"]) < capture_state["target_frames"]:
        raise HTTPException(status_code=400, detail=f"Need {capture_state['target_frames']} frames")
    
    gesture_dir = DATASET_DIR / capture_state["gesture_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence_file = gesture_dir / f"sequence_{timestamp}.json"
    
    # Analyze handedness for this gesture
    handedness_summary = {"left_count": 0, "right_count": 0, "both_count": 0}
    if "handedness_data" in capture_state:
        for frame_hands in capture_state["handedness_data"]:
            left_detected = any(h["hand"] == "Left" for h in frame_hands)
            right_detected = any(h["hand"] == "Right" for h in frame_hands)
            
            if left_detected and right_detected:
                handedness_summary["both_count"] += 1
            elif left_detected:
                handedness_summary["left_count"] += 1
            elif right_detected:
                handedness_summary["right_count"] += 1
    
    sequence_data = {
        "gesture_name": capture_state["gesture_name"],
        "gesture_text": capture_state["gesture_text"],
        "frames": capture_state["frames"][:capture_state["target_frames"]],
        "handedness_data": capture_state.get("handedness_data", [])[:capture_state["target_frames"]],
        "handedness_summary": handedness_summary,
        "timestamp": timestamp,
        "feature_count": len(capture_state["frames"][0]) if capture_state["frames"] else 0
    }
    
    with open(sequence_file, 'w') as f:
        json.dump(sequence_data, f)
    
    # Reset capture state
    capture_state = {"active": False, "gesture_name": None, "gesture_text": None, "frames": [], "handedness_data": [], "target_frames": 30}
    
    return {"message": "Gesture saved successfully", "file": str(sequence_file)}

@app.get("/list-gestures")
async def list_gestures():
    """List all available gestures with handedness information"""
    gestures = []
    
    for gesture_dir in DATASET_DIR.iterdir():
        if not gesture_dir.is_dir():
            continue
        
        gesture_name = gesture_dir.name
        gesture_text = gesture_name
        
        # Read gesture text
        text_file = gesture_dir / "text.txt"
        if text_file.exists():
            with open(text_file, 'r') as f:
                gesture_text = f.read().strip()
        
        # Analyze handedness across all sequences
        sequence_files = list(gesture_dir.glob("sequence_*.json"))
        sequence_count = len(sequence_files)
        
        handedness_stats = {"left_only": 0, "right_only": 0, "both_hands": 0, "unknown": 0}
        
        for seq_file in sequence_files:
            try:
                with open(seq_file, 'r') as f:
                    data = json.load(f)
                    if "handedness_summary" in data:
                        summary = data["handedness_summary"]
                        if summary["both_count"] > summary["left_count"] and summary["both_count"] > summary["right_count"]:
                            handedness_stats["both_hands"] += 1
                        elif summary["left_count"] > summary["right_count"]:
                            handedness_stats["left_only"] += 1
                        elif summary["right_count"] > summary["left_count"]:
                            handedness_stats["right_only"] += 1
                        else:
                            handedness_stats["unknown"] += 1
                    else:
                        handedness_stats["unknown"] += 1
            except:
                handedness_stats["unknown"] += 1
        
        gestures.append({
            "name": gesture_name,
            "text": gesture_text,
            "sequences": sequence_count,
            "handedness_stats": handedness_stats
        })
    
    return {"gestures": gestures}

@app.delete("/delete-gesture/{gesture_name}")
async def delete_gesture(gesture_name: str):
    """Delete a gesture and all its sequences"""
    from urllib.parse import unquote
    
    # URL decode the gesture name
    decoded_name = unquote(gesture_name)
    
    # Try exact match first
    gesture_dir = DATASET_DIR / decoded_name
    
    # If not found, try to find by partial match
    if not gesture_dir.exists():
        for existing_dir in DATASET_DIR.iterdir():
            if existing_dir.is_dir() and existing_dir.name.strip() == decoded_name.strip():
                gesture_dir = existing_dir
                break
    
    if not gesture_dir.exists():
        # List available gestures for debugging
        available = [d.name for d in DATASET_DIR.iterdir() if d.is_dir()]
        raise HTTPException(
            status_code=404, 
            detail=f"Gesture '{decoded_name}' not found. Available: {available}"
        )
    
    import shutil
    shutil.rmtree(gesture_dir)
    
    return {"message": f"Gesture '{gesture_dir.name}' deleted successfully"}

@app.post("/train-model")
@app.post("/api/train-model")
async def train_model():
    """Train enhanced CNN+LSTM model with rotation and velocity features"""
    print("🚀 Starting enhanced model training...")
    try:
        from train_enhanced_model import train_enhanced_model
        
        # Train the enhanced model
        model, label_encoder, training_info = train_enhanced_model(
            dataset_dir=str(DATASET_DIR),
            model_dir=str(MODEL_DIR)
        )
        
        result = {
            "message": f"Enhanced model trained successfully with {training_info['num_samples']} samples",
            "model_type": training_info["model_type"],
            "classes": training_info["classes"],
            "samples": training_info["num_samples"],
            "accuracy": training_info["final_accuracy"],
            "val_accuracy": training_info["final_val_accuracy"],
            "features": training_info["features_included"],
            "epochs": training_info["epochs_trained"]
        }
        print(f"✅ Enhanced training completed: {result}")
        return result
    
    except HTTPException as e:
        print(f"❌ HTTP Exception: {e.detail}")
        raise
    except Exception as e:
        print(f"❌ Enhanced training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced training failed: {str(e)}")

@app.post("/predict-gesture")
async def predict_gesture(request: PredictionRequest):
    """Predict gesture using enhanced model with rotation and velocity features"""
    global camera, detector, last_timestamp_ms
    
    if not camera or not camera.isOpened():
        raise HTTPException(status_code=500, detail="Camera not available")
        
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    # Check for hands in current frame first
    ret, test_frame = camera.read()
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to capture frame")
    
    test_frame = cv2.flip(test_frame, 1)
    rgb_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Calculate timestamp
    timestamp_ms = int(datetime.now().timestamp() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms
    
    results = detector.detect_for_video(mp_image, timestamp_ms)
    
    # Return "no hands" if no hands detected
    if not results.hand_landmarks:
        return {
            "gesture": "No hands detected",
            "text": "",
            "confidence": 0.0,
            "all_predictions": {},
            "status": "no_hands",
            "model_type": "none"
        }
    
    # Check if enhanced model exists
    enhanced_model_path = MODEL_DIR / "enhanced_gesture_model.h5"
    enhanced_scaler_path = MODEL_DIR / "enhanced_scaler.joblib"
    enhanced_encoder_path = MODEL_DIR / "enhanced_label_encoder.joblib"
    
    # Fallback to basic model if enhanced model doesn't exist
    if enhanced_model_path.exists() and enhanced_scaler_path.exists() and enhanced_encoder_path.exists():
        from enhanced_model import EnhancedSignLanguageModel
        model = EnhancedSignLanguageModel()
        model.load_model(str(enhanced_model_path), str(enhanced_scaler_path))
        label_encoder = joblib.load(enhanced_encoder_path)
        is_enhanced = True
    else:
        model_path = MODEL_DIR / "gesture_model.h5"
        encoder_path = MODEL_DIR / "label_encoder.joblib"
        if not model_path.exists() or not encoder_path.exists():
            raise HTTPException(status_code=400, detail="No trained model found")
        model = tf.keras.models.load_model(model_path)
        label_encoder = joblib.load(encoder_path)
        is_enhanced = False
    
    # Capture sequence
    sequence = []
    hands_detected_count = 0
    
    for _ in range(request.frames):
        ret, frame = camera.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        
        # Check if hands are present in this frame
        rgb_check = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_check = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_check)
        
        # Increment timestamp for loop
        last_timestamp_ms += 33 # approx 30fps
        check_results = detector.detect_for_video(mp_check, last_timestamp_ms)
        
        if check_results.hand_landmarks:
            hands_detected_count += 1
        
        features, _ = extract_comprehensive_features(frame)
        sequence.append(features)
    
    # Require hands in at least 70% of frames
    if hands_detected_count < (request.frames * 0.7):
        return {
            "gesture": "Insufficient hand detection",
            "text": "",
            "confidence": 0.0,
            "all_predictions": {},
            "status": "insufficient_hands",
            "model_type": "enhanced" if is_enhanced else "basic"
        }
    
    # Predict
    X = np.array([sequence])
    
    if is_enhanced and hasattr(model, 'predict'):
        predictions = model.predict(X)
    else:
        predictions = model.predict(X)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    # Apply confidence threshold
    num_classes = len(label_encoder.classes_)
    min_confidence = max(0.3, 1.0 / num_classes + 0.1)
    
    if confidence < min_confidence:
        return {
            "gesture": "Low confidence",
            "text": "",
            "confidence": confidence,
            "all_predictions": {
                label_encoder.inverse_transform([i])[0]: float(predictions[0][i])
                for i in range(len(predictions[0]))
            },
            "status": "low_confidence",
            "model_type": "enhanced" if is_enhanced else "basic"
        }
    
    gesture_name = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    # Get gesture text
    gesture_text = gesture_name
    text_file = DATASET_DIR / gesture_name / "text.txt"
    if text_file.exists():
        with open(text_file, 'r') as f:
            gesture_text = f.read().strip()
    
    return {
        "gesture": gesture_name,
        "text": gesture_text,
        "confidence": confidence,
        "all_predictions": {
            label_encoder.inverse_transform([i])[0]: float(predictions[0][i])
            for i in range(len(predictions[0]))
        },
        "status": "recognized",
        "model_type": "enhanced" if is_enhanced else "basic",
        "auto_speak": True  # Enable auto-speak for recognized gestures
    }

def get_landmark_name(landmark_id):
    """Get landmark name by ID"""
    landmark_names = {
        0: "WRIST", 1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
        5: "INDEX_FINGER_MCP", 6: "INDEX_FINGER_PIP", 7: "INDEX_FINGER_DIP", 8: "INDEX_FINGER_TIP",
        9: "MIDDLE_FINGER_MCP", 10: "MIDDLE_FINGER_PIP", 11: "MIDDLE_FINGER_DIP", 12: "MIDDLE_FINGER_TIP",
        13: "RING_FINGER_MCP", 14: "RING_FINGER_PIP", 15: "RING_FINGER_DIP", 16: "RING_FINGER_TIP",
        17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP"
    }
    return landmark_names.get(landmark_id, f"LANDMARK_{landmark_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)