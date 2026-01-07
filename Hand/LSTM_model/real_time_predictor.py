import serial
import numpy as np
import tensorflow as tf
from collections import deque
import re
import time
import pyttsx3

class GesturePredictor:
    def __init__(self, model_path='gesture_lstm_model.h5', port='/dev/cu.usbserial-A5069RR4'):
        # Load model and preprocessing
        self.model = tf.keras.models.load_model(model_path)
        scaler_params = np.load('scaler_params.npy')
        self.scaler_mean = scaler_params[0]
        self.scaler_scale = scaler_params[1]
        self.label_classes = np.load('label_encoder.npy', allow_pickle=True)
        
        # Serial connection
        self.ser = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)
        
        # Buffer for sequence data
        self.sequence_length = 50
        self.data_buffer = deque(maxlen=self.sequence_length)
        
        # Prediction stability
        self.last_predictions = deque(maxlen=5)
        self.last_gesture = None
        self.gesture_count = 0
        
        # Text-to-speech
        self.tts = pyttsx3.init()
        
        print("Gesture predictor initialized")
        print(f"Available gestures: {list(self.label_classes)}")
    
    def parse_sensor_data(self, line):
        flex_pattern = r'F(\d):(\d+)'
        accel_pattern = r'A([XYZ]):(-?\d+)'
        gyro_pattern = r'G([XYZ]):(-?\d+)'
        
        flex_data = {}
        accel_data = {}
        gyro_data = {}
        
        for match in re.finditer(flex_pattern, line):
            flex_data[f'F{match.group(1)}'] = int(match.group(2))
        
        for match in re.finditer(accel_pattern, line):
            accel_data[f'A{match.group(1)}'] = int(match.group(2))
        
        for match in re.finditer(gyro_pattern, line):
            gyro_data[f'G{match.group(1)}'] = int(match.group(2))
        
        if len(flex_data) == 5 and len(accel_data) == 3 and len(gyro_data) == 3:
            return [flex_data.get('F1', 0), flex_data.get('F2', 0), flex_data.get('F3', 0),
                   flex_data.get('F4', 0), flex_data.get('F5', 0),
                   accel_data.get('AX', 0), accel_data.get('AY', 0), accel_data.get('AZ', 0),
                   gyro_data.get('GX', 0), gyro_data.get('GY', 0), gyro_data.get('GZ', 0)]
        return None
    
    def normalize_data(self, data):
        return (np.array(data) - self.scaler_mean) / self.scaler_scale
    
    def predict_gesture(self):
        if len(self.data_buffer) == self.sequence_length:
            sequence = np.array(list(self.data_buffer))
            sequence = sequence.reshape(1, self.sequence_length, 11)
            
            prediction = self.model.predict(sequence, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            if confidence > 0.8:  # Higher threshold
                gesture = self.label_classes[predicted_class]
                return gesture, confidence
        
        return None, 0
    
    def run(self):
        print("Starting real-time gesture prediction...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                if self.ser.in_waiting > 0:
                    try:
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    except:
                        continue
                        
                    if line and 'F1:' in line:
                        sensor_data = self.parse_sensor_data(line)
                        
                        if sensor_data:
                            normalized_data = self.normalize_data(sensor_data)
                            self.data_buffer.append(normalized_data)
                            
                            gesture, confidence = self.predict_gesture()
                            
                            if gesture:
                                # Stability check
                                if gesture == self.last_gesture:
                                    self.gesture_count += 1
                                else:
                                    self.gesture_count = 1
                                    self.last_gesture = gesture
                                
                                # Only print and speak if gesture is stable (3+ consecutive predictions)
                                if self.gesture_count >= 3:
                                    print(f"Predicted Gesture: {gesture} (Confidence: {confidence:.2f})")
                                    self.tts.say(gesture)
                                    self.tts.runAndWait()
                                    self.gesture_count = 0  # Reset to avoid spam
                                    time.sleep(2)  # Pause between predictions
        
        except KeyboardInterrupt:
            print("\nGesture prediction stopped")
        finally:
            self.ser.close()

if __name__ == "__main__":
    predictor = GesturePredictor()
    predictor.run()