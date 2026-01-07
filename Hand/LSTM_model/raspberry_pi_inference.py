import serial
import numpy as np
import tensorflow as tf
from collections import deque
import re
import time
import pyttsx3

class GestureRecognizer:
    def __init__(self, model_path='gesture_lstm_model.h5', port='/dev/ttyUSB0'):
        # Load model and preprocessing
        self.model = tf.keras.models.load_model(model_path)
        scaler_params = np.load('scaler_params.npy')
        self.scaler_mean = scaler_params[0]
        self.scaler_scale = scaler_params[1]
        self.label_classes = np.load('label_encoder.npy')
        
        # Serial connection
        self.ser = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)
        
        # Buffer for sequence data
        self.sequence_length = 50
        self.data_buffer = deque(maxlen=self.sequence_length)
        
        # Text-to-speech
        self.tts = pyttsx3.init()
        
        # Gesture to text mapping
        self.gesture_texts = {
            'fist': 'Closed fist detected',
            'open_hand': 'Open hand detected',
            'peace': 'Peace sign detected',
            'thumbs_up': 'Thumbs up detected',
            'point': 'Pointing gesture detected'
        }
        
        print("Gesture recognizer initialized")
    
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
            # Prepare sequence for prediction
            sequence = np.array(list(self.data_buffer))
            sequence = sequence.reshape(1, self.sequence_length, 11)
            
            # Predict
            prediction = self.model.predict(sequence, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            if confidence > 0.7:  # Confidence threshold
                gesture = self.label_classes[predicted_class]
                return gesture, confidence
        
        return None, 0
    
    def speak_gesture(self, gesture):
        text = self.gesture_texts.get(gesture, f"{gesture} gesture detected")
        print(f"Speaking: {text}")
        self.tts.say(text)
        self.tts.runAndWait()
    
    def run(self):
        print("Starting real-time gesture recognition...")
        last_gesture = None
        gesture_count = 0
        
        try:
            while True:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line and 'F1:' in line:
                        sensor_data = self.parse_sensor_data(line)
                        
                        if sensor_data:
                            # Normalize and add to buffer
                            normalized_data = self.normalize_data(sensor_data)
                            self.data_buffer.append(normalized_data)
                            
                            # Predict gesture
                            gesture, confidence = self.predict_gesture()
                            
                            if gesture:
                                if gesture == last_gesture:
                                    gesture_count += 1
                                else:
                                    gesture_count = 1
                                    last_gesture = gesture
                                
                                # Speak gesture if detected consistently
                                if gesture_count >= 3:  # Require 3 consistent predictions
                                    print(f"Gesture: {gesture} (Confidence: {confidence:.2f})")
                                    self.speak_gesture(gesture)
                                    gesture_count = 0
                                    time.sleep(2)  # Prevent rapid repetition
        
        except KeyboardInterrupt:
            print("\nGesture recognition stopped")
        finally:
            self.ser.close()

if __name__ == "__main__":
    # Update port for your Raspberry Pi
    recognizer = GestureRecognizer(port='/dev/ttyUSB0')
    recognizer.run()