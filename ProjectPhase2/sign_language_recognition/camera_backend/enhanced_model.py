#!/usr/bin/env python3
"""
Enhanced Sign Language Recognition Model with Rotation and Velocity Features
LSTM + CNN Hybrid Architecture for Temporal and Spatial Feature Learning
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
    BatchNormalization, Input, Concatenate, 
    GlobalAveragePooling1D, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib

class EnhancedSignLanguageModel:
    def __init__(self, sequence_length=30, num_classes=10):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.feature_size = 176  # Updated for rotation + velocity + angle + handedness features
        self.model = None
        self.scaler = StandardScaler()
        
    def create_multi_stream_model(self):
        """Create multi-stream CNN+LSTM model for different feature types"""
        
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.feature_size))
        
        # Feature extraction streams
        # Stream 1: Hand landmarks (0-125: 2 hands * 63 features each)
        landmarks_features = input_layer[:, :, :126]
        
        # Stream 2: Finger angles and bend ratios (126-135: 2 hands * 5 features each)
        finger_features = input_layer[:, :, 126:136]
        
        # Stream 3: Rotation features (136-153: 2 hands * 9 features each)
        rotation_features = input_layer[:, :, 136:154]
        
        # Stream 4: Velocity features (154-160: 2 hands * 7 features each)
        velocity_features = input_layer[:, :, 154:168]
        
        # Stream 5: Hand angles and shape (168-169: 2 hands * 5 features each)
        shape_features = input_layer[:, :, 158:]
        
        # CNN processing for each stream
        def create_cnn_stream(features, filters, name):
            x = Conv1D(filters[0], 3, activation='relu', name=f'{name}_conv1')(features)
            x = BatchNormalization(name=f'{name}_bn1')(x)
            x = Dropout(0.2, name=f'{name}_dropout1')(x)
            
            x = Conv1D(filters[1], 3, activation='relu', name=f'{name}_conv2')(x)
            x = BatchNormalization(name=f'{name}_bn2')(x)
            x = MaxPooling1D(2, name=f'{name}_pool')(x)
            x = Dropout(0.2, name=f'{name}_dropout2')(x)
            
            return x
        
        # Process each stream
        landmarks_cnn = create_cnn_stream(landmarks_features, [32, 64], 'landmarks')
        finger_cnn = create_cnn_stream(finger_features, [16, 32], 'finger')
        rotation_cnn = create_cnn_stream(rotation_features, [16, 32], 'rotation')
        velocity_cnn = create_cnn_stream(velocity_features, [16, 32], 'velocity')
        shape_cnn = create_cnn_stream(shape_features, [8, 16], 'shape')
        
        # Concatenate all streams
        combined_features = Concatenate(axis=-1)([
            landmarks_cnn, finger_cnn, rotation_cnn, velocity_cnn, shape_cnn
        ])
        
        # LSTM layers for temporal modeling
        lstm1 = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(combined_features)
        lstm1_bn = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(lstm1_bn)
        lstm2_bn = BatchNormalization()(lstm2)
        
        lstm3 = LSTM(32, dropout=0.3, recurrent_dropout=0.3)(lstm2_bn)
        lstm3_bn = BatchNormalization()(lstm3)
        
        # Dense layers for classification
        dense1 = Dense(64, activation='relu')(lstm3_bn)
        dense1_dropout = Dropout(0.4)(dense1)
        dense1_bn = BatchNormalization()(dense1_dropout)
        
        dense2 = Dense(32, activation='relu')(dense1_bn)
        dense2_dropout = Dropout(0.3)(dense2)
        
        # Output layer
        output = Dense(self.num_classes, activation='softmax')(dense2_dropout)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        return model
    
    def create_simple_model(self):
        """Create simplified model for small datasets"""
        model = Sequential([
            # Simplified CNN layers
            Conv1D(32, 3, activation='relu', input_shape=(self.sequence_length, self.feature_size)),
            BatchNormalization(),
            Dropout(0.2),
            
            Conv1D(64, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            # LSTM layers
            LSTM(64, return_sequences=True, dropout=0.3),
            BatchNormalization(),
            
            LSTM(32, dropout=0.3),
            BatchNormalization(),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.4),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with appropriate optimizer and loss"""
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_data(self, X, y):
        """Prepare and normalize data"""
        # Reshape for scaling (flatten sequence dimension)
        X_reshaped = X.reshape(-1, self.feature_size)
        
        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Reshape back to sequences
        X_scaled = X_scaled.reshape(X.shape)
        
        return X_scaled, y
    
    def augment_data(self, X, y, augmentation_factor=3):
        """Enhanced data augmentation for rotation and velocity features"""
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(augmentation_factor):
            X_aug = X.copy()
            
            # Add noise to landmarks (first 126 features)
            noise_landmarks = np.random.normal(0, 0.005, X_aug[:, :, :126].shape)
            X_aug[:, :, :126] += noise_landmarks
            
            # Add noise to finger angles (126-136)
            noise_angles = np.random.normal(0, 2, X_aug[:, :, 126:136].shape)
            X_aug[:, :, 126:136] += noise_angles
            
            # Add rotation variations (136-154)
            rotation_noise = np.random.normal(0, 5, X_aug[:, :, 136:154].shape)
            X_aug[:, :, 136:154] += rotation_noise
            
            # Add velocity variations (154-168)
            velocity_noise = np.random.normal(0, 0.01, X_aug[:, :, 154:168].shape)
            X_aug[:, :, 154:168] += velocity_noise
            
            # Time shifting
            shift = np.random.randint(-3, 4)
            if shift != 0:
                X_aug = np.roll(X_aug, shift, axis=1)
            
            # Speed variation (temporal scaling)
            if np.random.random() > 0.5:
                # Randomly sample frames to simulate speed changes
                indices = np.sort(np.random.choice(
                    self.sequence_length, 
                    size=self.sequence_length, 
                    replace=True
                ))
                X_aug = X_aug[:, indices, :]
            
            X_augmented.append(X_aug)
            y_augmented.append(y)
        
        return np.concatenate(X_augmented), np.concatenate(y_augmented)
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=16, use_augmentation=True):
        """Train the model with enhanced features"""
        
        # Determine model complexity based on dataset size
        if len(X) > 50:
            self.model = self.create_multi_stream_model()
            print("Using multi-stream CNN+LSTM model")
        else:
            self.model = self.create_simple_model()
            print("Using simplified CNN+LSTM model")
        
        # Compile model
        self.compile_model()
        
        # Prepare data
        X_processed, y_processed = self.prepare_data(X, y)
        
        # Data augmentation
        if use_augmentation:
            X_processed, y_processed = self.augment_data(X_processed, y_processed)
            print(f"Data augmented: {len(X)} -> {len(X_processed)} samples")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_processed, y_processed,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions with preprocessing"""
        # Reshape and scale
        X_reshaped = X.reshape(-1, self.feature_size)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        return self.model.predict(X_scaled)
    
    def save_model(self, model_path, scaler_path):
        """Save model and scaler"""
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        """Load model and scaler"""
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")

# Feature extraction utilities
def extract_rotation_features(landmarks):
    """Extract rotation features from hand landmarks"""
    if len(landmarks) < 21:
        return [0.0] * 9
    
    # Calculate rotation angles and direction vectors
    # Implementation matches the server-side calculation
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]
    
    # Calculate hand plane normal vector
    v1 = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
    v2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
    normal = np.cross(v1, v2)
    
    if np.linalg.norm(normal) > 0:
        normal = normal / np.linalg.norm(normal)
    
    # Calculate rotation angles
    hand_vector = np.array([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, 0])
    if np.linalg.norm(hand_vector) > 0:
        hand_vector = hand_vector / np.linalg.norm(hand_vector)
        roll = np.arctan2(hand_vector[1], hand_vector[0]) * 180 / np.pi
    else:
        roll = 0
    
    pitch = np.arcsin(np.clip(normal[1], -1, 1)) * 180 / np.pi
    yaw = np.arctan2(normal[0], normal[2]) * 180 / np.pi
    
    # Direction vector
    direction = np.array([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, middle_mcp.z - wrist.z])
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
    
    return [roll, pitch, yaw] + direction.tolist() + normal.tolist()

def extract_velocity_features(current_landmarks, previous_landmarks, time_delta=0.1):
    """Extract velocity features from consecutive frames"""
    if not previous_landmarks or len(current_landmarks) < 21 or len(previous_landmarks) < 21:
        return [0.0] * 7
    
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
    
    return [speed] + velocity.tolist() + direction.tolist()

if __name__ == "__main__":
    # Example usage
    model = EnhancedSignLanguageModel(sequence_length=30, num_classes=5)
    
    # Create dummy data for testing
    X_dummy = np.random.random((10, 30, 180))
    y_dummy = np.random.randint(0, 5, 10)
    
    # Train model
    history = model.train(X_dummy, y_dummy, epochs=5)
    
    # Make prediction
    prediction = model.predict(X_dummy[:1])
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction: {prediction}")