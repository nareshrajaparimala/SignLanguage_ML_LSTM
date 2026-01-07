import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import glob

# Load all gesture data files
files = glob.glob("gesture_data_*.csv")
print(f"Found {len(files)} data files: {files}")

all_data = []
for file in files:
    df = pd.read_csv(file)
    all_data.append(df)
    print(f"Loaded {file}: {len(df)} samples, gesture: {df['Gesture'].iloc[0]}")

data = pd.concat(all_data, ignore_index=True)
print(f"Total samples: {len(data)}")
print(f"Unique gestures: {data['Gesture'].unique()}")

# Prepare features
feature_cols = ['F1', 'F2', 'F3', 'F4', 'F5', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
X = data[feature_cols].values
y = data['Gesture'].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Classes: {label_encoder.classes_}")

# Create overlapping sequences for better training
sequence_length = 50
X_sequences = []
y_sequences = []

# Group by gesture to create sequences within each gesture
for gesture in data['Gesture'].unique():
    gesture_data = data[data['Gesture'] == gesture]
    gesture_X = scaler.transform(gesture_data[feature_cols].values)
    gesture_y = y_categorical[label_encoder.transform([gesture])[0]]
    
    # Create overlapping sequences (step=10 instead of 50)
    for i in range(0, len(gesture_X) - sequence_length + 1, 10):
        X_sequences.append(gesture_X[i:i + sequence_length])
        y_sequences.append(gesture_y)
    
    print(f"Created {len(range(0, len(gesture_X) - sequence_length + 1, 10))} sequences for gesture '{gesture}'")

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print(f"Sequence data shape: {X_sequences.shape}")
print(f"Sequence labels shape: {y_sequences.shape}")

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_sequences.shape[1], X_sequences.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_sequences.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train
print("Training model...")
history = model.fit(X_sequences, y_sequences, epochs=50, batch_size=2, verbose=1, validation_split=0.2)

# Save everything
model.save('gesture_lstm_model.h5')
np.save('scaler_params.npy', [scaler.mean_, scaler.scale_])
np.save('label_encoder.npy', label_encoder.classes_)

print("Model saved successfully!")
print(f"Available gestures: {label_encoder.classes_}")
print("Training complete!")