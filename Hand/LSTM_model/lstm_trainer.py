import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def load_and_prepare_data():
    # Load all CSV files
    files = glob.glob("gesture_data_*.csv")
    if not files:
        print("No gesture data files found!")
        return None, None, None, None
    
    all_data = []
    for file in files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    
    # Prepare features and labels
    feature_cols = ['F1', 'F2', 'F3', 'F4', 'F5', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
    X = data[feature_cols].values
    y = data['Gesture'].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Reshape for LSTM (samples, timesteps, features)
    sequence_length = 50
    X_sequences = []
    y_sequences = []
    
    for i in range(0, len(X_scaled) - sequence_length + 1, sequence_length):
        X_sequences.append(X_scaled[i:i + sequence_length])
        y_sequences.append(y_categorical[i + sequence_length - 1])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    return X_sequences, y_sequences, scaler, label_encoder

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    # Load and prepare data
    X, y, scaler, label_encoder = load_and_prepare_data()
    if X is None:
        return
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = build_lstm_model((X.shape[1], X.shape[2]), y.shape[1])
    print(model.summary())
    
    # Train model
    history = model.fit(X_train, y_train,
                       epochs=50,
                       batch_size=32,
                       validation_data=(X_test, y_test),
                       verbose=1)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions for confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Plot results
    plot_training_history(history)
    try:
        plot_confusion_matrix(y_test_classes, y_pred_classes, label_encoder.classes_)
    except:
        print("Confusion matrix skipped - insufficient class diversity in test set")
    
    # Classification report
    print("\nClassification Report:")
    try:
        print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))
    except ValueError as e:
        print(f"Classification report skipped: {e}")
        print(f"Unique classes in test set: {np.unique(y_test_classes)}")
    
    # Save model and preprocessing objects
    model.save('gesture_lstm_model.h5')
    np.save('scaler_params.npy', [scaler.mean_, scaler.scale_])
    np.save('label_encoder.npy', label_encoder.classes_)
    
    print("Model saved as gesture_lstm_model.h5")

if __name__ == "__main__":
    main()