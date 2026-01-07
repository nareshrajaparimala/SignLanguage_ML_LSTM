#!/usr/bin/env python3
"""
Enhanced Training Script with Rotation and Velocity Features
"""

import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_model import EnhancedSignLanguageModel
import joblib

def load_enhanced_dataset(dataset_dir):
    """Load dataset with enhanced features including rotation and velocity"""
    X, y = [], []
    
    dataset_path = Path(dataset_dir)
    
    for gesture_dir in dataset_path.iterdir():
        if not gesture_dir.is_dir():
            continue
        
        gesture_name = gesture_dir.name
        
        for sequence_file in gesture_dir.glob("sequence_*.json"):
            try:
                with open(sequence_file, 'r') as f:
                    data = json.load(f)
                    
                if "frames" in data and data["frames"]:
                    # Ensure we have the right number of features (180)
                    frames = data["frames"]
                    
                    # Pad or truncate frames to ensure consistent sequence length
                    target_frames = 30
                    if len(frames) < target_frames:
                        # Pad with last frame
                        while len(frames) < target_frames:
                            frames.append(frames[-1] if frames else [0.0] * 176)
                    elif len(frames) > target_frames:
                        # Truncate to target length
                        frames = frames[:target_frames]
                    
                    # Ensure each frame has 176 features
                    processed_frames = []
                    for frame in frames:
                        if len(frame) < 176:
                            # Pad with zeros
                            frame.extend([0.0] * (176 - len(frame)))
                        elif len(frame) > 176:
                            # Truncate
                            frame = frame[:176]
                        processed_frames.append(frame)
                    
                    X.append(processed_frames)
                    y.append(gesture_name)
                    
            except Exception as e:
                print(f"Error loading {sequence_file}: {e}")
                continue
    
    return np.array(X), np.array(y)

def analyze_features(X, y, label_encoder):
    """Analyze feature distributions and patterns"""
    print("\\n=== Feature Analysis ===")
    
    # Feature ranges
    feature_ranges = {
        'landmarks': (0, 126),
        'finger_angles': (126, 136),
        'rotation': (136, 154),
        'velocity': (154, 168),
        'shape_angles': (168, 170)
    }
    
    for feature_type, (start, end) in feature_ranges.items():
        feature_data = X[:, :, start:end]
        print(f"{feature_type}: mean={feature_data.mean():.4f}, std={feature_data.std():.4f}")
    
    # Class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"\\nClass distribution:")
    for class_name, count in zip(unique_classes, counts):
        print(f"  {class_name}: {count} samples")

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training')
    if 'val_accuracy' in history.history:
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training')
    if 'val_loss' in history.history:
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
    
    # Additional metrics (if available)
    if 'val_accuracy' in history.history:
        axes[1, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1, 1].set_title('Accuracy Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
    else:
        axes[1, 1].plot(history.history['accuracy'])
        axes[1, 1].set_title('Training Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def train_enhanced_model(dataset_dir="gesture_dataset", model_dir="models"):
    """Train enhanced model with rotation and velocity features"""
    
    print("=== Enhanced Sign Language Model Training ===")
    
    # Load dataset
    print("Loading dataset...")
    X, y = load_enhanced_dataset(dataset_dir)
    
    if len(X) == 0:
        raise ValueError("No training data found!")
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} frames and {X.shape[2]} features each")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Analyze features
    analyze_features(X, y_encoded, label_encoder)
    
    # Create model
    model = EnhancedSignLanguageModel(
        sequence_length=X.shape[1],
        num_classes=num_classes
    )
    
    # Determine training parameters based on dataset size
    if len(X) >= 20:
        validation_split = 0.2
        epochs = 100
        batch_size = min(16, max(4, len(X) // 8))
        use_augmentation = True
    else:
        validation_split = 0.1 if len(X) >= 10 else 0.0
        epochs = 150
        batch_size = min(8, max(2, len(X) // 4))
        use_augmentation = True
    
    print(f"Training parameters:")
    print(f"  Validation split: {validation_split}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Data augmentation: {use_augmentation}")
    
    # Train model
    print("\\nStarting training...")
    history = model.train(
        X, y_encoded,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        use_augmentation=use_augmentation
    )
    
    # Save model and artifacts
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "enhanced_gesture_model.h5"
    scaler_path = model_dir / "enhanced_scaler.joblib"
    encoder_path = model_dir / "enhanced_label_encoder.joblib"
    
    model.save_model(str(model_path), str(scaler_path))
    joblib.dump(label_encoder, encoder_path)
    
    # Save training info
    training_info = {
        "model_type": "enhanced_cnn_lstm",
        "classes": label_encoder.classes_.tolist(),
        "num_samples": len(X),
        "num_classes": num_classes,
        "sequence_length": X.shape[1],
        "feature_size": X.shape[2],
        "final_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 0 else 0.0,
        "epochs_trained": len(history.history['accuracy']),
        "features_included": [
            "hand_landmarks", "finger_angles", "finger_bend_ratios",
            "hand_rotation", "hand_velocity", "hand_shape"
        ]
    }
    
    with open(model_dir / "enhanced_training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Plot training history
    plot_training_history(history, model_dir / "training_history.png")
    
    # Evaluate model
    print("\\n=== Training Results ===")
    print(f"Final training accuracy: {training_info['final_accuracy']:.4f}")
    if training_info['final_val_accuracy'] > 0:
        print(f"Final validation accuracy: {training_info['final_val_accuracy']:.4f}")
    print(f"Epochs trained: {training_info['epochs_trained']}")
    
    # Test prediction
    print("\\n=== Testing Prediction ===")
    test_sample = X[:1]
    prediction = model.predict(test_sample)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
    confidence = np.max(prediction[0])
    
    print(f"Test prediction: {predicted_class} (confidence: {confidence:.4f})")
    
    return model, label_encoder, training_info

if __name__ == "__main__":
    try:
        model, encoder, info = train_enhanced_model()
        print("\\n✅ Enhanced model training completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()