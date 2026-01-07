# LSTM Gesture Recognition System

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Collection

1. Run data collector for each gesture:
```bash
python data_collector.py
```

2. Follow prompts to collect 50 frames per gesture for:
   - fist
   - open_hand
   - peace
   - thumbs_up
   - point

## Model Training

1. Train LSTM model:
```bash
python lstm_trainer.py
```

This will:
- Load and normalize data
- Split into train/test sets
- Train LSTM with dropout
- Generate confusion matrix
- Plot learning curves
- Save model as `gesture_lstm_model.h5`

## Raspberry Pi Inference

1. Copy files to Raspberry Pi:
   - `gesture_lstm_model.h5`
   - `scaler_params.npy`
   - `label_encoder.npy`
   - `raspberry_pi_inference.py`

2. Run real-time recognition:
```bash
python raspberry_pi_inference.py
```

## Gesture-to-Text Mapping

- **fist**: "Closed fist detected"
- **open_hand**: "Open hand detected"  
- **peace**: "Peace sign detected"
- **thumbs_up**: "Thumbs up detected"
- **point**: "Pointing gesture detected"

## Model Architecture

- LSTM layers with dropout for sequence learning
- 50-frame sequences of 11 sensor features
- Categorical cross-entropy loss
- Adam optimizer