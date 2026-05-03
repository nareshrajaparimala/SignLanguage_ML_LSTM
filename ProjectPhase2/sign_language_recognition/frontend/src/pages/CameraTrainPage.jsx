import React, { useState, useEffect } from "react";
import axios from "axios";
import "./pages.css";
import "./camera.css";

function CameraTrainPage({ speak }) {
  const [gestures, setGestures] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState("");
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    loadGestures();
  }, []);

  const loadGestures = async () => {
    try {
      const response = await axios.get("/api/list-gestures");
      setGestures(response.data.gestures);
    } catch (error) {
      console.error("Error loading gestures:", error);
    }
  };

  const trainModel = async () => {
    if (gestures.length < 2) {
      speak("You need at least 2 different gestures to train the model");
      return;
    }

    setIsTraining(true);
    setTrainingStatus("Training CNN+LSTM model... This may take a few minutes.");
    speak("Starting model training. This will take a few minutes.");

    try {
      const response = await axios.post("/api/train-model");
      setModelInfo(response.data);
      setTrainingStatus("Model trained successfully!");
      speak(`Model trained successfully with ${response.data.accuracy.toFixed(2)} accuracy`);
    } catch (error) {
      console.error("Training error:", error);
      setTrainingStatus("Error training model: " + (error.response?.data?.detail || error.message));
      speak("Error training model");
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <section className="page train-page">
      <h2>⚙️ Train CNN+LSTM Model</h2>
      <p>Train a deep learning model on your captured gesture sequences</p>

      <div className="training-container">
        <div className="gestures-list">
          <h3>Captured Gestures ({gestures.length})</h3>
          {gestures.length === 0 ? (
            <p className="no-data">No gestures captured yet. Go to Capture page first.</p>
          ) : (
            <div className="gesture-grid">
              {gestures.map((gesture, index) => (
                <div key={index} className="gesture-card">
                  <h4>{gesture.label}</h4>
                  <p className="gesture-text">"{gesture.text}"</p>
                  <p className="sequence-count">{gesture.sequences} sequences</p>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="training-controls">
          <button
            className={`btn ${isTraining ? "btn-secondary" : "btn-primary"}`}
            onClick={trainModel}
            disabled={isTraining || gestures.length < 2}
          >
            {isTraining ? "Training..." : "Train Model"}
          </button>

          {isTraining && (
            <div className="training-progress">
              <div className="spinner"></div>
              <p>Training in progress...</p>
            </div>
          )}

          <div className="status-display">
            <p>{trainingStatus}</p>
          </div>

          {modelInfo && (
            <div className="model-info">
              <h3>Model Information</h3>
              <ul>
                <li>Classes: {modelInfo.classes.join(", ")}</li>
                <li>Training Samples: {modelInfo.samples}</li>
                <li>Accuracy: {(modelInfo.accuracy * 100).toFixed(2)}%</li>
              </ul>
            </div>
          )}
        </div>
      </div>

      <div className="training-info">
        <h3>About CNN+LSTM Model:</h3>
        <ul>
          <li><strong>CNN layers:</strong> Extract spatial features from hand/face landmarks</li>
          <li><strong>LSTM layers:</strong> Learn temporal patterns in gesture sequences</li>
          <li><strong>Input:</strong> 30 frames × 130 features (hand + face landmarks)</li>
          <li><strong>Training:</strong> 50 epochs with validation split</li>
          <li><strong>Output:</strong> Gesture classification with confidence scores</li>
        </ul>
      </div>

      <div className="requirements">
        <h3>Training Requirements:</h3>
        <ul>
          <li>Minimum 2 different gesture types</li>
          <li>At least 1 sequence per gesture (recommended: 3-5 sequences)</li>
          <li>Each sequence contains 30 frames of landmarks</li>
          <li>Training time: 2-5 minutes depending on data size</li>
        </ul>
      </div>
    </section>
  );
}

export default CameraTrainPage;