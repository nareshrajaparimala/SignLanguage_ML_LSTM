import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./pages.css";
import "./camera.css";
import "./hand-detection.css";

function CameraPredictPage({ speak }) {
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState("");
  const [preview, setPreview] = useState(null);
  const [isLiveMode, setIsLiveMode] = useState(false);
  const intervalRef = useRef(null);
  const liveIntervalRef = useRef(null);

  useEffect(() => {
    startPreview();
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (liveIntervalRef.current) clearInterval(liveIntervalRef.current);
    };
  }, []);

  const startPreview = () => {
    intervalRef.current = setInterval(async () => {
      if (!isPredicting) {
        try {
          const response = await axios.get("/api/hand-detection", {
            timeout: 5000
          });
          setPreview(response.data.annotated_frame);
          setStatus(`Camera ready - ${response.data.hand_count} hands detected`);
        } catch (error) {
          console.error("Preview error:", error);
          setStatus("Camera error: " + error.message);
        }
      }
    }, 500);
  };

  const predictOnce = async () => {
    setIsPredicting(true);
    setStatus("Capturing 30 frames for prediction...");
    setPrediction(null);

    try {
      const response = await axios.post("/api/predict-live");
      setPrediction(response.data);
      setStatus("Prediction complete!");
      // Only speak if it's a recognized gesture
      if (response.data.gesture !== "NO_GESTURE" && response.data.gesture !== "UNCERTAIN") {
        speak(response.data.text);
      }
    } catch (error) {
      console.error("Prediction error:", error);
      setStatus("Error: " + (error.response?.data?.detail || error.message));
      speak("Prediction failed");
    } finally {
      setIsPredicting(false);
    }
  };

  const toggleLiveMode = () => {
    if (isLiveMode) {
      // Stop live mode
      if (liveIntervalRef.current) {
        clearInterval(liveIntervalRef.current);
        liveIntervalRef.current = null;
      }
      setIsLiveMode(false);
      setStatus("Live mode stopped");
    } else {
      // Start live mode
      setIsLiveMode(true);
      setStatus("Live mode started - predicting every 5 seconds");
      
      liveIntervalRef.current = setInterval(async () => {
        if (!isPredicting) {
          try {
            setIsPredicting(true);
            const response = await axios.post("/api/predict-live");
            setPrediction(response.data);
            
            // Only speak if it's a real gesture with good confidence
            if (response.data.gesture !== "NO_GESTURE" && 
                response.data.gesture !== "UNCERTAIN" && 
                response.data.confidence > 0.35) {
              speak(response.data.text);
            }
          } catch (error) {
            console.error("Live prediction error:", error);
          } finally {
            setIsPredicting(false);
          }
        }
      }, 5000);
    }
  };

  return (
    <section className="page predict-page">
      <h2>🎯 Real-time Gesture Prediction</h2>
      <p>Use the trained CNN+LSTM model to recognize gestures in real-time</p>

      <div className="predict-container">
        <div className="camera-section">
          <h3>Camera Feed</h3>
          {preview ? (
            <img src={preview} alt="Camera preview" className="preview-image" />
          ) : (
            <div className="no-preview">No camera feed</div>
          )}
        </div>

        <div className="prediction-controls">
          <div className="control-buttons">
            <button
              className={`btn ${isPredicting ? "btn-secondary" : "btn-primary"}`}
              onClick={predictOnce}
              disabled={isPredicting || isLiveMode}
            >
              {isPredicting ? "Predicting..." : "Predict Once"}
            </button>

            <button
              className={`btn ${isLiveMode ? "btn-danger" : "btn-success"}`}
              onClick={toggleLiveMode}
              disabled={isPredicting}
            >
              {isLiveMode ? "Stop Live Mode" : "Start Live Mode"}
            </button>
          </div>

          <div className="status-display">
            <p>{status}</p>
          </div>

          {prediction && (
            <div className="prediction-result">
              <h3>Prediction Result</h3>
              <div className="main-prediction">
                <h4>Gesture: {prediction.gesture}</h4>
                <p className="predicted-text">"{prediction.text}"</p>
                <p className="confidence">
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </p>
              </div>

              {prediction.all_predictions && (
                <div className="all-predictions">
                  <h4>All Predictions:</h4>
                  <div className="prediction-list">
                    {Object.entries(prediction.all_predictions)
                      .sort(([,a], [,b]) => b - a)
                      .map(([gesture, confidence]) => (
                        <div key={gesture} className="prediction-item">
                          <span className="gesture-name">{gesture}</span>
                          <div className="confidence-bar">
                            <div 
                              className="confidence-fill"
                              style={{ width: `${confidence * 100}%` }}
                            ></div>
                          </div>
                          <span className="confidence-value">
                            {(confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="prediction-info">
        <h3>How it works:</h3>
        <ol>
          <li><strong>Frame Capture:</strong> Captures 30 consecutive frames (3 seconds)</li>
          <li><strong>Landmark Extraction:</strong> Detects hand and face landmarks using MediaPipe</li>
          <li><strong>Feature Processing:</strong> Converts landmarks to 130-dimensional feature vectors</li>
          <li><strong>CNN+LSTM Prediction:</strong> Processes sequence through trained neural network</li>
          <li><strong>Text-to-Speech:</strong> Speaks the predicted text if confidence &gt; 70%</li>
        </ol>
      </div>

      <div className="tips">
        <h3>Tips for better recognition:</h3>
        <ul>
          <li>Ensure good lighting and clear view of hands and face</li>
          <li>Perform gestures similar to how you trained them</li>
          <li>Hold gestures steady during the 3-second capture window</li>
          <li>Train with multiple sequences per gesture for better accuracy</li>
          <li>Live mode predicts every 5 seconds automatically</li>
        </ul>
      </div>
    </section>
  );
}

export default CameraPredictPage;