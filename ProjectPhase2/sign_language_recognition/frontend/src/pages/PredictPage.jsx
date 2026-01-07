import React, { useState } from "react";
import axios from "axios";
import { captureFrame, bufferFrame, predictLive, getBufferStatus } from "../api";
import "./pages.css";

function PredictPage({ speak }) {
  const [status, setStatus] = useState("idle"); // idle, capturing, predicting
  const [buffered, setBuffered] = useState(0);
  const [totalNeeded, setTotalNeeded] = useState(30);
  const [prediction, setPrediction] = useState(null);
  const [message, setMessage] = useState("");
  const [isCapturing, setIsCapturing] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [captureSpeed, setCaptureSpeed] = useState(100);

  const checkBuffer = async () => {
    try {
      const res = await getBufferStatus();
      setBuffered(res.data.frames_collected);
      setTotalNeeded(res.data.frames_needed);
    } catch (err) {
      console.error("Failed to check buffer", err);
    }
  };

  const startPrediction = async () => {
    setStatus("capturing");
    setIsCapturing(true);
    setMessage("🔄 Capturing frames for prediction...");
    setPrediction(null);
    speak("Starting prediction");

    try {
      for (let i = 0; i < totalNeeded; i++) {
        const frame = await captureFrame();
        await bufferFrame(frame.data);
        await checkBuffer();
        setMessage(`📸 Captured ${i + 1}/${totalNeeded} frames`);
        await new Promise((resolve) => setTimeout(resolve, captureSpeed));
      }

      // Prediction with buffered frames
      setStatus("predicting");
      setMessage("🤔 Analyzing gesture...");
      speak("Analyzing");

      const bufferRes = await getBufferStatus();
      if (bufferRes.data.frames_collected >= 30) {
        // Get current buffer frames for prediction
        // Note: In a real implementation, you'd need an endpoint to get buffered frames
        // For now, we'll capture 30 fresh frames again
        const frames = [];
        for (let i = 0; i < 30; i++) {
          try {
            const f = await captureFrame();
            frames.push(f.data);
            await new Promise((resolve) => setTimeout(resolve, 30));
          } catch (e) {
            console.warn("Frame capture error:", e);
          }
        }

        const predRes = await predictLive({ frames });
        const result = predRes.data;

        setPrediction(result.label);
        setConfidence(parseFloat((result.confidence * 100).toFixed(1)));
        setMessage(`✅ Prediction: "${result.label}" (${(result.confidence * 100).toFixed(1)}% confidence)`);
        speak(`Predicted gesture is ${result.label}`);
      }

      setStatus("complete");
    } catch (err) {
      setMessage(`❌ Error: ${err.message}`);
      speak("Error during prediction");
      setStatus("idle");
    } finally {
      setIsCapturing(false);
    }
  };

  const reset = () => {
    setPrediction(null);
    setMessage("");
    setStatus("idle");
    setBuffered(0);
  };

  const progress = (buffered / totalNeeded) * 100;
  const confidenceColor = confidence > 70 ? "#4caf50" : confidence > 50 ? "#ff9800" : "#f44336";

  return (
    <section className="page predict-page">
      <h2>🎯 Real-Time Prediction</h2>

      <div className="predict-container">
        {!prediction ? (
          <>
            <div className="progress-section">
              <h3>Capture for Prediction</h3>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="progress-text">
                {buffered} / {totalNeeded} frames
              </p>
            </div>

            <div className="controls">
              <div className="form-group">
                <label>Capture Speed (ms)</label>
                <input
                  type="range"
                  min="30"
                  max="300"
                  step="30"
                  value={captureSpeed}
                  onChange={(e) => setCaptureSpeed(Number(e.target.value))}
                  disabled={isCapturing}
                />
                <p className="help-text">{captureSpeed}ms</p>
              </div>

              <button
                className="btn btn-primary btn-large"
                onClick={startPrediction}
                disabled={isCapturing}
              >
                {isCapturing ? "🔄 Capturing..." : "🎬 Predict Gesture"}
              </button>
            </div>

            {message && <p className="status-message">{message}</p>}
          </>
        ) : (
          <div className="prediction-result">
            <div className="result-card">
              <h3 className="result-label">{prediction}</h3>
              <div className="confidence-meter">
                <div
                  className="confidence-bar"
                  style={{
                    width: `${confidence}%`,
                    backgroundColor: confidenceColor
                  }}
                ></div>
              </div>
              <p className="confidence-text">
                Confidence: {confidence.toFixed(1)}%
              </p>
            </div>

            {message && <p className="status-message success">{message}</p>}

            <button className="btn btn-secondary btn-large" onClick={reset}>
              🔄 Try Another Prediction
            </button>
          </div>
        )}
      </div>

      <div className="info-box">
        <h4>💡 Prediction Tips:</h4>
        <ul>
          <li>Make clear, consistent gestures</li>
          <li>Ensure Arduino is connected and sensors working</li>
          <li>Use the same movements as during training</li>
          <li>Higher confidence (greater than 70%) indicates more reliable prediction</li>
          <li>If confidence is low, retrain with more diverse samples</li>
        </ul>
      </div>
    </section>
  );
}

export default PredictPage;
