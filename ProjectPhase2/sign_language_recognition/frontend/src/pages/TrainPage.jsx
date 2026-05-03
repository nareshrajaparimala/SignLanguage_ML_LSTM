import React, { useState, useEffect } from "react";
import axios from "axios";
import { trainModel, listLabels } from "../api";
import "./pages.css";

function TrainPage({ speak }) {
  const [labels, setLabels] = useState([]);
  const [k, setK] = useState(3);
  const [isTraining, setIsTraining] = useState(false);
  const [message, setMessage] = useState("");
  const [trainedGestures, setTrainedGestures] = useState([]);

  useEffect(() => {
    fetchLabels();
  }, []);

  const fetchLabels = async () => {
    try {
      const res = await listLabels();
      setLabels(res.data.labels);
    } catch (err) {
      console.error("Failed to fetch labels", err);
    }
  };

  const handleTrain = async () => {
    if (labels.length === 0) {
      setMessage("❌ No gestures found. Capture some first!");
      speak("No gestures to train");
      return;
    }

    setIsTraining(true);
    setMessage("⚙️ Training model...");
    speak("Training model");

    try {
      const res = await trainModel(k);
      setTrainedGestures(res.data.gestures);
      setMessage(`✅ Model trained! ${res.data.num_gestures} gestures ready.`);
      speak(`Model trained with ${res.data.num_gestures} gestures`);
    } catch (err) {
      setMessage(`❌ Training failed: ${err.message}`);
      speak("Training failed");
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <section className="page train-page">
      <h2>⚙️ Train Model</h2>

      <div className="train-container">
        <div className="info-box">
          <h3>Training Overview</h3>
          <p>
            Train a KNN (k-Nearest Neighbors) model on your captured gesture data.
            The model learns the pattern of each gesture and can predict new ones.
          </p>
        </div>

        <div className="gestures-list">
          <h3>📊 Available Gestures ({labels.length})</h3>
          {labels.length === 0 ? (
            <p className="empty-text">
              No gestures captured yet. Go to <strong>Capture</strong> to collect gestures.
            </p>
          ) : (
            <div className="gesture-tags">
              {labels.map((label) => (
                <span key={label} className="gesture-tag">
                  {label}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="train-controls">
          <div className="form-group">
            <label>K Value (number of neighbors)</label>
            <select value={k} onChange={(e) => setK(Number(e.target.value))}>
              <option value={1}>1 (Exact match)</option>
              <option value={3}>3 (Recommended)</option>
              <option value={5}>5 (More robust)</option>
              <option value={7}>7 (Smoother)</option>
            </select>
            <p className="help-text">
              Lower k = more sensitive, Higher k = more robust
            </p>
          </div>

          <button
            className="btn btn-primary btn-large"
            onClick={handleTrain}
            disabled={isTraining || labels.length === 0}
          >
            {isTraining ? "⚙️ Training..." : "🚀 Train Model"}
          </button>
        </div>

        {message && <p className="status-message">{message}</p>}

        {trainedGestures.length > 0 && (
          <div className="success-box">
            <h3>✅ Model Trained!</h3>
            <p>The model is ready for predictions.</p>
            <div className="gesture-tags">
              {trainedGestures.map((g) => (
                <span key={g} className="gesture-tag success">
                  {g}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="info-box">
        <h4>📝 Model Training Details:</h4>
        <ul>
          <li><strong>Algorithm:</strong> k-Nearest Neighbors (KNN)</li>
          <li><strong>Features per gesture:</strong> Mean + Standard Deviation of 11 sensor values</li>
          <li><strong>Feature vector:</strong> 22 features per gesture (11 mean + 11 std)</li>
          <li><strong>Training time:</strong> Instant (KNN is lazy learner)</li>
          <li><strong>Best practices:</strong> Collect 5-10 samples per gesture for accuracy</li>
        </ul>
      </div>
    </section>
  );
}

export default TrainPage;
