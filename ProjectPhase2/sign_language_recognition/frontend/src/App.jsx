import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";
import CapturePage from "./pages/CapturePage";
import TrainPage from "./pages/TrainPage";
import PredictPage from "./pages/PredictPage";
import CameraCaptureePage from "./pages/CameraCaptureePage";
import CameraTrainPage from "./pages/CameraTrainPage";
import CameraPredictPage from "./pages/CameraPredictPage";
import MediaPipeHandsPage from "./pages/MediaPipeHandsPage";
import GestureManagerPage from "./pages/GestureManagerPage";
import LiveRecognitionPage from "./pages/LiveRecognitionPage";

function App() {
  const [currentPage, setCurrentPage] = useState("home");
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await axios.get("/api/status");
      setStatus(response.data);
      setError(null);
    } catch (err) {
      setError("Cannot connect to API");
    }
  };

  const speak = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 Sign Language Recognition System</h1>
        <p className="subtitle">Camera-based gesture recognition with CNN+LSTM</p>
      </header>

      <nav className="navbar">
        <button
          className={`nav-btn ${currentPage === "home" ? "active" : ""}`}
          onClick={() => setCurrentPage("home")}
        >
          Home
        </button>
        <button
          className={`nav-btn ${currentPage === "camera-capture" ? "active" : ""}`}
          onClick={() => setCurrentPage("camera-capture")}
        >
          📹 Camera Capture
        </button>
        <button
          className={`nav-btn ${currentPage === "camera-train" ? "active" : ""}`}
          onClick={() => setCurrentPage("camera-train")}
        >
          ⚙️ Train CNN+LSTM
        </button>
        <button
          className={`nav-btn ${currentPage === "camera-predict" ? "active" : ""}`}
          onClick={() => setCurrentPage("camera-predict")}
        >
          🎯 Live Predict
        </button>
        <button
          className={`nav-btn ${currentPage === "mediapipe-hands" ? "active" : ""}`}
          onClick={() => setCurrentPage("mediapipe-hands")}
        >
          ✋ MediaPipe Hands
        </button>
        <button
          className={`nav-btn ${currentPage === "gesture-manager" ? "active" : ""}`}
          onClick={() => setCurrentPage("gesture-manager")}
        >
          🗂️ Manage Gestures
        </button>
        <button
          className={`nav-btn ${currentPage === "live-recognition" ? "active" : ""}`}
          onClick={() => setCurrentPage("live-recognition")}
        >
          🎯 Live Recognition
        </button>
        <button
          className={`nav-btn ${currentPage === "capture" ? "active" : ""}`}
          onClick={() => setCurrentPage("capture")}
        >
          📊 Arduino (Old)
        </button>
      </nav>

      {error && <div className="error-banner">{error}</div>}

      {status && (
        <div className="status-bar">
          <span>
            {status.serial_connected ? "✅ Arduino Connected" : "❌ Arduino Disconnected"}
          </span>
          <span>
            {status.model_trained
              ? `✅ Model Ready (${status.gestures?.length || 0} gestures)`
              : "⚠️ Model Not Trained"}
          </span>
        </div>
      )}

      <main className="app-content">
        {currentPage === "home" && <HomePage status={status} speak={speak} />}
        {currentPage === "camera-capture" && <CameraCaptureePage speak={speak} />}
        {currentPage === "camera-train" && <CameraTrainPage speak={speak} />}
        {currentPage === "camera-predict" && <CameraPredictPage speak={speak} />}
        {currentPage === "mediapipe-hands" && <MediaPipeHandsPage />}
        {currentPage === "gesture-manager" && <GestureManagerPage speak={speak} />}
        {currentPage === "live-recognition" && <LiveRecognitionPage speak={speak} />}
        {currentPage === "capture" && <CapturePage speak={speak} />}
        {currentPage === "train" && <TrainPage speak={speak} />}
        {currentPage === "predict" && <PredictPage speak={speak} />}
      </main>

      <footer className="app-footer">
        <p>© 2025 Camera Sign Language Recognition | Powered by MediaPipe + TensorFlow</p>
      </footer>
    </div>
  );
}

function HomePage({ status, speak }) {
  return (
    <section className="page home-page">
      <h2>Welcome to Sign Language Recognition</h2>
      <div className="info-grid">
        <div className="info-card">
          <h3>📹 Step 1: Camera Capture</h3>
          <p>Record hand and face gestures using your camera with 30-frame sequences.</p>
        </div>
        <div className="info-card">
          <h3>🧠 Step 2: Train CNN+LSTM</h3>
          <p>Train a deep learning model on your captured gesture sequences.</p>
        </div>
        <div className="info-card">
          <h3>🎯 Step 3: Live Predict</h3>
          <p>Real-time gesture recognition with text-to-speech output.</p>
        </div>
      </div>

      {status && (
        <div className="system-info">
          <h3>System Status</h3>
          <ul>
            <li>
              Camera: {status.camera_connected ? "✅ Connected" : "❌ Disconnected"}
            </li>
            <li>
              CNN+LSTM Model: {status.model_trained ? "✅ Trained" : "⚠️ Not Trained"}
            </li>
            <li>Gestures Trained: {status.gestures?.length || 0}</li>
            <li>Dataset: {status.dataset_path}</li>
          </ul>
        </div>
      )}

      <div className="quick-links">
        <a href="#" className="btn btn-primary">
          📖 View Documentation
        </a>
        <a href="#" className="btn btn-secondary">
          🎓 Tutorial
        </a>
      </div>
    </section>
  );
}

export default App;
