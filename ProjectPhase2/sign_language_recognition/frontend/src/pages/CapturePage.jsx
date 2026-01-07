import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  captureFrame,
  bufferFrame,
  getBufferStatus,
  saveLabel
} from "../api";
import "./pages.css";

function CapturePage({ speak }) {
  const [status, setStatus] = useState("idle"); // idle, capturing, complete
  const [buffered, setBuffered] = useState(0);
  const [totalNeeded, setTotalNeeded] = useState(30);
  const [label, setLabel] = useState("");
  const [message, setMessage] = useState("");
  const [isCapturing, setIsCapturing] = useState(false);
  const [captureSpeed, setCaptureSpeed] = useState(100); // ms between captures

  const checkBuffer = async () => {
    try {
      const res = await getBufferStatus();
      setBuffered(res.data.frames_collected);
      setTotalNeeded(res.data.frames_needed);
    } catch (err) {
      console.error("Failed to check buffer", err);
    }
  };

  useEffect(() => {
    checkBuffer();
  }, []);

  const startCapture = async () => {
    setStatus("capturing");
    setIsCapturing(true);
    setMessage("🔄 Capturing frames...");
    speak("Starting capture");

    try {
      for (let i = 0; i < totalNeeded; i++) {
        const frame = await captureFrame();
        await bufferFrame(frame.data);
        await checkBuffer();
        setMessage(`📸 Captured ${i + 1}/${totalNeeded} frames`);
        await new Promise((resolve) => setTimeout(resolve, captureSpeed));
      }

      setStatus("complete");
      setMessage("✅ Capture complete! Enter a label and save.");
      speak("Capture complete");
    } catch (err) {
      setMessage(`❌ Error: ${err.message}`);
      speak("Error during capture");
      setStatus("idle");
    } finally {
      setIsCapturing(false);
    }
  };

  const handleSave = async () => {
    if (!label.trim()) {
      setMessage("❌ Please enter a label");
      return;
    }

    try {
      setMessage("💾 Saving...");
      const res = await saveLabel(label);
      setMessage(`✅ Saved as "${label}"`);
      speak(`Saved gesture ${label}`);
      setLabel("");
      setStatus("idle");
      await checkBuffer();
    } catch (err) {
      setMessage(`❌ Error: ${err.message}`);
      speak("Error saving gesture");
    }
  };

  const progress = (buffered / totalNeeded) * 100;

  return (
    <section className="page capture-page">
      <h2>📹 Capture Gesture Data</h2>

      <div className="capture-container">
        <div className="progress-section">
          <h3>Capture Progress</h3>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
          </div>
          <p className="progress-text">
            {buffered} / {totalNeeded} frames
          </p>
        </div>

        <div className="controls">
          <div className="form-group">
            <label>Capture Speed (ms between frames)</label>
            <input
              type="range"
              min="50"
              max="500"
              step="50"
              value={captureSpeed}
              onChange={(e) => setCaptureSpeed(Number(e.target.value))}
              disabled={isCapturing}
            />
            <p className="help-text">{captureSpeed}ms</p>
          </div>

          <button
            className="btn btn-primary btn-large"
            onClick={startCapture}
            disabled={isCapturing || status === "complete"}
          >
            {isCapturing ? "🔄 Capturing..." : "🎬 Start Capture"}
          </button>
        </div>

        {message && <p className="status-message">{message}</p>}

        {status === "complete" && (
          <div className="save-section">
            <h3>Save Gesture Label</h3>
            <div className="form-group">
              <label>Enter gesture name/label</label>
              <input
                type="text"
                placeholder="e.g., HELLO, GOODBYE, OK, YES..."
                value={label}
                onChange={(e) => setLabel(e.target.value)}
              />
            </div>
            <button className="btn btn-primary btn-large" onClick={handleSave}>
              💾 Save Gesture
            </button>
          </div>
        )}
      </div>

      <div className="info-box">
        <h4>💡 Tips:</h4>
        <ul>
          <li>Make consistent, controlled movements</li>
          <li>Capture 30 frames at a steady pace</li>
          <li>Give each gesture a clear, memorable label</li>
          <li>Collect 5-10 samples per gesture for best results</li>
        </ul>
      </div>
    </section>
  );
}

export default CapturePage;
