import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./pages.css";
import "./camera.css";
import "./hand-detection.css";

function CameraCaptureePage({ speak }) {
  const [isCapturing, setIsCapturing] = useState(false);
  const [gestureLabel, setGestureLabel] = useState("");
  const [gestureText, setGestureText] = useState("");
  const [frameCount, setFrameCount] = useState(0);
  const [status, setStatus] = useState("");
  const [preview, setPreview] = useState(null);
  const [handDetection, setHandDetection] = useState(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    // Start preview with delay
    const timer = setTimeout(() => {
      startPreview();
    }, 1000);
    
    return () => {
      clearTimeout(timer);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const startPreview = () => {
    intervalRef.current = setInterval(async () => {
      if (!isCapturing) {
        try {
          // Get regular frame
          const frameResponse = await axios.get("/api/capture-frame", {
            timeout: 5000,
            headers: {
              'Content-Type': 'application/json'
            }
          });
          
          // Get hand detection
          const handResponse = await axios.get("/api/hand-detection", {
            timeout: 5000
          });
          
          setPreview(handResponse.data.annotated_frame);
          setHandDetection(handResponse.data);
          setStatus(`Camera connected - ${handResponse.data.hand_count} hands detected`);
        } catch (error) {
          console.error("Preview error:", error);
          setStatus("Camera connection error: " + error.message);
          setPreview(null);
        }
      }
    }, 500);
  };

  const startCapture = async () => {
    if (!gestureLabel.trim() || !gestureText.trim()) {
      speak("Please enter both gesture label and text");
      return;
    }

    try {
      setIsCapturing(true);
      setFrameCount(0);
      setStatus("Starting capture...");

      // Start capture session
      await axios.post("/api/start-gesture-capture", {
        name: gestureLabel.trim(),
        text: gestureText.trim(),
        target_frames: 30
      });

      setStatus("Capturing frames... Get ready!");
      speak("Starting capture in 3 seconds. Get ready!");

      // Wait 3 seconds then start capturing
      setTimeout(async () => {
        for (let i = 0; i < 30; i++) {
          try {
            const response = await axios.post("/api/capture-frame");
            setFrameCount(response.data.frames_captured);
            setStatus(`Capturing frame ${response.data.frames_captured}/${response.data.target_frames}`);
            
            if (response.data.complete) {
              break;
            }
            
            // Wait between frames
            await new Promise(resolve => setTimeout(resolve, 200));
          } catch (error) {
            console.error("Frame capture error:", error);
            setStatus("Error capturing frame");
            break;
          }
        }

        // Save sequence
        try {
          await axios.post("/api/save-gesture");
          setStatus("Sequence saved successfully!");
          speak(`Gesture ${gestureLabel} captured successfully`);
        } catch (error) {
          setStatus("Error saving sequence");
          speak("Error saving gesture");
        }

        setIsCapturing(false);
      }, 3000);

    } catch (error) {
      console.error("Capture error:", error);
      setStatus("Error starting capture");
      setIsCapturing(false);
      speak("Error starting capture");
    }
  };

  return (
    <section className="page capture-page">
      <h2>📹 Camera Gesture Capture</h2>
      <p>Capture 30 frames of hand and face gestures for training</p>

      <div className="capture-container">
        <div className="camera-preview">
          <h3>Camera Preview</h3>
          {preview ? (
            <img src={preview} alt="Camera preview" className="preview-image" />
          ) : (
            <div className="no-preview">No camera feed</div>
          )}
        </div>

        <div className="capture-controls">
          <div className="input-group">
            <label>Gesture Label:</label>
            <input
              type="text"
              value={gestureLabel}
              onChange={(e) => setGestureLabel(e.target.value)}
              placeholder="e.g., HELLO, THANK_YOU"
              disabled={isCapturing}
            />
          </div>

          <div className="input-group">
            <label>Text to Speak:</label>
            <input
              type="text"
              value={gestureText}
              onChange={(e) => setGestureText(e.target.value)}
              placeholder="e.g., Hello there, Thank you"
              disabled={isCapturing}
            />
          </div>

          <button
            className={`btn ${isCapturing ? "btn-secondary" : "btn-primary"}`}
            onClick={startCapture}
            disabled={isCapturing || !gestureLabel.trim() || !gestureText.trim()}
          >
            {isCapturing ? "Capturing..." : "Start Capture"}
          </button>

          {isCapturing && (
            <div className="progress-info">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${(frameCount / 30) * 100}%` }}
                ></div>
              </div>
              <p>Frames: {frameCount}/30</p>
            </div>
          )}

          <div className="status-display">
            <p>{status}</p>
            {handDetection && (
              <div className="hand-info">
                <p>Hands detected: {handDetection.hand_count}</p>
                {handDetection.detailed_landmarks && handDetection.detailed_landmarks.map((hand, index) => (
                  <p key={index}>{hand.handedness} Hand: {(hand.confidence * 100).toFixed(1)}% confidence</p>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="instructions">
        <h3>Instructions:</h3>
        <ol>
          <li>Enter a unique gesture label (e.g., HELLO, GOODBYE)</li>
          <li>Enter the text you want spoken when this gesture is recognized</li>
          <li>Position yourself in front of the camera</li>
          <li>Click "Start Capture" and perform your gesture</li>
          <li>Hold the gesture steady for about 6 seconds</li>
          <li>Repeat for different gestures</li>
        </ol>
      </div>
    </section>
  );
}

export default CameraCaptureePage;