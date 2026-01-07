import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './GestureManager.css';

const GestureManagerPage = () => {
  const [gestures, setGestures] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [captureProgress, setCaptureProgress] = useState({ frames: 0, target: 30 });
  const [newGesture, setNewGesture] = useState({ name: '', text: '', targetFrames: 30 });
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [cameraFeed, setCameraFeed] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const cameraIntervalRef = useRef(null);

  useEffect(() => {
    loadGestures();
    return () => {
      if (cameraIntervalRef.current) {
        clearInterval(cameraIntervalRef.current);
      }
    };
  }, []);

  const loadGestures = async () => {
    try {
      const response = await axios.get('/api/list-gestures');
      setGestures(response.data.gestures);
    } catch (err) {
      setError('Failed to load gestures');
    }
  };

  const startCameraPreview = () => {
    setShowCamera(true);
    cameraIntervalRef.current = setInterval(async () => {
      try {
        const response = await axios.get('/api/hand-detection');
        setCameraFeed(response.data);
      } catch (err) {
        console.error('Camera feed error:', err);
      }
    }, 200);
  };

  const stopCameraPreview = () => {
    setShowCamera(false);
    setCameraFeed(null);
    if (cameraIntervalRef.current) {
      clearInterval(cameraIntervalRef.current);
      cameraIntervalRef.current = null;
    }
  };

  const startGestureCapture = async () => {
    if (!newGesture.name.trim() || !newGesture.text.trim()) {
      setError('Please enter both gesture name and text');
      return;
    }

    try {
      setError('');
      setSuccess('');
      
      await axios.post('/api/start-gesture-capture', {
        name: newGesture.name,
        text: newGesture.text,
        target_frames: newGesture.targetFrames
      });

      setIsCapturing(true);
      setCaptureProgress({ frames: 0, target: newGesture.targetFrames });
      setSuccess(`Started capturing "${newGesture.name}". Perform the gesture now!`);
      
      // Start camera preview if not already running
      if (!showCamera) {
        startCameraPreview();
      }
      
      // Start automatic frame capture
      captureFrames();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start capture');
    }
  };

  const captureFrames = async () => {
    const captureInterval = setInterval(async () => {
      try {
        const response = await axios.post('/api/capture-frame');
        setCaptureProgress({
          frames: response.data.frames_captured,
          target: response.data.target_frames
        });

        if (response.data.complete) {
          clearInterval(captureInterval);
          await saveGesture();
        }
      } catch (err) {
        clearInterval(captureInterval);
        setError('Failed to capture frame');
        setIsCapturing(false);
      }
    }, 100); // Capture every 100ms
  };

  const saveGesture = async () => {
    try {
      await axios.post('/api/save-gesture');
      setIsCapturing(false);
      setSuccess(`Gesture "${newGesture.name}" saved successfully!`);
      setNewGesture({ name: '', text: '', targetFrames: 30 });
      setCaptureProgress({ frames: 0, target: 30 });
      loadGestures();
    } catch (err) {
      setError('Failed to save gesture');
      setIsCapturing(false);
    }
  };

  const deleteGesture = async (gestureName) => {
    if (!window.confirm(`Are you sure you want to delete "${gestureName}"?`)) {
      return;
    }

    try {
      await axios.delete(`/api/delete-gesture/${gestureName}`);
      setSuccess(`Gesture "${gestureName}" deleted successfully`);
      loadGestures();
    } catch (err) {
      setError('Failed to delete gesture');
    }
  };

  const trainModel = async () => {
    if (gestures.length < 2) {
      setError('Need at least 2 gestures to train the model');
      return;
    }

    const totalSequences = gestures.reduce((sum, g) => sum + g.sequences, 0);
    if (totalSequences < 10) {
      setError('Need at least 10 total sequences to train the model');
      return;
    }

    setIsTraining(true);
    setError('');
    setTrainingResult(null);

    try {
      const response = await axios.post('/api/train-advanced-model');
      setTrainingResult(response.data);
      setSuccess('Model trained successfully!');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to train model');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="gesture-manager-page">
      <div className="container mx-auto p-6">
        <h1 className="text-3xl font-bold mb-6 text-center">
          Advanced Gesture Manager
        </h1>

        {/* Status Messages */}
        {error && (
          <div className="error-message bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}
        
        {success && (
          <div className="success-message bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
            {success}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Camera Preview */}
          <div className="camera-preview-section">
            <h2 className="text-2xl font-semibold mb-4">Camera Preview</h2>
            
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="mb-4">
                {!showCamera ? (
                  <button
                    onClick={startCameraPreview}
                    className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md font-semibold w-full"
                  >
                    Start Camera Preview
                  </button>
                ) : (
                  <button
                    onClick={stopCameraPreview}
                    className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-md font-semibold w-full"
                  >
                    Stop Camera Preview
                  </button>
                )}
              </div>
              
              <div className="camera-container bg-black rounded-lg overflow-hidden" style={{minHeight: '300px'}}>
                {cameraFeed && cameraFeed.annotated_frame ? (
                  <img
                    src={cameraFeed.annotated_frame}
                    alt="Camera Preview"
                    className="w-full h-auto"
                    style={{ maxHeight: '300px', objectFit: 'contain' }}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-white" style={{minHeight: '300px'}}>
                    {showCamera ? 'Loading camera...' : 'Camera preview stopped'}
                  </div>
                )}
              </div>
              
              {cameraFeed && (
                <div className="mt-4 p-3 bg-gray-50 rounded">
                  <p className="text-sm"><strong>Hands Detected:</strong> {cameraFeed.hand_count}</p>
                  {cameraFeed.detailed_landmarks && cameraFeed.detailed_landmarks.map((hand, idx) => (
                    <p key={idx} className="text-xs text-gray-600">
                      {hand.handedness}: {(hand.confidence * 100).toFixed(1)}% confidence
                    </p>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Add New Gesture */}
          <div className="add-gesture-section">
            <h2 className="text-2xl font-semibold mb-4">Add New Gesture</h2>
            
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Gesture Name
                </label>
                <input
                  type="text"
                  value={newGesture.name}
                  onChange={(e) => setNewGesture({...newGesture, name: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., Hello, Thank You, Peace"
                  disabled={isCapturing}
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Gesture Text (What it means)
                </label>
                <input
                  type="text"
                  value={newGesture.text}
                  onChange={(e) => setNewGesture({...newGesture, text: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., Hello there!, Thank you very much, Peace and love"
                  disabled={isCapturing}
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Frames
                </label>
                <input
                  type="number"
                  value={newGesture.targetFrames}
                  onChange={(e) => setNewGesture({...newGesture, targetFrames: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  min="10"
                  max="60"
                  disabled={isCapturing}
                />
              </div>

              {!isCapturing ? (
                <button
                  onClick={startGestureCapture}
                  className="w-full bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md font-semibold"
                >
                  Start Capturing Gesture
                </button>
              ) : (
                <div className="capture-progress">
                  <div className="mb-2">
                    <span className="text-sm font-medium text-gray-700">
                      Capturing: {captureProgress.frames} / {captureProgress.target} frames
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                      style={{ width: `${(captureProgress.frames / captureProgress.target) * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-gray-600 mt-2">
                    Perform the gesture now! Keep your hand visible to the camera.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Existing Gestures */}
          <div className="gestures-list-section">
            <h2 className="text-2xl font-semibold mb-4">
              Existing Gestures ({gestures.length})
            </h2>
            
            <div className="bg-white rounded-lg shadow-md max-h-96 overflow-y-auto">
              {gestures.length === 0 ? (
                <div className="p-6 text-center text-gray-500">
                  No gestures created yet. Add your first gesture!
                </div>
              ) : (
                <div className="divide-y divide-gray-200">
                  {gestures.map((gesture, index) => (
                    <div key={index} className="p-4">
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <h3 className="font-semibold text-lg">{gesture.name}</h3>
                          <p className="text-gray-600 text-sm mb-2">"{gesture.text}"</p>
                          <span className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                            {gesture.sequences} sequences
                          </span>
                        </div>
                        <button
                          onClick={() => deleteGesture(gesture.name)}
                          className="ml-4 bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded text-sm"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Model Training Section */}
        <div className="model-training-section mt-8">
          <h2 className="text-2xl font-semibold mb-4">CNN+LSTM Model Training</h2>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="mb-4">
              <p className="text-gray-700 mb-2">
                <strong>Training Requirements:</strong>
              </p>
              <ul className="list-disc list-inside text-sm text-gray-600 mb-4">
                <li>At least 2 different gestures</li>
                <li>At least 10 total sequences across all gestures</li>
                <li>More sequences = better accuracy</li>
                <li>Each gesture should have multiple sequences for variety</li>
              </ul>
              
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="bg-gray-50 p-3 rounded">
                  <span className="text-sm font-medium">Total Gestures:</span>
                  <span className="ml-2 text-lg font-bold text-blue-600">{gestures.length}</span>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <span className="text-sm font-medium">Total Sequences:</span>
                  <span className="ml-2 text-lg font-bold text-green-600">
                    {gestures.reduce((sum, g) => sum + g.sequences, 0)}
                  </span>
                </div>
              </div>
            </div>

            {!isTraining ? (
              <button
                onClick={trainModel}
                disabled={gestures.length < 2 || gestures.reduce((sum, g) => sum + g.sequences, 0) < 10}
                className="bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-6 py-3 rounded-md font-semibold"
              >
                Train CNN+LSTM Model
              </button>
            ) : (
              <div className="training-progress">
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-green-500 mr-3"></div>
                  <span>Training model... This may take a few minutes.</span>
                </div>
              </div>
            )}

            {trainingResult && (
              <div className="training-result mt-4 p-4 bg-green-50 border border-green-200 rounded">
                <h3 className="font-semibold text-green-800 mb-2">Training Complete!</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Classes:</span> {trainingResult.classes.join(', ')}
                  </div>
                  <div>
                    <span className="font-medium">Samples:</span> {trainingResult.samples}
                  </div>
                  <div>
                    <span className="font-medium">Training Accuracy:</span> {(trainingResult.accuracy * 100).toFixed(1)}%
                  </div>
                  <div>
                    <span className="font-medium">Validation Accuracy:</span> {(trainingResult.val_accuracy * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Instructions */}
        <div className="instructions-section mt-8">
          <h2 className="text-2xl font-semibold mb-4">How to Use</h2>
          <div className="bg-blue-50 p-6 rounded-lg">
            <ol className="list-decimal list-inside space-y-2 text-sm">
              <li><strong>Add Gestures:</strong> Create multiple gestures with descriptive names and meaningful text</li>
              <li><strong>Capture Sequences:</strong> For each gesture, capture multiple sequences (3-5 recommended) with slight variations</li>
              <li><strong>Train Model:</strong> Once you have enough data, train the CNN+LSTM model</li>
              <li><strong>Test Recognition:</strong> Use the Live Recognition page to test your trained model</li>
              <li><strong>Improve Accuracy:</strong> Add more sequences for gestures that aren't recognized well</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GestureManagerPage;