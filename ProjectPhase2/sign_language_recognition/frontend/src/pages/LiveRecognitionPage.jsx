import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const LiveRecognitionPage = () => {
  const [handData, setHandData] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [error, setError] = useState('');
  const [settings, setSettings] = useState({
    predictionFrames: 30,
    confidenceThreshold: 0.7,
    autoPredict: false,
    autoSpeak: true,
    speechRate: 1.0,
    speechPitch: 1.0,
    speechVolume: 1.0
  });
  const intervalRef = useRef(null);
  const predictionIntervalRef = useRef(null);

  useEffect(() => {
    return () => {
      stopDetection();
      stopAutoPrediction();
    };
  }, []);

  const startDetection = () => {
    setIsDetecting(true);
    setError('');
    
    intervalRef.current = setInterval(async () => {
      try {
        const response = await axios.get('/api/hand-detection');
        setHandData(response.data);
      } catch (err) {
        setError('Hand detection failed: ' + (err.response?.data?.detail || err.message));
        console.error('Hand detection error:', err);
      }
    }, 100); // Update every 100ms for smooth display

    // Start auto-prediction if enabled
    if (settings.autoPredict) {
      startAutoPrediction();
    }
  };

  const stopDetection = () => {
    setIsDetecting(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    stopAutoPrediction();
  };

  const startAutoPrediction = () => {
    predictionIntervalRef.current = setInterval(() => {
      if (!isPredicting) {
        predictGesture();
      }
    }, 2000); // Predict every 2 seconds
  };

  const stopAutoPrediction = () => {
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
    }
  };

  const predictGesture = async () => {
    if (isPredicting) return;

    setIsPredicting(true);
    setError('');

    try {
      const response = await axios.post('/api/predict-gesture', {
        frames: settings.predictionFrames
      });

      const newPrediction = {
        ...response.data,
        timestamp: new Date().toLocaleTimeString(),
        id: Date.now()
      };

      setPrediction(newPrediction);
      
      // Auto-speak if enabled and gesture is recognized
      if (settings.autoSpeak && 
          newPrediction.status === 'recognized' && 
          newPrediction.text && 
          newPrediction.text.trim() !== '') {
        speakText(newPrediction.text);
      }
      
      // Add to history only if gesture is recognized and has text
      if (newPrediction.status === 'recognized' && 
          newPrediction.confidence >= settings.confidenceThreshold &&
          newPrediction.text && newPrediction.text.trim() !== '') {
        setPredictionHistory(prev => [newPrediction, ...prev.slice(0, 9)]); // Keep last 10
      }

    } catch (err) {
      setError('Prediction failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setIsPredicting(false);
    }
  };

  const clearHistory = () => {
    setPredictionHistory([]);
  };

  const speakText = (text) => {
    if ('speechSynthesis' in window) {
      // Cancel any ongoing speech
      window.speechSynthesis.cancel();
      
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = settings.speechRate;
      utterance.pitch = settings.speechPitch;
      utterance.volume = settings.speechVolume;
      
      window.speechSynthesis.speak(utterance);
    } else {
      setError('Text-to-speech not supported in this browser');
    }
  };

  const stopSpeaking = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  };

  const renderFingerStates = (fingerStates) => {
    if (!fingerStates) return null;

    const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
    
    return (
      <div className="finger-states grid grid-cols-5 gap-2 mt-4">
        {fingers.map(finger => {
          const state = fingerStates[finger];
          if (!state) return null;

          const isExtended = state.state === 'extended';
          const bendRatio = state.bend_ratio || 0;

          return (
            <div key={finger} className="finger-state text-center">
              <div className={`finger-icon w-12 h-16 mx-auto mb-1 rounded ${
                isExtended ? 'bg-green-200 border-green-500' : 'bg-red-200 border-red-500'
              } border-2 flex items-center justify-center`}>
                <div className={`w-2 h-8 rounded ${
                  isExtended ? 'bg-green-500' : 'bg-red-500'
                }`} style={{
                  height: `${Math.max(20, 80 - (bendRatio * 60))}%`
                }}></div>
              </div>
              <div className="text-xs font-medium capitalize">{finger}</div>
              <div className={`text-xs ${isExtended ? 'text-green-600' : 'text-red-600'}`}>
                {state.state}
              </div>
              <div className="text-xs text-gray-500">
                {state.angle?.toFixed(0)}°
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const renderRotationData = (rotation) => {
    if (!rotation) return null;

    return (
      <div className="rotation-data mt-4 p-3 bg-blue-50 rounded-lg">
        <h4 className="font-semibold mb-2 text-blue-800">Hand Rotation & Direction:</h4>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="text-center">
            <div className="font-medium text-blue-700">Roll</div>
            <div className="text-lg font-bold">{rotation.roll?.toFixed(1)}°</div>
          </div>
          <div className="text-center">
            <div className="font-medium text-blue-700">Pitch</div>
            <div className="text-lg font-bold">{rotation.pitch?.toFixed(1)}°</div>
          </div>
          <div className="text-center">
            <div className="font-medium text-blue-700">Yaw</div>
            <div className="text-lg font-bold">{rotation.yaw?.toFixed(1)}°</div>
          </div>
        </div>
        <div className="mt-2 text-xs text-blue-600">
          Direction: [{rotation.direction_vector?.map(v => v.toFixed(2)).join(', ')}]
        </div>
      </div>
    );
  };

  const renderVelocityData = (velocity) => {
    if (!velocity) return null;

    return (
      <div className="velocity-data mt-4 p-3 bg-green-50 rounded-lg">
        <h4 className="font-semibold mb-2 text-green-800">Hand Movement:</h4>
        <div className="text-center">
          <div className="font-medium text-green-700">Speed</div>
          <div className="text-lg font-bold">{velocity.speed?.toFixed(3)}</div>
        </div>
        <div className="mt-2 text-xs text-green-600">
          Velocity: [{velocity.velocity?.map(v => v.toFixed(3)).join(', ')}]
        </div>
      </div>
    );
  };

  const renderPredictionConfidence = (allPredictions) => {
    if (!allPredictions) return null;

    const sortedPredictions = Object.entries(allPredictions)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5); // Top 5 predictions

    return (
      <div className="prediction-confidence mt-4">
        <h4 className="font-semibold mb-2">All Predictions:</h4>
        <div className="space-y-2">
          {sortedPredictions.map(([gesture, confidence], index) => (
            <div key={gesture} className="flex items-center">
              <span className="w-20 text-sm font-medium">{gesture}:</span>
              <div className="flex-1 mx-2 bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    index === 0 ? 'bg-blue-500' : 'bg-gray-400'
                  }`}
                  style={{ width: `${confidence * 100}%` }}
                ></div>
              </div>
              <span className="text-sm text-gray-600">{(confidence * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="live-recognition-page">
      <div className="container mx-auto p-6">
        <h1 className="text-3xl font-bold mb-6 text-center">
          Live Sign Language Recognition
        </h1>

        {/* Controls */}
        <div className="controls-section mb-6">
          <div className="bg-white p-4 rounded-lg shadow-md">
            <div className="flex flex-wrap items-center gap-4 mb-4">
              {!isDetecting ? (
                <button
                  onClick={startDetection}
                  className="bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded-lg font-semibold"
                >
                  Start Detection
                </button>
              ) : (
                <button
                  onClick={stopDetection}
                  className="bg-red-500 hover:bg-red-600 text-white px-6 py-2 rounded-lg font-semibold"
                >
                  Stop Detection
                </button>
              )}

              <button
                onClick={predictGesture}
                disabled={!isDetecting || isPredicting}
                className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg font-semibold"
              >
                {isPredicting ? 'Predicting...' : 'Predict Now'}
              </button>

              <button
                onClick={clearHistory}
                className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg"
              >
                Clear History
              </button>
            </div>

            {/* Settings */}
            <div className="settings grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Prediction Frames
                </label>
                <input
                  type="number"
                  value={settings.predictionFrames}
                  onChange={(e) => setSettings({...settings, predictionFrames: parseInt(e.target.value)})}
                  className="w-full px-3 py-1 border border-gray-300 rounded text-sm"
                  min="10"
                  max="60"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Confidence Threshold
                </label>
                <input
                  type="number"
                  value={settings.confidenceThreshold}
                  onChange={(e) => setSettings({...settings, confidenceThreshold: parseFloat(e.target.value)})}
                  className="w-full px-3 py-1 border border-gray-300 rounded text-sm"
                  min="0"
                  max="1"
                  step="0.1"
                />
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  checked={settings.autoPredict}
                  onChange={(e) => {
                    const newAutoPredict = e.target.checked;
                    setSettings({...settings, autoPredict: newAutoPredict});
                    
                    if (isDetecting) {
                      if (newAutoPredict) {
                        startAutoPrediction();
                      } else {
                        stopAutoPrediction();
                      }
                    }
                  }}
                  className="mr-2"
                />
                <label className="text-sm font-medium text-gray-700">
                  Auto Predict
                </label>
              </div>
            </div>

            {/* Speech Settings */}
            <div className="speech-settings mt-4 p-3 bg-gray-50 rounded">
              <h4 className="font-semibold mb-3">🔊 Text-to-Speech Settings</h4>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    checked={settings.autoSpeak}
                    onChange={(e) => setSettings({...settings, autoSpeak: e.target.checked})}
                    className="mr-2"
                  />
                  <label className="text-sm font-medium text-gray-700">
                    Auto Speak
                  </label>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Speed: {settings.speechRate}x
                  </label>
                  <input
                    type="range"
                    value={settings.speechRate}
                    onChange={(e) => setSettings({...settings, speechRate: parseFloat(e.target.value)})}
                    className="w-full"
                    min="0.5"
                    max="2"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Pitch: {settings.speechPitch}x
                  </label>
                  <input
                    type="range"
                    value={settings.speechPitch}
                    onChange={(e) => setSettings({...settings, speechPitch: parseFloat(e.target.value)})}
                    className="w-full"
                    min="0.5"
                    max="2"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Volume: {Math.round(settings.speechVolume * 100)}%
                  </label>
                  <input
                    type="range"
                    value={settings.speechVolume}
                    onChange={(e) => setSettings({...settings, speechVolume: parseFloat(e.target.value)})}
                    className="w-full"
                    min="0"
                    max="1"
                    step="0.1"
                  />
                </div>
              </div>
              <div className="mt-2 flex gap-2">
                <button
                  onClick={() => speakText('Test speech synthesis')}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm"
                >
                  🔊 Test Voice
                </button>
                <button
                  onClick={stopSpeaking}
                  className="bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded text-sm"
                >
                  🔇 Stop
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="error-message bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Camera Feed and Hand Analysis */}
          <div className="xl:col-span-2">
            <div className="camera-section bg-white rounded-lg shadow-md p-4">
              <h3 className="text-xl font-semibold mb-4">Live Camera Feed</h3>
              
              {handData ? (
                <>
                  <div className="camera-container bg-black rounded-lg overflow-hidden mb-4">
                    <img
                      src={handData.annotated_frame}
                      alt="Hand Detection"
                      className="w-full h-auto"
                      style={{ maxHeight: '400px', objectFit: 'contain' }}
                    />
                  </div>

                  <div className="hand-info grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="detection-stats bg-gray-50 p-3 rounded">
                      <h4 className="font-semibold mb-2">Detection Stats</h4>
                      <p><strong>Hands Detected:</strong> {handData.hand_count}</p>
                      <p><strong>Timestamp:</strong> {new Date(handData.timestamp).toLocaleTimeString()}</p>
                    </div>

                    {handData.detailed_landmarks && handData.detailed_landmarks.length > 0 && (
                      <div className="hand-details bg-gray-50 p-3 rounded">
                        <h4 className="font-semibold mb-2">Hand Details</h4>
                        {handData.detailed_landmarks.map((hand, idx) => (
                          <div key={idx} className="mb-2">
                            <p><strong>{hand.handedness} Hand</strong></p>
                            <p>Confidence: {(hand.confidence * 100).toFixed(1)}%</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Finger States Visualization */}
                  {handData.detailed_landmarks && handData.detailed_landmarks.length > 0 && (
                    <div className="finger-analysis mt-4">
                      <h4 className="font-semibold mb-2">Finger Analysis</h4>
                      {handData.detailed_landmarks.map((hand, idx) => (
                        <div key={idx} className="hand-analysis mb-4 p-3 bg-gray-50 rounded">
                          <div className="flex justify-between items-center mb-2">
                            <h5 className="font-medium">{hand.handedness} Hand</h5>
                            <div className={`px-2 py-1 rounded text-xs font-medium ${
                              hand.handedness === 'Left' ? 'bg-blue-100 text-blue-800' :
                              hand.handedness === 'Right' ? 'bg-green-100 text-green-800' :
                              'bg-gray-100 text-gray-800'
                            }`}>
                              {hand.handedness} ({(hand.confidence * 100).toFixed(0)}%)
                            </div>
                          </div>
                          {renderFingerStates(hand.finger_states)}
                          {renderRotationData(hand.rotation)}
                          {renderVelocityData(hand.velocity)}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div className="no-feed bg-gray-100 h-64 flex items-center justify-center rounded-lg">
                  <p className="text-gray-500">
                    {isDetecting ? 'Loading camera feed...' : 'Start detection to see camera feed'}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Prediction Results */}
          <div className="prediction-section">
            <div className="bg-white rounded-lg shadow-md p-4">
              <h3 className="text-xl font-semibold mb-4">Recognition Results</h3>

              {/* Current Prediction */}
              {prediction && (
                <div className={`current-prediction mb-6 p-4 border rounded-lg ${
                  prediction.status === 'recognized' ? 'bg-blue-50 border-blue-200' :
                  prediction.status === 'no_hands' ? 'bg-gray-50 border-gray-200' :
                  'bg-yellow-50 border-yellow-200'
                }`}>
                  <h4 className={`font-semibold mb-2 ${
                    prediction.status === 'recognized' ? 'text-blue-800' :
                    prediction.status === 'no_hands' ? 'text-gray-600' :
                    'text-yellow-700'
                  }`}>Latest Prediction</h4>
                  <div className="prediction-result">
                    <div className="flex justify-between items-start mb-2">
                      <div className={`gesture-name text-2xl font-bold ${
                        prediction.status === 'recognized' ? 'text-blue-900' :
                        prediction.status === 'no_hands' ? 'text-gray-500' :
                        'text-yellow-800'
                      }`}>
                        {prediction.gesture}
                      </div>
                      {prediction.status === 'recognized' && (
                        <div className="text-sm text-green-600 font-medium">
                          🔊 Auto-speaking...
                        </div>
                      )}
                    </div>
                    {prediction.text && prediction.text.trim() !== '' && (
                      <div className="gesture-text text-lg text-gray-700 mb-2 italic">
                        "{prediction.text}"
                      </div>
                    )}
                    <div className="confidence">
                      <span className="text-sm font-medium">Confidence: </span>
                      <span className={`font-bold ${
                        prediction.confidence >= 0.8 ? 'text-green-600' :
                        prediction.confidence >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="timestamp text-xs text-gray-500 mt-1">
                      {prediction.timestamp}
                    </div>
                  </div>

                  {renderPredictionConfidence(prediction.all_predictions)}
                </div>
              )}

              {/* Prediction History */}
              <div className="prediction-history">
                <h4 className="font-semibold mb-3">
                  Recent Predictions ({predictionHistory.length})
                </h4>
                
                {predictionHistory.length === 0 ? (
                  <p className="text-gray-500 text-center py-4">
                    No predictions yet. Start detecting and predicting gestures!
                  </p>
                ) : (
                  <div className="history-list space-y-2 max-h-64 overflow-y-auto">
                    {predictionHistory.map((pred) => (
                      <div key={pred.id} className="history-item p-3 bg-gray-50 rounded border">
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <div className="font-semibold">{pred.gesture}</div>
                            <div className="text-sm text-gray-600">"{pred.text}"</div>
                            <div className="text-xs text-gray-500">{pred.timestamp}</div>
                          </div>
                          <div className={`confidence-badge px-2 py-1 rounded text-xs font-medium ${
                            pred.confidence >= 0.8 ? 'bg-green-100 text-green-800' :
                            pred.confidence >= 0.6 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'
                          }`}>
                            {(pred.confidence * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="instructions mt-6 bg-blue-50 p-4 rounded-lg">
          <h3 className="font-semibold mb-2">How to Use Live Recognition</h3>
          <ol className="list-decimal list-inside text-sm space-y-1">
            <li>Make sure you have trained a model in the Gesture Manager</li>
            <li>Start detection to see your hands and finger bend analysis</li>
            <li>Perform a gesture and click "Predict Now" or enable auto-prediction</li>
            <li>Green fingers = extended, Red fingers = bent</li>
            <li>Higher confidence scores indicate more accurate predictions</li>
          </ol>
        </div>
      </div>
    </div>
  );
};

export default LiveRecognitionPage;