import React, { useState, useEffect, useRef } from 'react';

const MediaPipeHandsPage = () => {
  const [handData, setHandData] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState('');
  const intervalRef = useRef(null);
  const [showFingerDetails, setShowFingerDetails] = useState(true);

  const startDetection = () => {
    setIsDetecting(true);
    setError('');
    
    intervalRef.current = setInterval(async () => {
      try {
        const response = await fetch('/api/hand-detection');
        if (response.ok) {
          const data = await response.json();
          setHandData(data);
        } else {
          throw new Error('Failed to fetch hand detection data');
        }
      } catch (err) {
        setError(err.message);
        console.error('Hand detection error:', err);
      }
    }, 100); // Update every 100ms for smooth display
  };

  const stopDetection = () => {
    setIsDetecting(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const renderFingerAnalysis = (landmarks) => {
    if (!landmarks || landmarks.length === 0) return null;

    return landmarks.map((hand, handIndex) => (
      <div key={handIndex} className="hand-analysis mb-6">
        <h4 className="font-semibold text-lg mb-3">
          {hand.handedness} Hand (Confidence: {(hand.confidence * 100).toFixed(1)}%)
        </h4>
        
        {/* Finger States */}
        {hand.finger_states && (
          <div className="finger-states mb-4">
            <h5 className="font-medium mb-2">Finger States:</h5>
            <div className="grid grid-cols-5 gap-2">
              {Object.entries(hand.finger_states).map(([finger, state]) => (
                <div key={finger} className="finger-state text-center">
                  <div className={`w-12 h-16 mx-auto mb-1 rounded border-2 flex items-center justify-center ${
                    state.state === 'extended' ? 'bg-green-100 border-green-500' : 'bg-red-100 border-red-500'
                  }`}>
                    <div className={`w-2 rounded ${
                      state.state === 'extended' ? 'bg-green-500' : 'bg-red-500'
                    }`} style={{
                      height: `${Math.max(20, 80 - (state.bend_ratio * 60))}%`
                    }}></div>
                  </div>
                  <div className="text-xs font-medium capitalize">{finger}</div>
                  <div className={`text-xs ${
                    state.state === 'extended' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {state.state}
                  </div>
                  <div className="text-xs text-gray-500">
                    {state.angle?.toFixed(0)}°
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Finger Angles */}
        {hand.finger_angles && (
          <div className="finger-angles">
            <h5 className="font-medium mb-2">Finger Angles:</h5>
            <div className="grid grid-cols-2 gap-2 text-sm">
              {Object.entries(hand.finger_angles).map(([finger, angle]) => (
                <div key={finger} className="flex justify-between p-2 bg-white rounded">
                  <span className="capitalize font-medium">{finger}:</span>
                  <span>{angle.toFixed(1)}°</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    ));
  };

  const renderLandmarkInfo = (landmarks) => {
    if (!landmarks || landmarks.length === 0) return null;

    return landmarks.map((hand, handIndex) => (
      <div key={handIndex} className="hand-info">
        <h4>
          {hand.handedness} Hand (Confidence: {(hand.confidence * 100).toFixed(1)}%)
        </h4>
        <div className="landmarks-grid">
          {hand.landmarks.map((landmark, idx) => (
            <div key={idx} className="landmark-item">
              <span className="landmark-name">{landmark.name}</span>
              <span className="landmark-coords">
                ({landmark.x.toFixed(3)}, {landmark.y.toFixed(3)}, {landmark.z.toFixed(3)})
              </span>
            </div>
          ))}
        </div>
      </div>
    ));
  };

  return (
    <div className="mediapipe-hands-page">
      <div className="container mx-auto p-6">
        <h1 className="text-3xl font-bold mb-6 text-center">
          MediaPipe Hand Detection
        </h1>

        <div className="controls mb-6">
          <div className="text-center mb-4">
            {!isDetecting ? (
              <button
                onClick={startDetection}
                className="bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-lg text-lg font-semibold"
              >
                Start Advanced Hand Detection
              </button>
            ) : (
              <button
                onClick={stopDetection}
                className="bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-lg text-lg font-semibold"
              >
                Stop Detection
              </button>
            )}
          </div>
          
          <div className="flex justify-center">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showFingerDetails}
                onChange={(e) => setShowFingerDetails(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm font-medium">Show Finger Bend Analysis</span>
            </label>
          </div>
        </div>

        {error && (
          <div className="error-message bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            Error: {error}
          </div>
        )}

        {handData && (
          <div className="detection-results">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Camera Feed */}
              <div className="camera-section">
                <h3 className="text-xl font-semibold mb-4">Live Camera Feed</h3>
                <div className="camera-container bg-black rounded-lg overflow-hidden">
                  {handData.annotated_frame && (
                    <img
                      src={handData.annotated_frame}
                      alt="Hand Detection"
                      className="w-full h-auto"
                      style={{ maxHeight: '400px', objectFit: 'contain' }}
                    />
                  )}
                </div>
                <div className="camera-info mt-4 p-4 bg-gray-100 rounded-lg">
                  <p><strong>Hands Detected:</strong> {handData.hand_count}</p>
                  <p><strong>Timestamp:</strong> {new Date(handData.timestamp).toLocaleTimeString()}</p>
                  {showFingerDetails && handData.detailed_landmarks && (
                    <div className="finger-summary mt-2">
                      <p><strong>Finger Analysis:</strong> Available for {handData.detailed_landmarks.length} hand(s)</p>
                      {handData.detailed_landmarks.map((hand, idx) => (
                        <div key={idx} className="text-sm text-gray-600 ml-4">
                          {hand.handedness}: {Object.values(hand.finger_states || {}).filter(f => f.state === 'extended').length} extended, {Object.values(hand.finger_states || {}).filter(f => f.state === 'bent').length} bent
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Hand Analysis Data */}
              <div className="analysis-section">
                <h3 className="text-xl font-semibold mb-4">
                  {showFingerDetails ? 'Hand Analysis & Finger Bends' : 'Hand Landmarks (21 points per hand)'}
                </h3>
                <div className="analysis-container max-h-96 overflow-y-auto bg-gray-50 p-4 rounded-lg">
                  {handData.detailed_landmarks && handData.detailed_landmarks.length > 0 ? (
                    showFingerDetails ? renderFingerAnalysis(handData.detailed_landmarks) : renderLandmarkInfo(handData.detailed_landmarks)
                  ) : (
                    <p className="text-gray-500 text-center">No hands detected</p>
                  )}
                </div>
              </div>
            </div>

            {/* Landmark Visualization */}
            {handData.detailed_landmarks && handData.detailed_landmarks.length > 0 && (
              <div className="landmark-visualization mt-6">
                <h3 className="text-xl font-semibold mb-4">Landmark Positions</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {handData.detailed_landmarks.map((hand, handIndex) => (
                    <div key={handIndex} className="hand-visualization">
                      <h4 className="font-semibold mb-2">
                        {hand.handedness} Hand - {hand.landmarks.length} landmarks
                      </h4>
                      <div className="landmark-points bg-white border rounded-lg p-4">
                        <svg width="300" height="300" viewBox="0 0 300 300" className="border">
                          {/* Draw hand landmarks as circles */}
                          {hand.landmarks.map((landmark, idx) => (
                            <g key={idx}>
                              <circle
                                cx={landmark.x * 300}
                                cy={landmark.y * 300}
                                r={idx === 0 ? 6 : idx % 4 === 0 ? 5 : 3} // Larger for wrist and finger tips
                                fill={
                                  idx === 0 ? '#ff0000' : // Wrist - red
                                  [4, 8, 12, 16, 20].includes(idx) ? '#0000ff' : // Finger tips - blue
                                  '#00ff00' // Other joints - green
                                }
                                stroke="#000"
                                strokeWidth="1"
                              />
                              <text
                                x={landmark.x * 300 + 8}
                                y={landmark.y * 300 + 4}
                                fontSize="10"
                                fill="#333"
                              >
                                {idx}
                              </text>
                            </g>
                          ))}
                          
                          {/* Draw connections between landmarks */}
                          {/* Thumb */}
                          {[0, 1, 2, 3, 4].map((idx, i) => 
                            i < 4 && hand.landmarks[idx] && hand.landmarks[idx + 1] ? (
                              <line
                                key={`thumb-${i}`}
                                x1={hand.landmarks[idx].x * 300}
                                y1={hand.landmarks[idx].y * 300}
                                x2={hand.landmarks[idx + 1].x * 300}
                                y2={hand.landmarks[idx + 1].y * 300}
                                stroke="#666"
                                strokeWidth="2"
                              />
                            ) : null
                          )}
                          
                          {/* Index finger */}
                          {[0, 5, 6, 7, 8].map((idx, i) => 
                            i < 4 && hand.landmarks[idx] && hand.landmarks[i === 0 ? 5 : idx + 1] ? (
                              <line
                                key={`index-${i}`}
                                x1={hand.landmarks[idx].x * 300}
                                y1={hand.landmarks[idx].y * 300}
                                x2={hand.landmarks[i === 0 ? 5 : idx + 1].x * 300}
                                y2={hand.landmarks[i === 0 ? 5 : idx + 1].y * 300}
                                stroke="#666"
                                strokeWidth="2"
                              />
                            ) : null
                          )}
                          
                          {/* Middle finger */}
                          {[0, 9, 10, 11, 12].map((idx, i) => 
                            i < 4 && hand.landmarks[idx] && hand.landmarks[i === 0 ? 9 : idx + 1] ? (
                              <line
                                key={`middle-${i}`}
                                x1={hand.landmarks[idx].x * 300}
                                y1={hand.landmarks[idx].y * 300}
                                x2={hand.landmarks[i === 0 ? 9 : idx + 1].x * 300}
                                y2={hand.landmarks[i === 0 ? 9 : idx + 1].y * 300}
                                stroke="#666"
                                strokeWidth="2"
                              />
                            ) : null
                          )}
                          
                          {/* Ring finger */}
                          {[0, 13, 14, 15, 16].map((idx, i) => 
                            i < 4 && hand.landmarks[idx] && hand.landmarks[i === 0 ? 13 : idx + 1] ? (
                              <line
                                key={`ring-${i}`}
                                x1={hand.landmarks[idx].x * 300}
                                y1={hand.landmarks[idx].y * 300}
                                x2={hand.landmarks[i === 0 ? 13 : idx + 1].x * 300}
                                y2={hand.landmarks[i === 0 ? 13 : idx + 1].y * 300}
                                stroke="#666"
                                strokeWidth="2"
                              />
                            ) : null
                          )}
                          
                          {/* Pinky */}
                          {[0, 17, 18, 19, 20].map((idx, i) => 
                            i < 4 && hand.landmarks[idx] && hand.landmarks[i === 0 ? 17 : idx + 1] ? (
                              <line
                                key={`pinky-${i}`}
                                x1={hand.landmarks[idx].x * 300}
                                y1={hand.landmarks[idx].y * 300}
                                x2={hand.landmarks[i === 0 ? 17 : idx + 1].x * 300}
                                y2={hand.landmarks[i === 0 ? 17 : idx + 1].y * 300}
                                stroke="#666"
                                strokeWidth="2"
                              />
                            ) : null
                          )}
                        </svg>
                        <div className="legend mt-2 text-sm">
                          <span className="inline-block w-3 h-3 bg-red-500 mr-1"></span>Wrist
                          <span className="inline-block w-3 h-3 bg-blue-500 mr-1 ml-4"></span>Finger Tips
                          <span className="inline-block w-3 h-3 bg-green-500 mr-1 ml-4"></span>Joints
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {isDetecting && !handData && (
          <div className="loading text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <p className="mt-2 text-gray-600">Initializing hand detection...</p>
          </div>
        )}
      </div>

      <style jsx>{`
        .landmark-item {
          display: flex;
          justify-content: space-between;
          padding: 4px 8px;
          margin: 2px 0;
          background: white;
          border-radius: 4px;
          font-size: 12px;
        }
        
        .landmark-name {
          font-weight: bold;
          color: #333;
        }
        
        .landmark-coords {
          color: #666;
          font-family: monospace;
        }
        
        .hand-info {
          margin-bottom: 20px;
          padding: 16px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background: #f9f9f9;
        }
        
        .landmarks-grid {
          max-height: 200px;
          overflow-y: auto;
        }
      `}</style>
    </div>
  );
};

export default MediaPipeHandsPage;