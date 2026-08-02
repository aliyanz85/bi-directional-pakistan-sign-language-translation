import React, { useEffect, useRef, useState } from 'react';
import PSLInferenceEngine from '../inference/PSLInferenceEngine';
import HandTracker from '../inference/HandTracker';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { HAND_CONNECTIONS } from '@mediapipe/hands';
import './PSLRecognizer.css';

const PSLRecognizer = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  const [engineState, setEngineState] = useState({
    isInitialized: false,
    isLoading: true,
    error: null
  });
  
  const [predictionState, setPredictionState] = useState({
    currentPrediction: null,
    confidence: 0,
    alternatives: [],
    isStable: false,
    latency: 0
  });
  
  const [stats, setStats] = useState({
    fps: 0,
    avgLatency: 0,
    bufferProgress: 0
  });
  
  const inferenceEngineRef = useRef(null);
  const handTrackerRef = useRef(null);
  const fpsCounterRef = useRef({ frames: 0, lastTime: performance.now() });
  
  // Initialize on mount
  useEffect(() => {
    initializeSystem();
    
    return () => {
      cleanup();
    };
  }, []);
  
  const initializeSystem = async () => {
    try {
      setEngineState({ isInitialized: false, isLoading: true, error: null });
      
      // Initialize inference engine
      const engine = new PSLInferenceEngine();
      await engine.initialize();
      inferenceEngineRef.current = engine;
      
      // Initialize hand tracker
      const tracker = new HandTracker();
      await tracker.initialize(videoRef.current, handleTrackingResults);
      handTrackerRef.current = tracker;
      
      setEngineState({ isInitialized: true, isLoading: false, error: null });
      
    } catch (error) {
      console.error('Initialization error:', error);
      setEngineState({ 
        isInitialized: false, 
        isLoading: false, 
        error: error.message 
      });
    }
  };
  
  const handleTrackingResults = async (results) => {
    // Update FPS
    updateFPS();
    
    // Draw landmarks on canvas
    drawResults(results);
    
    // Process features for inference
    if (results.features && inferenceEngineRef.current) {
      const prediction = await inferenceEngineRef.current.processFrame(results.features);
      
      if (prediction) {
        if (prediction.status === 'success') {
          const topPred = prediction.predictions[0];
          const isStable = inferenceEngineRef.current.isPredictionStable();
          
          setPredictionState({
            currentPrediction: topPred.class,
            confidence: topPred.confidence,
            alternatives: prediction.predictions.slice(1, 3),
            isStable: isStable,
            latency: prediction.latency
          });
          
          // Update stats
          const engineStats = inferenceEngineRef.current.getStats();
          setStats({
            fps: fpsCounterRef.current.fps,
            avgLatency: engineStats.avgLatency,
            bufferProgress: 1.0
          });
          
        } else if (prediction.status === 'buffering') {
          setStats(prev => ({
            ...prev,
            bufferProgress: prediction.progress
          }));
        }
      }
    } else {
      // No hands detected
      setPredictionState(prev => ({
        ...prev,
        currentPrediction: null,
        confidence: 0
      }));
    }
  };
  
  const drawResults = (results) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!canvas || !ctx) return;
    
    // Set canvas size to match video
    if (results.image) {
      canvas.width = results.image.width;
      canvas.height = results.image.height;
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw landmarks
    if (results.landmarks) {
      for (const landmarks of results.landmarks) {
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
          color: '#00FF00',
          lineWidth: 2
        });
        drawLandmarks(ctx, landmarks, {
          color: '#FF0000',
          lineWidth: 1,
          radius: 3
        });
      }
    }
  };
  
  const updateFPS = () => {
    const now = performance.now();
    fpsCounterRef.current.frames++;
    
    const elapsed = now - fpsCounterRef.current.lastTime;
    if (elapsed >= 1000) {
      fpsCounterRef.current.fps = Math.round(
        (fpsCounterRef.current.frames * 1000) / elapsed
      );
      fpsCounterRef.current.frames = 0;
      fpsCounterRef.current.lastTime = now;
    }
  };
  
  const cleanup = () => {
    if (handTrackerRef.current) {
      handTrackerRef.current.stop();
    }
    if (inferenceEngineRef.current) {
      inferenceEngineRef.current.dispose();
    }
  };
  
  const resetPredictions = () => {
    if (inferenceEngineRef.current) {
      inferenceEngineRef.current.reset();
    }
    setPredictionState({
      currentPrediction: null,
      confidence: 0,
      alternatives: [],
      isStable: false,
      latency: 0
    });
  };
  
  return (
    <div className="psl-recognizer">
      <header className="header">
        <h1>VOICE PSL - Pakistan Sign Language Recognition</h1>
        <p className="subtitle">Real-time browser-based sign language recognition</p>
      </header>
      
      <div className="main-content">
        {/* Video and Canvas Container */}
        <div className="video-container">
          <video
            ref={videoRef}
            className="video-element"
            playsInline
          />
          <canvas
            ref={canvasRef}
            className="canvas-overlay"
          />
          
          {!engineState.isInitialized && engineState.isLoading && (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <p>Initializing AI models...</p>
            </div>
          )}
          
          {engineState.error && (
            <div className="error-overlay">
              <p>Error: {engineState.error}</p>
              <button onClick={initializeSystem}>Retry</button>
            </div>
          )}
          
          {stats.bufferProgress < 1.0 && stats.bufferProgress > 0 && (
            <div className="buffer-indicator">
              <div className="buffer-bar">
                <div 
                  className="buffer-progress" 
                  style={{ width: `${stats.bufferProgress * 100}%` }}
                />
              </div>
              <p>Collecting frames: {Math.round(stats.bufferProgress * 100)}%</p>
            </div>
          )}
        </div>
        
        {/* Predictions Panel */}
        <div className="predictions-panel">
          <h2>Recognition Results</h2>
          
          {predictionState.currentPrediction ? (
            <div className="prediction-result">
              <div className={`main-prediction ${predictionState.isStable ? 'stable' : 'unstable'}`}>
                <span className="word">{predictionState.currentPrediction}</span>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{ width: `${predictionState.confidence * 100}%` }}
                  />
                </div>
                <span className="confidence-value">
                  {(predictionState.confidence * 100).toFixed(1)}%
                </span>
                {predictionState.isStable && (
                  <span className="stable-badge">✓ Stable</span>
                )}
              </div>
              
              {predictionState.alternatives.length > 0 && (
                <div className="alternatives">
                  <h3>Alternative Predictions:</h3>
                  {predictionState.alternatives.map((alt, idx) => (
                    <div key={idx} className="alternative-item">
                      <span className="alt-word">{alt.class}</span>
                      <span className="alt-confidence">
                        {(alt.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="no-prediction">
              <p>👋 Show a sign to start recognition</p>
            </div>
          )}
          
          <button className="reset-button" onClick={resetPredictions}>
            Reset
          </button>
        </div>
      </div>
      
      {/* Stats Footer */}
      <footer className="stats-footer">
        <div className="stat-item">
          <span className="stat-label">FPS:</span>
          <span className="stat-value">{stats.fps}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Latency:</span>
          <span className="stat-value">{stats.avgLatency.toFixed(1)}ms</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Status:</span>
          <span className={`stat-value ${engineState.isInitialized ? 'ready' : 'not-ready'}`}>
            {engineState.isInitialized ? 'Ready' : 'Not Ready'}
          </span>
        </div>
      </footer>
    </div>
  );
};

export default PSLRecognizer;
