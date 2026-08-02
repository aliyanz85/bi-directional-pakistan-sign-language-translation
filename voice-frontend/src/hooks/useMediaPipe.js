/**
 * useMediaPipe Hook
 *
 * Integrates MediaPipe Hands for real-time hand landmark detection and feature extraction.
 * Extracts 188-dimensional feature vectors compatible with the PSL recognition model.
 */

import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * Extract 188-dimensional feature vector from MediaPipe hand landmarks
 *
 * Feature breakdown (per hand: 94 features × 2 hands = 188 total):
 * 1. Wrist-relative coordinates: 21 landmarks × 3D = 63 features
 * 2. Geometric features: 29 features (distances, angles, ratios)
 * 3. Hand label: 2 features (one-hot: [1,0] for left, [0,1] for right)
 *
 * @param {Object} results - MediaPipe Hands results
 * @returns {number[]} 188-dimensional feature vector
 */
const extractFeatures = (results) => {
  const features = [];

  // Process up to 2 hands (left and right)
  const hands = ['Left', 'Right'];

  for (const handLabel of hands) {
    let handFeatures = [];

    // Find this hand in the results
    const handIndex = results.multiHandedness?.findIndex(
      h => h.label === handLabel
    );

    if (handIndex >= 0 && results.multiHandLandmarks[handIndex]) {
      const landmarks = results.multiHandLandmarks[handIndex];

      // ===== 1. Wrist-Relative Coordinates (63 features) =====
      const wrist = landmarks[0]; // Wrist is landmark 0

      for (let i = 0; i < 21; i++) {
        const lm = landmarks[i];
        handFeatures.push(lm.x - wrist.x);
        handFeatures.push(lm.y - wrist.y);
        handFeatures.push(lm.z - wrist.z);
      }

      // ===== 2. Geometric Features (29 features) =====

      // Helper function to calculate Euclidean distance
      const distance = (lm1, lm2) => {
        const dx = lm1.x - lm2.x;
        const dy = lm1.y - lm2.y;
        const dz = lm1.z - lm2.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
      };

      // Palm width (distance from index base to pinky base)
      const palmWidth = distance(landmarks[5], landmarks[17]);

      // Fingertips indices
      const fingertips = [4, 8, 12, 16, 20]; // Thumb, Index, Middle, Ring, Pinky

      // a) Finger distances from wrist (5 features) - normalized by palm width
      for (const tipIdx of fingertips) {
        const dist = distance(wrist, landmarks[tipIdx]) / (palmWidth + 1e-6);
        handFeatures.push(dist);
      }

      // b) Inter-finger distances (10 features) - all pairs of fingertips
      for (let i = 0; i < fingertips.length; i++) {
        for (let j = i + 1; j < fingertips.length; j++) {
          const dist = distance(landmarks[fingertips[i]], landmarks[fingertips[j]]) / (palmWidth + 1e-6);
          handFeatures.push(dist);
        }
      }

      // c) Palm aspect ratio (1 feature)
      const palmHeight = distance(landmarks[0], landmarks[9]); // Wrist to middle finger base
      const aspectRatio = palmHeight / (palmWidth + 1e-6);
      handFeatures.push(aspectRatio);

      // d) Finger extension ratios (5 features)
      const fingerBases = [1, 5, 9, 13, 17]; // Bases of each finger
      for (let i = 0; i < 5; i++) {
        const tipDist = distance(wrist, landmarks[fingertips[i]]);
        const baseDist = distance(wrist, landmarks[fingerBases[i]]);
        const ratio = tipDist / (baseDist + 1e-6);
        handFeatures.push(ratio);
      }

      // e) Finger angles (5 features) - simplified as normalized dot products
      for (let i = 0; i < 5; i++) {
        const base = landmarks[fingerBases[i]];
        const tip = landmarks[fingertips[i]];

        // Vector from base to tip
        const vx = tip.x - base.x;
        const vy = tip.y - base.y;
        const vz = tip.z - base.z;
        const mag = Math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-6;

        // Use the y-component as a simple angle proxy (upward direction)
        handFeatures.push(vy / mag);
      }

      // f) Hand orientation (3 features) - direction from wrist to middle finger base
      const middleBase = landmarks[9];
      const orientX = (middleBase.x - wrist.x);
      const orientY = (middleBase.y - wrist.y);
      const orientZ = (middleBase.z - wrist.z);
      const orientMag = Math.sqrt(orientX * orientX + orientY * orientY + orientZ * orientZ) + 1e-6;

      handFeatures.push(orientX / orientMag);
      handFeatures.push(orientY / orientMag);
      handFeatures.push(orientZ / orientMag);

      // Total geometric features: 5 + 10 + 1 + 5 + 5 + 3 = 29 ✓

      // ===== 3. Hand Label (2 features) =====
      if (handLabel === 'Left') {
        handFeatures.push(1.0); // Left hand
        handFeatures.push(0.0);
      } else {
        handFeatures.push(0.0);
        handFeatures.push(1.0); // Right hand
      }

      // Total per hand: 63 + 29 + 2 = 94 features ✓
    } else {
      // Hand not detected - fill with zeros
      handFeatures = new Array(94).fill(0.0);
    }

    features.push(...handFeatures);
  }

  // Total features: 94 × 2 = 188 ✓
  return features;
};

/**
 * useMediaPipe Hook
 *
 * @param {React.RefObject} videoRef - Reference to the video element
 * @param {Object} options - Configuration options
 * @param {Function} options.onFeatures - Callback when features are extracted
 * @param {Function} options.onResults - Callback when MediaPipe results are available
 * @param {boolean} options.drawLandmarks - Whether to draw landmarks on canvas
 * @returns {Object} MediaPipe state and control functions
 */
export const useMediaPipe = (videoRef, options = {}) => {
  const {
    onFeatures,
    onResults,
    drawLandmarks = true
  } = options;

  const [isInitialized, setIsInitialized] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState(null);
  const [handsDetected, setHandsDetected] = useState(0);
  const [fps, setFps] = useState(0);

  const handsRef = useRef(null);
  const cameraRef = useRef(null);
  const canvasRef = useRef(null);
  const fpsCounterRef = useRef({ frames: 0, lastTime: Date.now() });

  /**
   * Initialize MediaPipe Hands
   */
  const initializeMediaPipe = useCallback(async () => {
    try {
      // Dynamically import MediaPipe modules
      const { Hands } = await import('@mediapipe/hands');
      const { Camera } = await import('@mediapipe/camera_utils');

      console.log('Initializing MediaPipe Hands...');

      // Create Hands instance
      const hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
      });

      // Configure Hands
      hands.setOptions({
        maxNumHands: 2, // Detect up to 2 hands
        modelComplexity: 1, // 0=lite, 1=full (better accuracy)
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.6
      });

      // Set up results handler
      hands.onResults((results) => {
        // Update hands detected count
        setHandsDetected(results.multiHandLandmarks?.length || 0);

        // Extract 188-dim features
        const features = extractFeatures(results);

        // Call feature callback
        if (onFeatures) {
          onFeatures(features);
        }

        // Call results callback
        if (onResults) {
          onResults(results);
        }

        // Draw landmarks if enabled
        if (drawLandmarks && canvasRef.current && videoRef.current) {
          drawHandLandmarks(results);
        }

        // Update FPS
        updateFPS();
      });

      handsRef.current = hands;

      // Initialize camera
      if (videoRef.current) {
        const camera = new Camera(videoRef.current, {
          onFrame: async () => {
            if (handsRef.current && videoRef.current) {
              await handsRef.current.send({ image: videoRef.current });
            }
          },
          width: 640,
          height: 480
        });

        cameraRef.current = camera;
      }

      setIsInitialized(true);
      console.log('MediaPipe Hands initialized successfully');

    } catch (err) {
      console.error('Failed to initialize MediaPipe:', err);
      setError(err.message);
    }
  }, [videoRef, onFeatures, onResults, drawLandmarks]);

  /**
   * Start hand detection
   */
  const startDetection = useCallback(async () => {
    if (!isInitialized) {
      await initializeMediaPipe();
    }

    if (cameraRef.current) {
      try {
        await cameraRef.current.start();
        setIsDetecting(true);
        console.log('Hand detection started');
      } catch (err) {
        console.error('Failed to start camera:', err);
        setError(err.message);
      }
    }
  }, [isInitialized, initializeMediaPipe]);

  /**
   * Stop hand detection
   */
  const stopDetection = useCallback(() => {
    if (cameraRef.current) {
      cameraRef.current.stop();
      setIsDetecting(false);
      console.log('Hand detection stopped');
    }
  }, []);

  /**
   * Draw hand landmarks on canvas
   */
  const drawHandLandmarks = (results) => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Match canvas size to video
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw landmarks
    if (results.multiHandLandmarks) {
      for (let i = 0; i < results.multiHandLandmarks.length; i++) {
        const landmarks = results.multiHandLandmarks[i];
        const handedness = results.multiHandedness[i].label;

        // Draw connections
        ctx.strokeStyle = handedness === 'Left' ? '#00FF00' : '#FF0000';
        ctx.lineWidth = 2;

        // MediaPipe hand connections
        const connections = [
          [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
          [0, 5], [5, 6], [6, 7], [7, 8], // Index
          [0, 9], [9, 10], [10, 11], [11, 12], // Middle
          [0, 13], [13, 14], [14, 15], [15, 16], // Ring
          [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
          [5, 9], [9, 13], [13, 17] // Palm
        ];

        for (const [start, end] of connections) {
          const startLm = landmarks[start];
          const endLm = landmarks[end];

          ctx.beginPath();
          ctx.moveTo(startLm.x * canvas.width, startLm.y * canvas.height);
          ctx.lineTo(endLm.x * canvas.width, endLm.y * canvas.height);
          ctx.stroke();
        }

        // Draw landmarks
        ctx.fillStyle = handedness === 'Left' ? '#00FF00' : '#FF0000';
        for (const lm of landmarks) {
          ctx.beginPath();
          ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }
  };

  /**
   * Update FPS counter
   */
  const updateFPS = () => {
    fpsCounterRef.current.frames++;
    const now = Date.now();
    const elapsed = now - fpsCounterRef.current.lastTime;

    if (elapsed >= 1000) {
      const currentFPS = Math.round((fpsCounterRef.current.frames * 1000) / elapsed);
      setFps(currentFPS);
      fpsCounterRef.current.frames = 0;
      fpsCounterRef.current.lastTime = now;
    }
  };

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      stopDetection();
      if (handsRef.current) {
        handsRef.current.close();
      }
    };
  }, [stopDetection]);

  return {
    isInitialized,
    isDetecting,
    error,
    handsDetected,
    fps,
    startDetection,
    stopDetection,
    canvasRef,
    initializeMediaPipe
  };
};

export default useMediaPipe;
