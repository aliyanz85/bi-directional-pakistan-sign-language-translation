/**
 * MediaPipe Hand Tracking Integration
 * Extracts hand landmarks and processes them for inference
 */

import { Hands } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';

class HandTracker {
  constructor() {
    this.hands = null;
    this.camera = null;
    this.isInitialized = false;
    this.onResults = null;
    
    // Feature extraction settings
    this.numLandmarks = 21;
    this.featureDim = 188; // 2 hands × 94 features per hand
  }
  
  /**
   * Initialize MediaPipe Hands
   */
  async initialize(videoElement, onResults) {
    console.log('Initializing hand tracker...');
    
    this.onResults = onResults;
    
    // Initialize MediaPipe Hands
    this.hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });
    
    // Configure hands
    this.hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.6
    });
    
    // Set results callback
    this.hands.onResults((results) => this.handleResults(results));
    
    // Initialize camera
    if (videoElement) {
      this.camera = new Camera(videoElement, {
        onFrame: async () => {
          if (this.hands) {
            await this.hands.send({ image: videoElement });
          }
        },
        width: 640,
        height: 480,
        facingMode: 'user'
      });
      
      await this.camera.start();
    }
    
    this.isInitialized = true;
    console.log('Hand tracker initialized');
    
    return true;
  }
  
  /**
   * Handle MediaPipe results
   */
  handleResults(results) {
    if (!this.onResults) return;
    
    // Extract features from detected hands
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const features = this.extractFeatures(
        results.multiHandLandmarks,
        results.multiHandedness,
        results.image
      );
      
      this.onResults({
        landmarks: results.multiHandLandmarks,
        handedness: results.multiHandedness,
        features: features,
        image: results.image
      });
    } else {
      // No hands detected
      this.onResults({
        landmarks: null,
        handedness: null,
        features: null,
        image: results.image
      });
    }
  }
  
  /**
   * Extract feature vector from hand landmarks
   */
  extractFeatures(handLandmarksList, handednessList, image) {
    const features = [];
    
    // Sort hands by handedness (left first, then right)
    const handsData = handLandmarksList.map((landmarks, idx) => ({
      landmarks: landmarks,
      handedness: handednessList[idx].label
    })).sort((a, b) => a.handedness.localeCompare(b.handedness));
    
    // Process each hand (max 2)
    for (let i = 0; i < Math.min(2, handsData.length); i++) {
      const handFeatures = this.processHandLandmarks(handsData[i].landmarks);
      features.push(...handFeatures);
    }
    
    // Pad with zeros if only one hand detected
    if (handsData.length === 1) {
      const paddingSize = this.featureDim / 2;
      features.push(...new Array(paddingSize).fill(0));
    }
    
    // If no hands, return zeros
    if (handsData.length === 0) {
      return new Array(this.featureDim).fill(0);
    }
    
    return features;
  }
  
  /**
   * Process single hand landmarks into feature vector
   */
  processHandLandmarks(landmarks) {
    const features = [];
    
    // Extract raw coordinates (21 landmarks × 3 coordinates = 63 features)
    const rawCoords = [];
    for (const lm of landmarks) {
      rawCoords.push(lm.x, lm.y, lm.z);
    }
    
    // Normalize relative to wrist (translation invariance)
    const wrist = { x: landmarks[0].x, y: landmarks[0].y, z: landmarks[0].z };
    const relativeCoords = [];
    
    for (let i = 0; i < landmarks.length; i++) {
      relativeCoords.push(
        landmarks[i].x - wrist.x,
        landmarks[i].y - wrist.y,
        landmarks[i].z - wrist.z
      );
    }
    
    // Compute hand geometry features
    const geometry = this.computeHandGeometry(landmarks);
    
    // Hand label (assuming left=1,0 / right=0,1)
    // Note: This should match the training data encoding
    const handLabel = [1, 0]; // Placeholder - should be determined from handedness
    
    // Combine all features
    features.push(...relativeCoords);  // 63 features
    features.push(...geometry);        // ~29 features
    features.push(...handLabel);       // 2 features
    
    return features;
  }
  
  /**
   * Compute geometric features from landmarks
   */
  computeHandGeometry(landmarks) {
    const geometry = [];
    
    // Landmark indices
    const WRIST = 0;
    const THUMB_TIP = 4;
    const INDEX_TIP = 8;
    const MIDDLE_TIP = 12;
    const RING_TIP = 16;
    const PINKY_TIP = 20;
    
    const fingertips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP];
    
    // 1. Finger tip to wrist distances (5 features)
    const wrist = landmarks[WRIST];
    for (const tipIdx of fingertips) {
      const dist = this.distance3D(landmarks[tipIdx], wrist);
      geometry.push(dist);
    }
    
    // 2. Inter-finger tip distances (10 features)
    for (let i = 0; i < fingertips.length; i++) {
      for (let j = i + 1; j < fingertips.length; j++) {
        const dist = this.distance3D(landmarks[fingertips[i]], landmarks[fingertips[j]]);
        geometry.push(dist);
      }
    }
    
    // 3. Palm dimensions (2 features)
    const palmWidth = this.distance3D(landmarks[2], landmarks[17]);
    const palmHeight = this.distance3D(landmarks[0], landmarks[9]);
    geometry.push(palmWidth, palmHeight);
    
    // 4. Finger extension ratios (5 features)
    const fingerBases = [2, 5, 9, 13, 17];
    for (let i = 0; i < fingertips.length; i++) {
      const tipDist = this.distance3D(landmarks[fingertips[i]], wrist);
      const baseDist = this.distance3D(landmarks[fingerBases[i]], wrist);
      const ratio = tipDist / (baseDist + 1e-6);
      geometry.push(ratio);
    }
    
    // 5. Finger angles (5 features)
    const fingerStarts = [1, 5, 9, 13, 17];
    for (const start of fingerStarts) {
      const angle = this.calculateAngle(
        landmarks[start],
        landmarks[start + 1],
        landmarks[start + 2]
      );
      geometry.push(angle);
    }
    
    // 6. Hand orientation (2 features)
    const orientation_x = landmarks[9].x - landmarks[0].x;
    const orientation_y = landmarks[9].y - landmarks[0].y;
    geometry.push(orientation_x, orientation_y);
    
    return geometry;
  }
  
  /**
   * Calculate 3D distance between two points
   */
  distance3D(p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const dz = p2.z - p1.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  
  /**
   * Calculate angle between three points
   */
  calculateAngle(p1, p2, p3) {
    // Vectors
    const v1 = {
      x: p2.x - p1.x,
      y: p2.y - p1.y,
      z: p2.z - p1.z
    };
    
    const v2 = {
      x: p3.x - p2.x,
      y: p3.y - p2.y,
      z: p3.z - p2.z
    };
    
    // Dot product
    const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    
    // Magnitudes
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
    
    // Angle
    const cosAngle = dot / (mag1 * mag2 + 1e-6);
    return Math.acos(Math.max(-1, Math.min(1, cosAngle)));
  }
  
  /**
   * Stop tracking
   */
  stop() {
    if (this.camera) {
      this.camera.stop();
    }
    
    if (this.hands) {
      this.hands.close();
    }
    
    this.isInitialized = false;
    console.log('Hand tracker stopped');
  }
}

export default HandTracker;
