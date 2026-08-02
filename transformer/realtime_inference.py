"""
Real-time PSL Recognition using Webcam
Captures hand landmarks and predicts PSL words in real-time
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import json
from pathlib import Path
import time

# Import custom layers
import sys
sys.path.append('models/training')
from model_architecture import TCNBlock, TransformerBlock, PositionalEncoding


class RealtimePSLRecognizer:
    """Real-time PSL word recognition from webcam"""
    
    def __init__(self, model_path, metadata_path, normalization_path, sequence_length=60, confidence_threshold=0.6):
        """
        Initialize the real-time recognizer
        
        Args:
            model_path: Path to trained model
            metadata_path: Path to metadata JSON
            normalization_path: Path to normalization parameters JSON
            sequence_length: Number of frames for prediction
            confidence_threshold: Minimum confidence for prediction
        """
        # Load model
        print("Loading model...")
        custom_objects = {
            'TCNBlock': TCNBlock,
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        }
        self.model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully!")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.class_names = metadata['class_names']
        self.num_classes = len(self.class_names)
        
        # Load normalization parameters
        with open(normalization_path, 'r') as f:
            norm_params = json.load(f)
        self.norm_mean = np.array(norm_params['mean'], dtype=np.float32)
        self.norm_std = np.array(norm_params['std'], dtype=np.float32)
        print("Normalization parameters loaded!")
        
        # Settings
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # MediaPipe setup with FAST settings
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # FAST: 0=lite, 1=full (default)
        )
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        self.current_prediction = None
        self.current_confidence = 0.0
        
        # Performance metrics
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        
        # Detailed logging for debugging (ENABLED for debugging)
        self.log_file = Path("realtime_debug_log.txt")
        self.csv_file = Path("realtime_coordinates.csv")
        self.feature_log = Path("feature_analysis.txt")
        self.frame_count = 0
        self.log_coordinates = True  # Set to True to enable detailed logging
        self.debug_mode = True  # Comprehensive debugging
        
        # Initialize log files
        if self.log_coordinates:
            with open(self.csv_file, 'w') as f:
                f.write("frame,timestamp,hand_idx,landmark_idx,x,y,z,x_rel,y_rel,z_rel\n")
        
        # Initialize feature analysis log
        with open(self.feature_log, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("FEATURE EXTRACTION ANALYSIS LOG\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model expects: {self.sequence_length} frames × 188 features\n")
            f.write(f"Normalization mean shape: {self.norm_mean.shape}\n")
            f.write(f"Normalization std shape: {self.norm_std.shape}\n")
            f.write(f"Norm mean range: [{self.norm_mean.min():.6f}, {self.norm_mean.max():.6f}]\n")
            f.write(f"Norm std range: [{self.norm_std.min():.6f}, {self.norm_std.max():.6f}]\n")
            f.write(f"Classes: {self.class_names}\n")
            f.write("="*80 + "\n\n")
        
    def debug_features(self, features, stage="", frame_num=0):
        """Comprehensive feature debugging"""
        if not self.debug_mode:
            return
            
        with open(self.feature_log, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"DEBUG: {stage} (Frame {frame_num})\n")
            f.write(f"{'='*80}\n")
            
            if features is None:
                f.write("❌ Features are None!\n")
                return
            
            f.write(f"✓ Shape: {features.shape}\n")
            f.write(f"✓ Dtype: {features.dtype}\n")
            f.write(f"✓ Range: [{features.min():.6f}, {features.max():.6f}]\n")
            f.write(f"✓ Mean: {features.mean():.6f}\n")
            f.write(f"✓ Std: {features.std():.6f}\n")
            f.write(f"✓ NaN count: {np.isnan(features).sum()}\n")
            f.write(f"✓ Inf count: {np.isinf(features).sum()}\n")
            
            # Check wrist coords (should be 0,0,0 for wrist-relative)
            if len(features.shape) == 1 and features.shape[0] >= 3:
                f.write(f"✓ Wrist coords (should be 0,0,0): [{features[0]:.6f}, {features[1]:.6f}, {features[2]:.6f}]\n")
                if abs(features[0]) > 0.001 or abs(features[1]) > 0.001 or abs(features[2]) > 0.001:
                    f.write("⚠️  WARNING: Wrist not at origin! Feature extraction may be wrong!\n")
            
            # Sample values
            f.write(f"✓ First 10 features: {features.flatten()[:10]}\n")
            f.write(f"✓ Last 10 features: {features.flatten()[-10:]}\n")
            
            # Feature breakdown (assuming 188 features)
            if len(features.shape) == 1 and features.shape[0] == 188:
                f.write(f"\nFeature breakdown:\n")
                f.write(f"  Hand 1 coords [0:63]: range [{features[0:63].min():.4f}, {features[0:63].max():.4f}]\n")
                f.write(f"  Hand 1 geometry [63:92]: range [{features[63:92].min():.4f}, {features[63:92].max():.4f}]\n")
                f.write(f"  Hand 1 label [92:94]: {features[92:94]}\n")
                f.write(f"  Hand 2 coords [94:157]: range [{features[94:157].min():.4f}, {features[94:157].max():.4f}]\n")
                f.write(f"  Hand 2 geometry [157:186]: range [{features[157:186].min():.4f}, {features[157:186].max():.4f}]\n")
                f.write(f"  Hand 2 label [186:188]: {features[186:188]}\n")
    
    def extract_landmarks(self, image):
        """
        Extract hand landmarks from image (COMPLETE: wrist-relative + geometry features)
        
        Args:
            image: BGR image from webcam
            
        Returns:
            Landmark features (188-dim) or None if no hands detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract landmarks for up to 2 hands
        all_features = []
        timestamp = time.time()
        
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            # Extract raw x, y, z for each landmark (21 landmarks × 3 = 63)
            raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # CRITICAL: Normalize relative to wrist (same as training!)
            wrist = raw_coords[0]
            relative_coords = raw_coords - wrist
            
            # Log coordinates to CSV for debugging
            if self.log_coordinates and self.frame_count % 5 == 0:  # Log every 5th frame
                with open(self.csv_file, 'a') as f:
                    for lm_idx in range(21):
                        f.write(f"{self.frame_count},{timestamp:.4f},{hand_idx},{lm_idx},"
                               f"{raw_coords[lm_idx][0]:.6f},{raw_coords[lm_idx][1]:.6f},{raw_coords[lm_idx][2]:.6f},"
                               f"{relative_coords[lm_idx][0]:.6f},{relative_coords[lm_idx][1]:.6f},{relative_coords[lm_idx][2]:.6f}\n")
            
            # Compute geometric features (29 features - MUST match training!)
            geometry = self._compute_hand_geometry(raw_coords)
            
            # Hand label (left=1,0 / right=0,1) - 2 features
            if results.multi_handedness:
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                hand_onehot = np.array([1, 0] if hand_label == 'Left' else [0, 1])
            else:
                hand_onehot = np.array([0, 0])
            
            # Combine: 63 (relative coords) + 29 (geometry) + 2 (handedness) = 94 per hand
            hand_features = np.concatenate([
                relative_coords.flatten(),  # 63 features
                geometry,                    # 29 features
                hand_onehot                  # 2 features
            ])
            
            all_features.append(hand_features)
        
        # Handle single hand case (pad with zeros for second hand)
        if len(all_features) == 1:
            all_features.append(np.zeros(94, dtype=np.float32))
        elif len(all_features) == 0:
            # No hands detected (shouldn't reach here, but safety)
            all_features = [np.zeros(94, dtype=np.float32), np.zeros(94, dtype=np.float32)]
        
        # Concatenate both hands: 94 + 94 = 188 features
        final_features = np.concatenate(all_features).astype(np.float32)
        
        # Debug every 30th frame
        if self.debug_mode and self.frame_count % 30 == 0:
            self.debug_features(final_features, "Raw Features Extracted", self.frame_count)
        
        return final_features
    
    def _compute_hand_geometry(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute SCALE-INVARIANT geometric features (EXACT SAME as training!)
        
        Args:
            landmarks: Array of shape (21, 3) with landmark coordinates
        
        Returns:
            Array of 29 geometric features
        """
        geometry = []
        
        # Landmark indices
        WRIST = 0
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        
        fingertips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        
        # CRITICAL: Compute palm width as reference scale for normalization
        # This makes features scale-invariant (works for big/small hands, near/far camera)
        palm_width = np.linalg.norm(landmarks[2] - landmarks[17])
        scale = palm_width + 1e-6  # Avoid division by zero
        
        # Debug: Check scale value
        if self.debug_mode and self.frame_count % 30 == 0:
            with open(self.feature_log, 'a', encoding='utf-8') as f:
                f.write(f"\n>>> Geometry computation (Frame {self.frame_count}):\n")
                f.write(f"    Palm width (scale): {palm_width:.6f}\n")
        
        # 1. Finger tip to wrist distances NORMALIZED by palm width (5 features)
        wrist_pos = landmarks[WRIST]
        for tip_idx in fingertips:
            dist = np.linalg.norm(landmarks[tip_idx] - wrist_pos)
            geometry.append(dist / scale)  # Scale-invariant!
        
        # 2. Inter-finger tip distances NORMALIZED by palm width (10 features)
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[j]])
                geometry.append(dist / scale)  # Scale-invariant!
        
        # 3. Palm aspect ratio (height/width) - already scale-invariant (1 feature)
        palm_height = np.linalg.norm(landmarks[0] - landmarks[9])
        geometry.append(palm_height / scale)  # Ratio is scale-invariant
        
        # 4. Finger extension ratios (5 features)
        finger_bases = [2, 5, 9, 13, 17]
        for tip_idx, base_idx in zip(fingertips, finger_bases):
            tip_dist = np.linalg.norm(landmarks[tip_idx] - wrist_pos)
            base_dist = np.linalg.norm(landmarks[base_idx] - wrist_pos)
            ratio = tip_dist / (base_dist + 1e-6)
            geometry.append(ratio)
        
        # 5. Finger angles (5 features)
        for finger_start in [1, 5, 9, 13, 17]:
            p1 = landmarks[finger_start]
            p2 = landmarks[finger_start + 1]
            p3 = landmarks[finger_start + 2]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            geometry.append(angle)
        
        # 6. Hand orientation NORMALIZED (3 features - x, y, z)
        # Direction vector from wrist to middle finger base, normalized by palm width
        orientation = landmarks[9] - landmarks[0]
        geometry.extend([orientation[0] / scale, orientation[1] / scale, orientation[2] / scale])  # Scale-invariant!
        
        result = np.array(geometry)
        
        # Debug geometry features
        if self.debug_mode and self.frame_count % 30 == 0:
            with open(self.feature_log, 'a', encoding='utf-8') as f:
                f.write(f"    Geometry features count: {len(result)}\n")
                f.write(f"    Geometry range: [{result.min():.6f}, {result.max():.6f}]\n")
                f.write(f"    Expected: 30 features\n")
                if len(result) != 30:
                    f.write(f"    ⚠️  ERROR: Expected 30 geometry features, got {len(result)}!\n")
        
        return result
    
    def predict(self, sequence):
        """
        Predict PSL word from sequence
        
        Args:
            sequence: Array of shape (sequence_length, 188)
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        # Debug sequence before normalization
        if self.debug_mode:
            with open(self.feature_log, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"PREDICTION PIPELINE (Frame {self.frame_count})\n")
                f.write(f"{'='*80}\n")
                f.write(f"Sequence shape: {sequence.shape}\n")
                f.write(f"Sequence range BEFORE norm: [{sequence.min():.6f}, {sequence.max():.6f}]\n")
                f.write(f"Sequence mean BEFORE norm: {sequence.mean():.6f}\n")
                f.write(f"Sequence std BEFORE norm: {sequence.std():.6f}\n")
        
        # Apply normalization (CRITICAL: same as training)
        normalized_sequence = (sequence - self.norm_mean) / self.norm_std
        
        # Debug after normalization
        if self.debug_mode:
            with open(self.feature_log, 'a', encoding='utf-8') as f:
                f.write(f"\nAfter normalization:\n")
                f.write(f"  Range: [{normalized_sequence.min():.6f}, {normalized_sequence.max():.6f}]\n")
                f.write(f"  Mean: {normalized_sequence.mean():.6f} (should be ~0)\n")
                f.write(f"  Std: {normalized_sequence.std():.6f} (should be ~1)\n")
                
                # Check for anomalies
                if abs(normalized_sequence.mean()) > 2.0:
                    f.write(f"  ⚠️  WARNING: Mean too far from 0! Normalization may be wrong!\n")
                if normalized_sequence.std() < 0.1 or normalized_sequence.std() > 10:
                    f.write(f"  ⚠️  WARNING: Std unusual! Check normalization params!\n")
                if np.isnan(normalized_sequence).any():
                    f.write(f"  ❌ ERROR: NaN values after normalization!\n")
                if np.isinf(normalized_sequence).any():
                    f.write(f"  ❌ ERROR: Inf values after normalization!\n")
        
        # Prepare input
        input_data = np.expand_dims(normalized_sequence, axis=0)  # (1, 60, 188)
        
        # Get predictions
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_class = self.class_names[predicted_idx]
        
        # Debug predictions
        if self.debug_mode:
            with open(self.feature_log, 'a', encoding='utf-8') as f:
                f.write(f"\nModel output:\n")
                f.write(f"  Probabilities shape: {predictions.shape}\n")
                f.write(f"  Probabilities sum: {predictions.sum():.6f} (should be ~1.0)\n")
                f.write(f"  Max probability: {predictions.max():.6f}\n")
                f.write(f"  Predicted class: {predicted_class} (index {predicted_idx})\n")
                f.write(f"  Confidence: {confidence:.6f}\n")
                
                # Top 5 predictions
                top5_indices = np.argsort(predictions)[-5:][::-1]
                f.write(f"\n  Top 5 predictions:\n")
                for i, idx in enumerate(top5_indices, 1):
                    f.write(f"    {i}. {self.class_names[idx]}: {predictions[idx]:.4f}\n")
                
                # Check for issues
                if predictions.max() < 0.1:
                    f.write(f"  ⚠️  WARNING: All confidences very low! Model may not be trained properly!\n")
                if predictions.max() > 0.99:
                    f.write(f"  ℹ️  Very high confidence - could be overfitting or perfect match\n")
        
        return predicted_class, confidence, predictions
    
    def smooth_prediction(self, predicted_class, confidence):
        """
        Smooth predictions over time to reduce jitter
        
        Args:
            predicted_class: Current prediction
            confidence: Current confidence
            
        Returns:
            Tuple of (smoothed_class, smoothed_confidence)
        """
        # Add to history
        self.prediction_history.append((predicted_class, confidence))
        
        # Get most common prediction from recent history
        if len(self.prediction_history) >= 3:
            recent_predictions = [p[0] for p in self.prediction_history if p[1] > self.confidence_threshold]
            if recent_predictions:
                # Most common prediction
                from collections import Counter
                most_common = Counter(recent_predictions).most_common(1)[0][0]
                avg_confidence = np.mean([p[1] for p in self.prediction_history if p[0] == most_common])
                return most_common, avg_confidence
        
        return predicted_class, confidence
    
    def draw_landmarks(self, image, results):
        """Draw hand landmarks on image"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
    
    def draw_ui(self, image, predicted_class, confidence, top_predictions, fps):
        """
        Draw UI elements on image
        
        Args:
            image: Image to draw on
            predicted_class: Predicted word
            confidence: Confidence score
            top_predictions: List of (class, probability) tuples
            fps: Current FPS
        """
        height, width = image.shape[:2]
        
        # Draw semi-transparent overlay for text background
        overlay = image.copy()
        
        # Top bar - Prediction
        if confidence > self.confidence_threshold:
            color = (0, 255, 0)  # Green
            text = f"Sign: {predicted_class.upper()}"
        else:
            color = (0, 165, 255)  # Orange
            text = "Detecting..."
        
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, color, 3, cv2.LINE_AA)
        
        # Confidence bar
        if confidence > self.confidence_threshold:
            bar_width = int(300 * confidence)
            cv2.rectangle(image, (20, 60), (20 + bar_width, 70), color, -1)
            cv2.rectangle(image, (20, 60), (320, 70), (255, 255, 255), 2)
            cv2.putText(image, f"{confidence*100:.1f}%", (330, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Top 5 predictions sidebar
        sidebar_x = width - 300
        cv2.rectangle(overlay, (sidebar_x, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        cv2.putText(image, "Top Predictions:", (sidebar_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, (cls, prob) in enumerate(top_predictions[:5]):
            y = 70 + i * 40
            # Class name
            cv2.putText(image, f"{i+1}. {cls}", (sidebar_x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Probability bar
            bar_w = int(250 * prob)
            bar_color = (0, 255, 0) if i == 0 else (100, 100, 255)
            cv2.rectangle(image, (sidebar_x + 10, y + 5), 
                         (sidebar_x + 10 + bar_w, y + 15), bar_color, -1)
            # Percentage
            cv2.putText(image, f"{prob*100:.1f}%", (sidebar_x + 10, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS and buffer status
        buffer_status = f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}"
        cv2.putText(image, f"FPS: {fps:.1f}", (20, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, buffer_status, (20, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(image, "Press 'Q' or 'P' to quit | 'R' to reset", 
                   (width // 2 - 250, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Run real-time recognition"""
        # Open webcam with FAST settings
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Target 30 FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer latency
        
        print("\n" + "="*60)
        print("REAL-TIME PSL RECOGNITION")
        print("="*60)
        print(f"Loaded {self.num_classes} classes")
        print(f"Sequence length: {self.sequence_length} frames")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("\nControls:")
        print("  Q/P - Quit")
        print("  R - Reset buffer")
        print("="*60 + "\n")
        
        predicted_class = ""
        confidence = 0.0
        top_predictions = []
        
        while cap.isOpened():
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            self.frame_count += 1
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks = self.extract_landmarks(frame)
            
            # Process with MediaPipe for visualization
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            # Draw landmarks
            self.draw_landmarks(frame, results)
            
            # Add to buffer if landmarks detected
            if landmarks is not None:
                self.frame_buffer.append(landmarks)
                # Log buffer addition
                if self.debug_mode and self.frame_count % 30 == 0:  # Log every 30 frames
                    with open(self.feature_log, 'a', encoding='utf-8') as f:
                        f.write(f"\n>>> Frame {self.frame_count}: Landmarks detected and added\n")
                        f.write(f"    Buffer size: {len(self.frame_buffer)}/{self.sequence_length}\n")
                        f.write(f"    Landmarks shape: {landmarks.shape}\n")
                        if landmarks.shape[0] != 188:
                            f.write(f"    ❌ ERROR: Expected 188 features, got {landmarks.shape[0]}!\n")
            else:
                # Add zero frame if no hands detected
                zero_frame = np.zeros(188, dtype=np.float32)
                self.frame_buffer.append(zero_frame)
                if self.debug_mode and self.frame_count % 30 == 0:
                    with open(self.feature_log, 'a', encoding='utf-8') as f:
                        f.write(f"\n>>> Frame {self.frame_count}: No hands detected, added zeros\n")
                        f.write(f"    Buffer size: {len(self.frame_buffer)}/{self.sequence_length}\n")
            
            # Predict when buffer is full
            if len(self.frame_buffer) == self.sequence_length:
                # Debug: check shapes
                buffer_list = list(self.frame_buffer)
                for i, buf_frame in enumerate(buffer_list):
                    if not isinstance(buf_frame, np.ndarray):
                        print(f"Buffer frame {i} is not ndarray: {type(buf_frame)}")
                    elif buf_frame.shape != (188,):
                        print(f"Buffer frame {i} has wrong shape: {buf_frame.shape}")
                
                try:
                    sequence = np.stack(buffer_list)  # Use stack instead of array
                except Exception as e:
                    print(f"Error stacking: {e}")
                    print(f"Buffer shapes: {[f.shape if isinstance(f, np.ndarray) else type(f) for f in buffer_list]}")
                    continue
                    
                pred_class, conf, all_probs = self.predict(sequence)
                
                # Smooth prediction
                predicted_class, confidence = self.smooth_prediction(pred_class, conf)
                
                # Get top 5 predictions
                top_indices = np.argsort(all_probs)[-5:][::-1]
                top_predictions = [(self.class_names[i], all_probs[i]) for i in top_indices]
                
                # Log sequence statistics
                if self.log_coordinates:
                    with open(self.log_file, 'a') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Frame {self.frame_count} - PREDICTION MADE\n")
                        f.write(f"Sequence shape: {sequence.shape}\n")
                        f.write(f"Sequence mean: {sequence.mean():.4f}, std: {sequence.std():.4f}\n")
                        f.write(f"Sequence min: {sequence.min():.4f}, max: {sequence.max():.4f}\n")
                        f.write(f"First 10 features of frame 0: {sequence[0][:10]}\n")
                        f.write(f"First 10 features of frame 30: {sequence[30][:10]}\n")
                        f.write(f"First 10 features of frame 59: {sequence[59][:10]}\n")
                        f.write(f"\nPredicted: {pred_class} (conf: {conf:.4f})\n")
                        f.write(f"Top 5: {[(c, f'{p:.4f}') for c, p in top_predictions]}\n")
                        f.write(f"{'='*80}\n")
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            self.fps = 1.0 / np.mean(self.frame_times) if len(self.frame_times) > 0 else 0
            
            # Draw UI
            if frame is None:
                print("Warning: frame is None!")
                continue
            if len(frame.shape) != 3:
                print(f"Warning: frame has unexpected shape: {frame.shape}")
                continue
                
            self.draw_ui(frame, predicted_class, confidence, top_predictions, self.fps)
            
            # Add debug info overlay
            debug_y = 100
            if landmarks is not None:
                cv2.putText(frame, f"Wrist-rel coords: {landmarks[0]:.3f}, {landmarks[1]:.3f}, {landmarks[2]:.3f}", 
                           (20, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"Feature range: [{landmarks.min():.3f}, {landmarks.max():.3f}]", 
                           (20, debug_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Display
            cv2.imshow('PSL Recognition - Real-time', frame)
            
            # Handle key presses (1ms wait for max FPS)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == ord('p') or key == ord('P'):
                print("\nQuitting...")
                break
            elif key == ord('r') or key == ord('R'):
                print("Resetting buffer...")
                self.frame_buffer.clear()
                self.prediction_history.clear()
                predicted_class = ""
                confidence = 0.0
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Final log summary
        if self.log_coordinates:
            with open(self.log_file, 'a') as f:
                f.write(f"\n\n{'='*80}\n")
                f.write(f"SESSION SUMMARY\n")
                f.write(f"Total frames processed: {self.frame_count}\n")
                f.write(f"Average FPS: {self.fps:.2f}\n")
                f.write(f"Coordinates logged to: {self.csv_file}\n")
                f.write(f"{'='*80}\n")
        
        print("\nReal-time recognition stopped.")
        print(f"\nDebug data saved to:")
        print(f"  - {self.log_file} (detailed logs)")
        print(f"  - {self.csv_file} (coordinate data)")


def main():
    # Paths for latest trained 5-class run
    model_path = "models/saved_models/tcn_transformer_20260417_011511/best_model.h5"
    metadata_path = "data/extracted_landmarks/metadata.json"
    normalization_path = "models/saved_models/tcn_transformer_20260417_011511/normalization_params.json"
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    if not Path(metadata_path).exists():
        print(f"Error: Metadata not found at {metadata_path}")
        return
    
    if not Path(normalization_path).exists():
        print(f"Error: Normalization parameters not found at {normalization_path}")
        return
    
    # Create recognizer
    recognizer = RealtimePSLRecognizer(
        model_path=model_path,
        metadata_path=metadata_path,
        normalization_path=normalization_path,
        sequence_length=60,
        confidence_threshold=0.6
    )
    
    # Run
    recognizer.run()


if __name__ == "__main__":
    main()
