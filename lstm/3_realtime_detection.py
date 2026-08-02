"""
Step 3: Real-time Sign Language Recognition via Webcam
Professional implementation with smooth predictions and visual feedback
"""
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from collections import deque
import time

# Configuration
MODEL_PATH = 'models/sign_language_model_final.h5'
CLASSES_PATH = 'processed_data/classes.json'
SEQ_LENGTH = 30
FEATURE_VECTOR_SIZE = 1662

# Prediction settings
CONFIDENCE_THRESHOLD = 0.70  # Higher threshold for confident predictions
SMOOTHING_WINDOW = 2  # Minimal smoothing for faster response
RESET_THRESHOLD = 0.40  # If confidence drops below this, reset to "waiting"
NO_HAND_RESET_FRAMES = 5  # Reset after N frames with no hands detected

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class SignLanguageDetector:
    """Real-time sign language detection with smooth predictions."""
    
    def __init__(self, model_path, classes_path):
        """Initialize detector with model and classes."""
        print("🚀 Initializing Sign Language Detector...")
        
        # Load model
        print(f"  Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("  ✓ Model loaded")
        
        # Load classes
        with open(classes_path, 'r') as f:
            self.classes = json.load(f)
        print(f"  ✓ Classes loaded: {self.classes}")
        
        # Initialize sequence buffer
        self.sequence = []
        self.predictions_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.last_prediction = None
        self.last_confidence = 0.0
        self.no_hand_counter = 0  # Count frames without hands
        self.frame_skip_counter = 0  # For processing every N frames
        
        # MediaPipe holistic
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✅ Detector ready!\n")
    
    def mediapipe_detection(self, image):
        """Process frame with MediaPipe."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def has_hands_detected(self, results):
        """Check if any hands are detected."""
        return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
    
    def extract_keypoints(self, results):
        """Extract and normalize keypoints from MediaPipe results."""
        if not results.pose_landmarks:
            return np.zeros(FEATURE_VECTOR_SIZE)

        pose_lm = results.pose_landmarks.landmark
        
        # Anchor and scale
        anchor = np.array([pose_lm[0].x, pose_lm[0].y, pose_lm[0].z])
        s_left = np.array([pose_lm[11].x, pose_lm[11].y, pose_lm[11].z])
        s_right = np.array([pose_lm[12].x, pose_lm[12].y, pose_lm[12].z])
        scale = np.linalg.norm(s_left - s_right)

        if scale < 1e-6:
            return np.zeros(FEATURE_VECTOR_SIZE)

        # Pose
        pose_coords = np.array([[lm.x, lm.y, lm.z] for lm in pose_lm])
        pose_visibility = np.array([[lm.visibility] for lm in pose_lm])
        normalized_pose_coords = (pose_coords - anchor) / scale
        pose = np.concatenate([normalized_pose_coords, pose_visibility], axis=1).flatten()

        # Face
        if results.face_landmarks:
            face_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
            normalized_face_coords = (face_coords - anchor) / scale
            face = normalized_face_coords.flatten()
        else:
            face = np.zeros(468 * 3)

        # Left hand
        if results.left_hand_landmarks:
            lh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            normalized_lh_coords = (lh_coords - anchor) / scale
            left_hand = normalized_lh_coords.flatten()
        else:
            left_hand = np.zeros(21 * 3)

        # Right hand
        if results.right_hand_landmarks:
            rh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            normalized_rh_coords = (rh_coords - anchor) / scale
            right_hand = normalized_rh_coords.flatten()
        else:
            right_hand = np.zeros(21 * 3)

        return np.concatenate([pose, face, left_hand, right_hand])
    
    def draw_landmarks(self, image, results):
        """Draw MediaPipe landmarks on image."""
        # Draw pose
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Draw face (simplified)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Draw hands
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style()
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style()
        )
    
    def draw_ui(self, image, prediction_text, confidence, fps):
        """Draw UI elements on frame."""
        h, w, _ = image.shape
        
        # Create semi-transparent overlay for top bar
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        # Determine color based on state
        if "Waiting" in prediction_text or "Collecting" in prediction_text or "Uncertain" in prediction_text:
            text_color = (150, 150, 150)
        elif confidence >= CONFIDENCE_THRESHOLD:
            text_color = (0, 255, 0)  # Green for confident
        else:
            text_color = (0, 200, 255)  # Orange for uncertain
        
        # Draw prediction
        bar_width = int(w * 0.8)
        bar_x = int(w * 0.1)
        
        if confidence > 0:
            # Confidence bar background
            cv2.rectangle(image, (bar_x, 80), (bar_x + bar_width, 105), (50, 50, 50), -1)
            
            # Filled portion based on confidence
            fill_width = int(bar_width * min(confidence, 1.0))
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(image, (bar_x, 80), (bar_x + fill_width, 105), color, -1)
            
            # Confidence percentage
            conf_text = f"{confidence*100:.0f}%"
            cv2.putText(image, conf_text, (bar_x + bar_width + 10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Prediction text (larger and clearer)
        display_text = prediction_text.replace(" (?)", "").replace("...", "")
        cv2.putText(image, display_text, (bar_x, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
        
        # FPS counter
        cv2.putText(image, f"FPS: {fps:.1f}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(image, "Press 'P' to quit", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image
    
    def predict(self, sequence):
        """Make prediction with minimal smoothing for fast response."""
        # Reshape for model input
        sequence = np.expand_dims(sequence, axis=0)
        
        # Get prediction
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Store raw prediction
        self.last_prediction = self.classes[predicted_class]
        self.last_confidence = confidence
        
        # Only use smoothing if confidence is borderline
        if CONFIDENCE_THRESHOLD * 0.8 < confidence < CONFIDENCE_THRESHOLD * 1.2:
            # Add to buffer for smoothing
            self.predictions_buffer.append(predictions)
            
            if len(self.predictions_buffer) >= SMOOTHING_WINDOW:
                # Average recent predictions
                buffer_array = np.array(self.predictions_buffer)
                avg_predictions = np.mean(buffer_array, axis=0)
                smoothed_class = np.argmax(avg_predictions)
                smoothed_confidence = avg_predictions[smoothed_class]
                
                return self.classes[smoothed_class], smoothed_confidence
        else:
            # Clear buffer for fresh start on confident or very uncertain predictions
            self.predictions_buffer.clear()
        
        return self.classes[predicted_class], confidence
    
    def run(self):
        """Main detection loop."""
        print("="*60)
        print("🎥 Starting webcam detection...")
        print("="*60)
        print("\n📌 Instructions:")
        print("  - Perform sign language gestures in front of the camera")
        print("  - Wait for the sequence buffer to fill (30 frames)")
        print("  - Predictions will appear at the top")
        print("  - Press 'P' to quit\n")
        
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("✅ Webcam started successfully!\n")
        
        # FPS calculation
        fps_time = time.time()
        fps = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                image, results = self.mediapipe_detection(frame)
                
                # Draw landmarks
                self.draw_landmarks(image, results)
                
                # Check if hands are detected
                hands_detected = self.has_hands_detected(results)
                
                if not hands_detected:
                    self.no_hand_counter += 1
                else:
                    self.no_hand_counter = 0
                
                # Reset sequence if no hands for several frames
                if self.no_hand_counter >= NO_HAND_RESET_FRAMES:
                    self.sequence = []
                    self.predictions_buffer.clear()
                    prediction_text = "Waiting for sign..."
                    confidence = 0.0
                else:
                    # Extract keypoints
                    keypoints = self.extract_keypoints(results)
                    self.sequence.append(keypoints)
                    
                    # Keep only last SEQ_LENGTH frames
                    self.sequence = self.sequence[-SEQ_LENGTH:]
                    
                    # Make prediction if we have enough frames
                    prediction_text = "Collecting..."
                    confidence = 0.0
                    
                    if len(self.sequence) == SEQ_LENGTH:
                        # Only predict every other frame for speed
                        self.frame_skip_counter += 1
                        if self.frame_skip_counter % 1 == 0:  # Can change to 2 for even faster
                            prediction_text, confidence = self.predict(np.array(self.sequence))
                            
                            # Reset if confidence too low
                            if confidence < RESET_THRESHOLD:
                                prediction_text = "Uncertain..."
                            elif confidence < CONFIDENCE_THRESHOLD:
                                prediction_text = f"{prediction_text} (?)"
                
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - fps_time)
                fps_time = current_time
                
                # Draw UI
                image = self.draw_ui(image, prediction_text, confidence, fps)
                
                # Show frame
                cv2.imshow('Sign Language Detection', image)
                
                # Check for quit - changed to 'p' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p') or key == ord('P'):
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ Detection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
            print("\n✅ Webcam detection stopped")
            print("="*60)

def main():
    """Main function."""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using 2_train_model.py")
        return
    
    if not os.path.exists(CLASSES_PATH):
        print(f"❌ Error: Classes file not found at {CLASSES_PATH}")
        print("Please extract data first using 1_extract_data.py")
        return
    
    # Create detector and run
    detector = SignLanguageDetector(MODEL_PATH, CLASSES_PATH)
    detector.run()

if __name__ == "__main__":
    main()
