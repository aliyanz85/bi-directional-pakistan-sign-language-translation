"""
Standalone real-time inference script (adapted from `3_realtime_detection.py`).
Opens webcam, uses `models/sign_language_model_final.h5` and `processed_data/classes.json`.
"""
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from collections import deque
import time
from tensorflow import keras

MODEL_PATH = 'models/sign_language_model_final.h5'
CLASSES_PATH = 'processed_data/classes.json'
SEQ_LENGTH = 30
FEATURE_VECTOR_SIZE = 1662
CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 2
NO_HAND_RESET_FRAMES = 5

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class LiveDetector:
    def __init__(self, model_path=MODEL_PATH, classes_path=CLASSES_PATH):
        self.model = keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.classes = json.load(f)
        self.sequence = []
        self.predictions_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.no_hand_counter = 0
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def extract_keypoints(self, results):
        if not results.pose_landmarks:
            return np.zeros(FEATURE_VECTOR_SIZE)
        pose_lm = results.pose_landmarks.landmark
        anchor = np.array([pose_lm[0].x, pose_lm[0].y, pose_lm[0].z])
        s_left = np.array([pose_lm[11].x, pose_lm[11].y, pose_lm[11].z])
        s_right = np.array([pose_lm[12].x, pose_lm[12].y, pose_lm[12].z])
        scale = np.linalg.norm(s_left - s_right)
        if scale < 1e-6:
            return np.zeros(FEATURE_VECTOR_SIZE)
        pose_coords = np.array([[lm.x, lm.y, lm.z] for lm in pose_lm])
        pose_visibility = np.array([[lm.visibility] for lm in pose_lm])
        normalized_pose_coords = (pose_coords - anchor) / scale
        pose = np.concatenate([normalized_pose_coords, pose_visibility], axis=1).flatten()
        if results.face_landmarks:
            face_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
            normalized_face_coords = (face_coords - anchor) / scale
            face = normalized_face_coords.flatten()
        else:
            face = np.zeros(468 * 3)
        if results.left_hand_landmarks:
            lh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            normalized_lh_coords = (lh_coords - anchor) / scale
            left_hand = normalized_lh_coords.flatten()
        else:
            left_hand = np.zeros(21 * 3)
        if results.right_hand_landmarks:
            rh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            normalized_rh_coords = (rh_coords - anchor) / scale
            right_hand = normalized_rh_coords.flatten()
        else:
            right_hand = np.zeros(21 * 3)
        return np.concatenate([pose, face, left_hand, right_hand])

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        fps_time = time.time()
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                image, results = self.mediapipe_detection(frame)
                keypoints = self.extract_keypoints(results)
                hands_detected = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
                if not hands_detected:
                    self.no_hand_counter += 1
                else:
                    self.no_hand_counter = 0
                if self.no_hand_counter >= NO_HAND_RESET_FRAMES:
                    self.sequence = []
                    prediction_text = 'Waiting...'
                    confidence = 0.0
                else:
                    self.sequence.append(keypoints)
                    self.sequence = self.sequence[-SEQ_LENGTH:]
                    prediction_text = 'Collecting...'
                    confidence = 0.0
                    if len(self.sequence) == SEQ_LENGTH:
                        seq = np.expand_dims(np.array(self.sequence), axis=0)
                        preds = self.model.predict(seq, verbose=0)[0]
                        pred_class = preds.argmax()
                        confidence = preds[pred_class]
                        prediction_text = self.classes[pred_class]
                        if confidence < CONFIDENCE_THRESHOLD:
                            prediction_text += ' (?)'
                fps = 1.0 / (time.time() - fps_time)
                fps_time = time.time()
                cv2.putText(image, f"{prediction_text} {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
                cv2.putText(image, f"FPS: {fps:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                cv2.imshow('Inference', image)
                if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()


if __name__ == '__main__':
    d = LiveDetector()
    d.run()
