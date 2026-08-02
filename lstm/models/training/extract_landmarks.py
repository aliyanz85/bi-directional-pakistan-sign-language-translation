"""
Landmark extraction / preprocessing utilities using MediaPipe.
Provides frame sampling, keypoint extraction and a `process_dataset` helper to build
`processed_data/dataset.npz` and `processed_data/classes.json`.
"""
import os
import cv2
import mediapipe as mp
import numpy as np
import json
from tqdm import tqdm

DATA_PATH = os.path.join(os.getcwd(), 'videosDataset')
OUTPUT_DIR = os.path.join(os.getcwd(), 'processed_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LENGTH = 30
FEATURE_VECTOR_SIZE = 1662
mp_holistic = mp.solutions.holistic


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
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


def sample_frames(video_frames, seq_length=SEQ_LENGTH):
    if len(video_frames) > seq_length:
        indices = np.linspace(0, len(video_frames) - 1, seq_length, dtype=int)
        sampled_frames = [video_frames[i] for i in indices]
    else:
        sampled_frames = video_frames.copy()
        pad_len = seq_length - len(video_frames)
        for _ in range(pad_len):
            sampled_frames.append(np.zeros(FEATURE_VECTOR_SIZE))
    return np.array(sampled_frames)


def process_dataset(selected_classes=None):
    classes = selected_classes if selected_classes is not None else [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    sequences, labels = [], []
    class_map = {c: i for i, c in enumerate(classes)}

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for class_name in classes:
            class_idx = class_map[class_name]
            class_path = os.path.join(DATA_PATH, class_name)
            if not os.path.isdir(class_path):
                continue
            videos = [v for v in os.listdir(class_path) if v.endswith('.mp4')]
            for v in tqdm(videos, desc=f"Processing {class_name}"):
                cap = cv2.VideoCapture(os.path.join(class_path, v))
                if not cap.isOpened():
                    continue
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, results = mediapipe_detection(frame, holistic)
                    frames.append(extract_keypoints(results))
                cap.release()
                if not frames:
                    continue
                seq = sample_frames(frames)
                sequences.append(seq)
                labels.append(class_idx)

    X = np.array(sequences)
    y = np.array(labels)

    # stratified split if possible
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    np.savez(os.path.join(OUTPUT_DIR, 'dataset.npz'), X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    with open(os.path.join(OUTPUT_DIR, 'classes.json'), 'w') as f:
        json.dump(classes, f)
    print('Saved processed_data/dataset.npz and classes.json')


if __name__ == '__main__':
    # default: process only good and funny (matches repo extract script)
    process_dataset(selected_classes=['good', 'funny'])
