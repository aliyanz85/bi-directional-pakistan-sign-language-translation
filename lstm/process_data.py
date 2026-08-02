import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
import json

DATA_PATH = os.path.join(os.getcwd(), 'videosDataset') 

# 2. List of classes 
CLASSES = np.array([folder for folder in os.listdir(DATA_PATH) 
                    if os.path.isdir(os.path.join(DATA_PATH, folder))])

# Number of frames
SEQ_LENGTH = 30

# Name of the output file
OUTPUT_FILE = os.path.join(os.getcwd(), 'data.npz')

#Name of the file to save your class labels
CLASSES_FILE = os.path.join(os.getcwd(), 'classes.json')

# Feature Vector Size
# Pose: 33 * 4 = 132
# Face: 468 * 3 = 1404
# Left Hand: 21 * 3 = 63
# Right Hand: 21 * 3 = 63
# TOTAL = 132 + 1404 + 63 + 63 = 1662 features per frame
FEATURE_VECTOR_SIZE = 1662 


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    """
    Processes an image with the MediaPipe Holistic model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB
    image.flags.writeable = False                  
    results = model.process(image)                
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB to BGR
    return image, results

def extract_keypoints(results):
    """
    Extracts, flattens, and NORMALIZES keypoints from MediaPipe results.
    
    Normalization makes the data position- and scale-invariant, which is
    CRITICAL for a robust sign language model.
    
    Returns a single flat array of 1662 features or zeros if no pose.
    """
    
    if not results.pose_landmarks:
        return np.zeros(FEATURE_VECTOR_SIZE)

    pose_lm = results.pose_landmarks.landmark
    
    # Anchor Point
    anchor = np.array([pose_lm[0].x, pose_lm[0].y, pose_lm[0].z])
    
    # Scale Factor
    s_left = np.array([pose_lm[11].x, pose_lm[11].y, pose_lm[11].z])
    s_right = np.array([pose_lm[12].x, pose_lm[12].y, pose_lm[12].z])
    scale = np.linalg.norm(s_left - s_right)

    if scale < 1e-6:
        return np.zeros(FEATURE_VECTOR_SIZE)

    # Normalize POSE (33 landmarks)
    pose_coords = np.array([[lm.x, lm.y, lm.z] for lm in pose_lm])
    pose_visibility = np.array([[lm.visibility] for lm in pose_lm])
    
    # Normalize x, y, z by subtracting anchor and dividing by scale
    normalized_pose_coords = (pose_coords - anchor) / scale
    pose = np.concatenate([normalized_pose_coords, pose_visibility], axis=1).flatten()

    # Normalize FACE (468 landmarks)
    if results.face_landmarks:
        face_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        normalized_face_coords = (face_coords - anchor) / scale
        face = normalized_face_coords.flatten()
    else:
        face = np.zeros(468 * 3) # 1404

    # Normalize LEFT HAND (21 landmarks)
    if results.left_hand_landmarks:
        lh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        normalized_lh_coords = (lh_coords - anchor) / scale
        left_hand = normalized_lh_coords.flatten()
    else:
        left_hand = np.zeros(21 * 3) # 63

    # Normalize RIGHT HAND (21 landmarks)
    if results.right_hand_landmarks:
        rh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        normalized_rh_coords = (rh_coords - anchor) / scale
        right_hand = normalized_rh_coords.flatten()
    else:
        right_hand = np.zeros(21 * 3) # 63

    # Concatenate all features into a single vector
    return np.concatenate([pose, face, left_hand, right_hand])

# --- sample_frames function ---
def sample_frames(video_frames, seq_length):

    # Samples a fixed number of frames from a list of frames
    
    if len(video_frames) > seq_length:
        # If more frames than seq_length, sample evenly
        indices = np.linspace(0, len(video_frames) - 1, seq_length, dtype=int)
        sampled_frames = [video_frames[i] for i in indices]
    else:
        # If fewer frames, pad with zero vectors
        sampled_frames = video_frames
        
        # Calculate number of frames to pad
        pad_len = seq_length - len(video_frames)
        feature_length = FEATURE_VECTOR_SIZE 
        for _ in range(pad_len):
            sampled_frames.append(np.zeros(feature_length))
            
    return np.array(sampled_frames)

def process_data():
    """
    Main function to iterate through videos, extract keypoints,
    sample/pad sequences, and save the final dataset.
    """
    print(f"Starting data processing...")
    print(f"Found {len(CLASSES)} classes: {CLASSES}")
    
    # Save the class list for use during prediction
    with open(CLASSES_FILE, 'w') as f:
        json.dump(CLASSES.tolist(), f)
    print(f"Saved class labels to {CLASSES_FILE}")

    # Create a mapping of class name to integer label
    class_map = {class_name: index for index, class_name in enumerate(CLASSES)}
    
    # Lists to hold all sequences (X) and labels (y)
    sequences, labels = [], []
    
    # Use MediaPipe Holistic model
    # Note: static_image_mode=False is default, which is correct for video
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop over each class (e.g., "hello")
        for class_name in tqdm(CLASSES, desc="Processing Classes"):
            class_index = class_map[class_name]
            class_path = os.path.join(DATA_PATH, class_name)
            
            if not os.path.isdir(class_path):
                continue
            
            # Loop over each video in the class folder
            for video_name in tqdm(os.listdir(class_path), desc=f"Videos for {class_name}", leave=False):
                video_path = os.path.join(class_path, video_name)
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"   [Warning] Could not open video: {video_path}")
                    continue
                
                # List to hold all frames from *this* video
                video_frames = []
                
                # Read frame by frame
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break # End of video
                        
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Extract, normalize, and flatten keypoints
                    keypoints = extract_keypoints(results)
                    video_frames.append(keypoints)

                cap.release()
                
                if not video_frames:
                    print(f"   [Warning] No frames extracted from video: {video_path}")
                    continue

                # Sample/pad the frames to SEQ_LENGTH
                sequence = sample_frames(video_frames, SEQ_LENGTH)
                
                # Add the sequence and its label to our lists
                sequences.append(sequence)
                labels.append(class_index)

    print(f"\nTotal sequences processed: {len(sequences)}")

    if not sequences:
        print("[Error] No sequences were processed! Check video paths and content.")
        return

    # Convert lists to NumPy arrays
    X = np.array(sequences)
    
    # One-hot encode the labels (e.g., 2 -> [0, 0, 1, 0, ...])
    # This is the correct format for 'categorical_crossentropy' loss
    y = np.eye(len(CLASSES))[labels] 
    
    print(f"Shape of X (sequences): {X.shape}")
    print(f"Shape of y (labels): {y.shape}")

    # Save the final dataset
    try:
        np.savez(OUTPUT_FILE, X=X, y=y)
        print(f"Successfully saved processed data to {OUTPUT_FILE}")
    except Exception as e:
        print(f"[Error] Failed to save data: {e}")

# --- Run the script ---
if __name__ == "__main__":
    
    # Check if dataset path exists
    if not os.path.exists(DATA_PATH):
        print(f"[Error] Dataset folder not found at: {DATA_PATH}")
        print("Please check the DATA_PATH variable in the script.")
    elif len(CLASSES) == 0:
        print(f"[Error] No class folders found in {DATA_PATH}.")
        print("Ensure your 'videosDataset' folder contains subfolders like 'hello', 'thanks', etc.")
    else:
        process_data()