import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm # A great progress bar!
import json

# --- Configuration ---

# 1. Path to your dataset folder (the one containing "hello", "thanks", etc.)
DATA_PATH = os.path.join(os.getcwd(), 'videosDataset') 

# 2. List of classes (signs)
CLASSES = np.array([folder for folder in os.listdir(DATA_PATH) 
                    if os.path.isdir(os.path.join(DATA_PATH, folder))])

# 3. Number of frames to sample from each video
SEQ_LENGTH = 30

# 4. Name of the output file
OUTPUT_FILE = os.path.join(os.getcwd(), 'data.npz')

# 5. Name of the file to save your class labels
CLASSES_FILE = os.path.join(os.getcwd(), 'classes.json')

# 6. Name of the file to save detailed frame-by-frame landmarks for evaluation
EVALUATION_LANDMARKS_FILE = os.path.join(os.getcwd(), 'alert_frame_landmarks.json')

# --- NEW: Define Feature Vector Size ---
# This is crucial for consistency
# Pose: 33 * 4 = 132
# Face: 468 * 3 = 1404
# Left Hand: 21 * 3 = 63
# Right Hand: 21 * 3 = 63
# TOTAL = 132 + 1404 + 63 + 63 = 1662 features per frame
FEATURE_VECTOR_SIZE = 1662 


# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    """
    Processes an image with the MediaPipe Holistic model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB to BGR
    return image, results

# --- UPDATED: extract_keypoints function ---
def extract_keypoints(results):
    """
    Extracts, flattens, and NORMALIZES keypoints from MediaPipe results.
    
    Normalization makes the data position- and scale-invariant, which is
    CRITICAL for a robust sign language model.
    
    Returns a single flat array of 1662 features or zeros if no pose.
    """
    
    # 1. Check for Pose - Our anchor depends on it.
    if not results.pose_landmarks:
        # Return a zero vector if no pose is detected (bad frame)
        return np.zeros(FEATURE_VECTOR_SIZE)

    # 2. Get Pose landmarks and find Anchor + Scale
    pose_lm = results.pose_landmarks.landmark
    
    # Anchor Point: Use the nose (index 0) to center all keypoints
    anchor = np.array([pose_lm[0].x, pose_lm[0].y, pose_lm[0].z])
    
    # Scale Factor: Use inter-shoulder distance to normalize for scale
    s_left = np.array([pose_lm[11].x, pose_lm[11].y, pose_lm[11].z])
    s_right = np.array([pose_lm[12].x, pose_lm[12].y, pose_lm[12].z])
    scale = np.linalg.norm(s_left - s_right)

    # Handle edge case (division by zero if shoulders aren't detected)
    if scale < 1e-6:
        return np.zeros(FEATURE_VECTOR_SIZE)

    # 3. Extract and Normalize POSE (33 landmarks)
    # We keep x, y, z, visibility for pose
    pose_coords = np.array([[lm.x, lm.y, lm.z] for lm in pose_lm])
    pose_visibility = np.array([[lm.visibility] for lm in pose_lm])
    
    # Normalize x, y, z by subtracting anchor and dividing by scale
    normalized_pose_coords = (pose_coords - anchor) / scale
    
    # Re-combine with visibility
    pose = np.concatenate([normalized_pose_coords, pose_visibility], axis=1).flatten()

    # 4. Extract and Normalize FACE (468 landmarks)
    if results.face_landmarks:
        face_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        normalized_face_coords = (face_coords - anchor) / scale
        face = normalized_face_coords.flatten()
    else:
        face = np.zeros(468 * 3) # 1404

    # 5. Extract and Normalize LEFT HAND (21 landmarks)
    if results.left_hand_landmarks:
        lh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        normalized_lh_coords = (lh_coords - anchor) / scale
        left_hand = normalized_lh_coords.flatten()
    else:
        left_hand = np.zeros(21 * 3) # 63

    # 6. Extract and Normalize RIGHT HAND (21 landmarks)
    if results.right_hand_landmarks:
        rh_coords = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        normalized_rh_coords = (rh_coords - anchor) / scale
        right_hand = normalized_rh_coords.flatten()
    else:
        right_hand = np.zeros(21 * 3) # 63

    # 7. Concatenate all features into a single vector
    return np.concatenate([pose, face, left_hand, right_hand])

# --- UPDATED: sample_frames function ---
def sample_frames(video_frames, seq_length):
    """
    Samples a fixed number of frames (seq_length) from a list of frames.
    
    - If video has more frames, it samples evenly.
    - If video has fewer frames, it pads with zeros.
    """
    if len(video_frames) > seq_length:
        # If more frames than seq_length, sample evenly
        indices = np.linspace(0, len(video_frames) - 1, seq_length, dtype=int)
        sampled_frames = [video_frames[i] for i in indices]
    else:
        # If fewer frames, pad with zero vectors
        sampled_frames = video_frames
        
        # Calculate number of frames to pad
        pad_len = seq_length - len(video_frames)
        
        # Get the feature vector length (use our global constant)
        # This is safer than checking the first frame
        feature_length = FEATURE_VECTOR_SIZE 
        
        # Append zero frames
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
    
    # Dictionary to store frame-by-frame landmarks for first 2 "alert" videos (for FYP evaluation)
    evaluation_data = {
        "class": "alert",
        "videos": []
    }
    
    # Use MediaPipe Holistic model
    # Note: static_image_mode=False is default, which is correct for video
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop over each class (e.g., "hello")
        for class_name in tqdm(CLASSES, desc="Processing Classes"):
            class_index = class_map[class_name]
            class_path = os.path.join(DATA_PATH, class_name)
            
            if not os.path.isdir(class_path):
                continue

            # Counter for "alert" videos (for FYP evaluation)
            alert_video_count = 0
            
            # Loop over each video in the class folder
            for video_name in tqdm(os.listdir(class_path), desc=f"Videos for {class_name}", leave=False):
                video_path = os.path.join(class_path, video_name)
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"   [Warning] Could not open video: {video_path}")
                    continue
                
                # List to hold all frames from *this* video
                video_frames = []
                
                # For FYP evaluation: Store detailed frame data for first 2 "alert" videos
                store_detailed_landmarks = (class_name == "alert" and alert_video_count < 2)
                detailed_frames = [] if store_detailed_landmarks else None
                
                # Read frame by frame
                frame_number = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break # End of video
                        
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Extract, normalize, and flatten keypoints
                    keypoints = extract_keypoints(results)
                    video_frames.append(keypoints)
                    
                    # Store detailed landmark data for FYP evaluation
                    if store_detailed_landmarks:
                        frame_data = {
                            "frame_number": frame_number,
                            "landmarks_vector": keypoints.tolist(),
                            "vector_size": len(keypoints)
                        }
                        detailed_frames.append(frame_data)
                    
                    frame_number += 1

                cap.release()
                
                # Save detailed frame data for first 2 "alert" videos
                if store_detailed_landmarks:
                    video_info = {
                        "video_name": video_name,
                        "total_frames": len(video_frames),
                        "frames": detailed_frames
                    }
                    evaluation_data["videos"].append(video_info)
                    alert_video_count += 1
                    print(f"\n   [FYP Evaluation] Stored {len(detailed_frames)} frames from {video_name}")
                
                
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
    
    # Save detailed frame-by-frame landmarks for FYP evaluation
    if evaluation_data["videos"]:
        try:
            with open(EVALUATION_LANDMARKS_FILE, 'w') as f:
                json.dump(evaluation_data, f, indent=2)
            print(f"\n[FYP Evaluation] Successfully saved frame-by-frame landmarks to {EVALUATION_LANDMARKS_FILE}")
            print(f"[FYP Evaluation] Stored data for {len(evaluation_data['videos'])} 'alert' videos")
            for video in evaluation_data["videos"]:
                print(f"  - {video['video_name']}: {video['total_frames']} frames")
        except Exception as e:
            print(f"[Error] Failed to save evaluation landmarks: {e}")

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