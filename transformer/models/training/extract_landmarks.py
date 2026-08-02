"""
MediaPipe Landmark Extraction Pipeline
Extracts robust hand landmark features from video files
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
import pickle


class LandmarkExtractor:
    """Extract hand landmarks from videos using MediaPipe"""
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.6,
                 static_image_mode: bool = False):
        """
        Initialize MediaPipe Hands
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            static_image_mode: Whether to treat each frame independently
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def extract_from_video(self, video_path: str, 
                          visualize: bool = False) -> Optional[np.ndarray]:
        """
        Extract landmark sequence from a video file
        
        Args:
            video_path: Path to video file
            visualize: Whether to show visualization
        
        Returns:
            Array of shape (num_frames, feature_dim) or None if extraction fails
        """
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None
        
        landmarks_sequence = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Extract features from detected hands
                frame_features = self._process_landmarks(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                    frame.shape
                )
                landmarks_sequence.append(frame_features)
                
                # Visualization
                if visualize:
                    annotated_frame = frame.copy()
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                    cv2.imshow('Hand Landmarks', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                # No hands detected - use zero padding or interpolation
                if len(landmarks_sequence) > 0:
                    # Repeat last valid frame
                    landmarks_sequence.append(landmarks_sequence[-1])
                else:
                    # Use zeros if no hands detected yet
                    landmarks_sequence.append(np.zeros(self._get_feature_dim()))
            
            frame_count += 1
        
        cap.release()
        if visualize:
            cv2.destroyAllWindows()
        
        if len(landmarks_sequence) == 0:
            print(f"No landmarks extracted from {video_path}")
            return None
        
        return np.array(landmarks_sequence)
    
    def _process_landmarks(self, 
                          hand_landmarks_list: List,
                          handedness_list: List,
                          frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Process detected hand landmarks into feature vector
        
        Args:
            hand_landmarks_list: List of detected hand landmarks
            handedness_list: List of hand labels (left/right)
            frame_shape: Shape of the frame (height, width, channels)
        
        Returns:
            Feature vector for this frame
        """
        features = []
        
        # Sort hands by handedness (left first, then right)
        hands_data = list(zip(hand_landmarks_list, handedness_list))
        hands_data.sort(key=lambda x: x[1].classification[0].label)
        
        for hand_lms, handedness in hands_data[:2]:  # Max 2 hands
            # Extract raw coordinates (21 landmarks × 3 coordinates)
            raw_coords = np.array([[lm.x, lm.y, lm.z] 
                                   for lm in hand_lms.landmark])
            
            # Normalize relative to wrist (translation invariance)
            wrist = raw_coords[0]
            relative_coords = raw_coords - wrist
            
            # Compute hand geometry features
            geometry = self._compute_hand_geometry(raw_coords)
            
            # Hand label (left=1,0 / right=0,1)
            hand_label = handedness.classification[0].label
            hand_onehot = np.array([1, 0] if hand_label == 'Left' else [0, 1])
            
            # Combine all features for this hand
            hand_features = np.concatenate([
                relative_coords.flatten(),  # 63 features (21 points × 3)
                geometry,                   # ~30 geometric features
                hand_onehot                 # 2 features
            ])
            
            features.append(hand_features)
        
        # Handle single hand case (pad with zeros for second hand)
        if len(features) == 1:
            single_hand_dim = len(features[0])
            features.append(np.zeros(single_hand_dim))
        elif len(features) == 0:
            # No hands detected (shouldn't happen, but safety check)
            features = [np.zeros(self._get_feature_dim() // 2)] * 2
        
        # Concatenate both hands
        return np.concatenate(features)
    
    def _compute_hand_geometry(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute rotation and scale invariant geometric features
        
        Args:
            landmarks: Array of shape (21, 3) with landmark coordinates
        
        Returns:
            Array of geometric features
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
        # Ratio of tip-to-wrist distance vs base-to-wrist distance
        finger_bases = [2, 5, 9, 13, 17]  # Base of each finger
        for tip_idx, base_idx in zip(fingertips, finger_bases):
            tip_dist = np.linalg.norm(landmarks[tip_idx] - wrist_pos)
            base_dist = np.linalg.norm(landmarks[base_idx] - wrist_pos)
            ratio = tip_dist / (base_dist + 1e-6)
            geometry.append(ratio)
        
        # 5. Finger angles (5 features)
        # Angle between consecutive finger segments
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
        
        return np.array(geometry)
    
    def _get_feature_dim(self) -> int:
        """Calculate total feature dimension"""
        # Per hand: 63 (coords) + 29 (geometry) + 2 (handedness) = 94
        # Two hands: 94 * 2 = 188
        return 188
    
    def normalize_sequence_length(self, sequence: np.ndarray, 
                                  target_length: int = 60) -> np.ndarray:
        """
        Normalize sequence to fixed length using interpolation
        
        Args:
            sequence: Input sequence of variable length
            target_length: Desired sequence length
        
        Returns:
            Normalized sequence of shape (target_length, feature_dim)
        """
        current_length = len(sequence)
        
        if current_length == target_length:
            return sequence
        
        # Interpolate each feature dimension
        old_indices = np.linspace(0, current_length - 1, current_length)
        new_indices = np.linspace(0, current_length - 1, target_length)
        
        normalized = np.zeros((target_length, sequence.shape[1]))
        
        for feat_idx in range(sequence.shape[1]):
            normalized[:, feat_idx] = np.interp(
                new_indices, 
                old_indices, 
                sequence[:, feat_idx]
            )
        
        return normalized
    
    def extract_from_dataset(self, 
                            data_dir: str,
                            output_dir: str,
                            target_length: int = 60,
                            visualize: bool = False) -> Dict:
        """
        Extract landmarks from entire dataset
        
        Args:
            data_dir: Directory containing video files organized by class
            output_dir: Directory to save extracted landmarks
            target_length: Target sequence length
            visualize: Whether to visualize extraction
        
        Returns:
            Dictionary with extraction statistics
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'classes': {}
        }
        
        # Get all class directories
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        print(f"Found {len(class_dirs)} classes")
        print("=" * 60)
        
        all_sequences = []
        all_labels = []
        class_names = []
        
        for class_idx, class_dir in enumerate(sorted(class_dirs)):
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Get all video files in this class
            video_files = list(class_dir.glob('*.mp4')) + \
                         list(class_dir.glob('*.avi')) + \
                         list(class_dir.glob('*.mov'))
            
            stats['classes'][class_name] = {
                'total': len(video_files),
                'successful': 0,
                'failed': 0
            }
            
            print(f"\nProcessing class: {class_name} ({len(video_files)} videos)")
            
            for video_file in tqdm(video_files, desc=f"  {class_name}"):
                stats['total_videos'] += 1
                
                # Extract landmarks
                sequence = self.extract_from_video(str(video_file), visualize)
                
                if sequence is not None and len(sequence) > 0:
                    # Normalize length
                    normalized_seq = self.normalize_sequence_length(sequence, target_length)
                    
                    all_sequences.append(normalized_seq)
                    all_labels.append(class_idx)
                    
                    stats['successful'] += 1
                    stats['classes'][class_name]['successful'] += 1
                    
                    # Save individual sequence
                    seq_filename = output_dir / f"{class_name}_{video_file.stem}.npy"
                    np.save(seq_filename, normalized_seq)
                else:
                    stats['failed'] += 1
                    stats['classes'][class_name]['failed'] += 1
                    print(f"  Failed to extract: {video_file.name}")
        
        # Save complete dataset
        dataset = {
            'sequences': np.array(all_sequences),
            'labels': np.array(all_labels),
            'class_names': class_names,
            'feature_dim': self._get_feature_dim(),
            'sequence_length': target_length
        }
        
        dataset_file = output_dir / 'dataset.pkl'
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Save metadata
        metadata = {
            'num_classes': len(class_names),
            'class_names': class_names,
            'num_samples': len(all_sequences),
            'feature_dim': self._get_feature_dim(),
            'sequence_length': target_length,
            'extraction_stats': stats
        }
        
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Total videos processed: {stats['total_videos']}")
        print(f"Successful extractions: {stats['successful']}")
        print(f"Failed extractions: {stats['failed']}")
        print(f"Success rate: {stats['successful']/stats['total_videos']*100:.1f}%")
        print(f"\nDataset saved to: {dataset_file}")
        print(f"Metadata saved to: {metadata_file}")
        
        return stats


class LandmarkNormalizer:
    """Normalize and preprocess extracted landmarks"""
    
    @staticmethod
    def standardize(sequences: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Standardize features to zero mean and unit variance
        
        Args:
            sequences: Array of shape (num_samples, seq_len, feature_dim)
        
        Returns:
            Standardized sequences and normalization parameters
        """
        # Reshape to (num_samples * seq_len, feature_dim)
        original_shape = sequences.shape
        flattened = sequences.reshape(-1, sequences.shape[-1])
        
        # Compute statistics
        mean = flattened.mean(axis=0)
        std = flattened.std(axis=0) + 1e-8
        
        # Standardize
        standardized = (flattened - mean) / std
        
        # Reshape back
        standardized = standardized.reshape(original_shape)
        
        params = {'mean': mean, 'std': std}
        
        return standardized, params
    
    @staticmethod
    def apply_normalization(sequences: np.ndarray, params: Dict) -> np.ndarray:
        """Apply saved normalization parameters"""
        original_shape = sequences.shape
        flattened = sequences.reshape(-1, sequences.shape[-1])
        
        normalized = (flattened - params['mean']) / params['std']
        
        return normalized.reshape(original_shape)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract landmarks from PSL videos')
    parser.add_argument('--data_dir', type=str, default='data/raw_videos',
                       help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='data/extracted_landmarks',
                       help='Directory to save extracted landmarks')
    parser.add_argument('--target_length', type=int, default=60,
                       help='Target sequence length')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize landmark extraction')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = LandmarkExtractor()
    
    # Extract landmarks
    stats = extractor.extract_from_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_length=args.target_length,
        visualize=args.visualize
    )
