import cv2
import mediapipe as mp
import time # To calculate FPS

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    """
    Processes an image with the MediaPipe Holistic model.
    """
    # Convert BGR (OpenCV) to RGB (MediaPipe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    # Set image to not writeable for performance
    image.flags.writeable = False                  
    
    # Make prediction
    results = model.process(image)                 
    
    # Set image back to writeable
    image.flags.writeable = True                   
    
    # Convert RGB back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_styled_landmarks(image, results):
    """
    Draws the landmarks on the image with custom styling for the demo.
    """
    # 1. Draw face connections (light green, subtle)
    #    FACEMESH_TESSELATION gives the full mesh
    mp_drawing.draw_landmarks(
        image, 
        results.face_landmarks, 
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None, # Don't draw individual face points
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(80,110,10), thickness=1, circle_radius=1
        )
    )

    # 2. Draw pose connections (purple)
    mp_drawing.draw_landmarks(
        image, 
        results.pose_landmarks, 
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(80,22,10), thickness=2, circle_radius=2
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(80,44,121), thickness=2, circle_radius=2
        )
    )

    # 3. Draw left hand connections (blue)
    mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(121,22,76), thickness=2, circle_radius=2
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(121,44,250), thickness=2, circle_radius=2
        )
    )

    # 4. Draw right hand connections (red/orange)
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(245,117,66), thickness=2, circle_radius=2
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(245,66,230), thickness=2, circle_radius=2
        )
    )

# --- Main Real-Time Loop ---
def main():
    # Use 0 for built-in webcam, 1 or 2 for external webcams
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("[Error] Cannot open webcam.")
        print("Please check if your webcam is connected and not in use by another application.")
        return

    print("Starting webcam feed... Press 'q' to quit.")

    # For calculating FPS
    prev_frame_time = 0
    new_frame_time = 0

    # Use MediaPipe Holistic model
    # min_detection_confidence: More sensitive detection (0.5 is default)
    # min_tracking_confidence: More robust tracking (0.5 is default)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            
            # Read feed
            ret, frame = cap.read()
            if not ret:
                print("[Info] Ignoring empty camera frame.")
                continue

            # --- IMPORTANT ---
            # Flip the frame horizontally for a "mirror" view.
            # This makes it intuitive to use.
            frame = cv2.flip(frame, 1)

            # Make detections (using our function)
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # --- Calculate and Display FPS ---
            new_frame_time = time.time()
            # Avoid division by zero on the first frame
            if (new_frame_time - prev_frame_time) > 0:
                fps = 1 / (new_frame_time - prev_frame_time)
            else:
                fps = 0
            prev_frame_time = new_frame_time
            
            # Display FPS on the image
            cv2.putText(
                image, 
                f"FPS: {int(fps)}", 
                (10, 30), # Position
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, # Font scale
                (0, 255, 0), # Color (Green)
                2, # Thickness
                cv2.LINE_AA
            )
            # --- End of FPS ---

            # Show to screen
            cv2.imshow('FYP Demo - MediaPipe Holistic', image)

            # Break loop when 'p' is pressed
            if cv2.waitKey(5) & 0xFF == ord('p'):
                break
                
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Demo stopped.")

if __name__ == "__main__":
    main()