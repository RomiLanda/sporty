import cv2
import mediapipe as mp
from src.utils import *


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def video():
    cap = cv2.VideoCapture("input/pull-up.mp4")

    # Initialize holistic model 
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break
        
        # Transform frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = holistic.process(rgb_frame)
        
        landmarks = results.pose_landmarks.landmark
        print(detection_body_parts(landmarks))


        # Pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__== '__main__':
    video()