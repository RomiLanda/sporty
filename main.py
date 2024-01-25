import cv2
import mediapipe as mp
from src.exercises import Exercises
from src.utils import *
from src.angle_body_part import BodyPartDistance

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

EXERCISE = "squat"
VIDEO = "input/squat.mp4"

def video_analyzer():
    cap = cv2.VideoCapture(VIDEO)

    # Initialize holistic model 
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    counter = 0  # movement of exercise
    status = True  # state of move
    pose = ""
        
    while cap.isOpened():
        status_video, frame = cap.read()

        if not status_video:
            break
        
        # Transform frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = holistic.process(rgb_frame)
        
        landmarks = results.pose_landmarks.landmark


        counter, status, pose = Exercises(landmarks=landmarks).calculate_exercise(EXERCISE, counter, status, pose)
        
        print(counter, status, pose)


        # Pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()


if __name__== '__main__':
    video_analyzer()