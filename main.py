import cv2
import mediapipe as mp
from src.sporty.exercises import Exercises
from src.sporty.utils import *
from src.sporty.geometry_body_part import BodyPartDistance
from src.classifier import inference


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


INPUT_VIDEO = "input/workout_classifier_split/data_test/squat/squat_11.mp4" # path video or 0 to camara
CLIP_LEN = 16
IMGSZ = (256, 256)
CROP_SIZE = (224, 224)
EXERCISE = inference.predict_exercise(INPUT_VIDEO, CROP_SIZE, IMGSZ, CLIP_LEN)[0]

# Set the desired width and height for the resized video
FRAME_WIDTH = 1200
FRAME_HEIGHT = 800


def initialize_holistic_model():
    return mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def video_analyzer(video, exercise):
    cap = cv2.VideoCapture(video)

    # Set the new frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Initialize holistic model 
    holistic = initialize_holistic_model()
    counter, status, pose = 0, True, ""
       

    while cap.isOpened():
        status_video, frame = cap.read()

        if not status_video:
            break
        
        # Transform frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = holistic.process(rgb_frame)

        try:
            landmarks = results.pose_landmarks.landmark
            counter, status, pose = Exercises(landmarks=landmarks).calculate_exercise(exercise, counter, status, pose)
            print(counter, status, pose)
        
        except:
            pass

        # Pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.putText(
            frame, 
            text= f'Exercise: {exercise} - Repetition Number: {counter} - Observation: {pose}', 
            org = (15, 25),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = 1, 
            color = (255, 150, 0), 
            thickness = 2, 
            lineType=cv2.LINE_AA
        )

    
        cv2.imshow("frame", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__== '__main__': 
    print(EXERCISE)

    video_analyzer(INPUT_VIDEO, EXERCISE)