import math
import pandas as pd
import mediapipe as mp

mp_holistic = mp.solutions.holistic

def calculate_angle(landmark1, landmark2, landmark3):
    '''
    This function calculates the angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x, y, and z coordinates.
        landmark2: The second landmark containing the x, y, and z coordinates.
        landmark3: The third landmark containing the x, y, and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''

    # Get the required landmarks' coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:
        angle += 360

    return angle


def detection_body_part(landmarks, body_part_name):
    index = mp_holistic.PoseLandmark[body_part_name].value
    return [landmarks[index].x, landmarks[index].y, landmarks[index].visibility]


def detection_body_parts(landmarks):
    body_part_names = [str(lm).split(".")[1] for lm in mp_holistic.PoseLandmark]
    data = [[name, *detection_body_part(landmarks, name)] for name in body_part_names]
    return pd.DataFrame(data, columns=["body_part", "x", "y", "visibility"])

