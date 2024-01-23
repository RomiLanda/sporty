import numpy as np
from src.angle_body_part import BodyPartAngle
from src.utils import *


class Exercises(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def pull_up(self, counter, status):

        left_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        right_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        
        left_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        right_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")

        avg_elbow = (left_elbow[1] + right_elbow[1]) / 2
        avg_shoulder= (left_shoulder[1] + right_shoulder[1]) / 2

        if status:
            if avg_shoulder > avg_elbow:
                status = False
                counter += 1
                
        else:
            if avg_shoulder < avg_elbow:
                status = True

        return counter, status

    def calculate_exercise(self, exercise_type, counter, status):
        method_name = getattr(self, exercise_type.lower())
        return method_name(counter, status)
