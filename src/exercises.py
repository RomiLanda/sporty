import numpy as np
from src.angle_body_part import BodyPartAngle, BodyPartDistance
from src.utils import *


class Exercises(BodyPartAngle, BodyPartDistance):
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

    
    def squat(self, counter, status, pose):

        right_leg_angle = self.angle_of_the_leg("RIGHT")
        left_leg_angle = self.angle_of_the_leg("LEFT")
        avg_leg_angle = (right_leg_angle + left_leg_angle) / 2

  
        if status:
            if avg_leg_angle < 120:
                status = False
                counter +=1
                
                if self.is_knee_inward():
                    pose = "Knee inward"
                else:
                    pose = "Correct"
                
    
        else:
            if avg_leg_angle > 160:
                status = True
                pose = ""

        return counter, status, pose

    
    def calculate_exercise(self, exercise_type, counter, status, pose):
        method_name = getattr(self, exercise_type.lower())
        return method_name(counter, status, pose)
