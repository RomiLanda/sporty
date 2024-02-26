import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
from src.sporty.utils import *


class BodyPartAngle:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def _calculate_avg_point(self, part1, part2):
        return [(part1[0] + part2[0]) / 2, (part1[1] + part2[1]) / 2]

    def _calculate_angle_from_parts(self, part1, part2, part3):
        return calculate_angle(part1, part2, part3)

    def _get_body_parts(self, *body_part_names):
        return [detection_body_part(self.landmarks, name) for name in body_part_names]

    def angle_of_the_arm(self, side):
        shoulder = detection_body_part(self.landmarks, f"{side}_SHOULDER")
        elbow = detection_body_part(self.landmarks, f"{side}_ELBOW")
        wrist = detection_body_part(self.landmarks, f"{side}_WRIST")
        return self._calculate_angle_from_parts(shoulder, elbow, wrist)

    def angle_of_the_leg(self, side):
        hip = detection_body_part(self.landmarks, f"{side}_HIP")
        knee = detection_body_part(self.landmarks, f"{side}_KNEE")
        ankle = detection_body_part(self.landmarks, f"{side}_ANKLE")
        return self._calculate_angle_from_parts(hip, knee, ankle)

    def angle_of_the_neck(self):
        r_shoulder, l_shoulder, r_mouth, l_mouth, r_hip, l_hip = self._get_body_parts(
            "RIGHT_SHOULDER", "LEFT_SHOULDER", "MOUTH_RIGHT", "MOUTH_LEFT", "RIGHT_HIP", "LEFT_HIP"
        )
        shoulder_avg = self._calculate_avg_point(r_shoulder, l_shoulder)
        mouth_avg = self._calculate_avg_point(r_mouth, l_mouth)
        hip_avg = self._calculate_avg_point(r_hip, l_hip)
        return abs(180 - self._calculate_angle_from_parts(mouth_avg, shoulder_avg, hip_avg))

    def angle_of_the_abdomen(self):
        shoulder_avg, hip_avg, knee_avg = self._get_body_parts(
            "RIGHT_SHOULDER", "LEFT_SHOULDER", "RIGHT_HIP", "LEFT_HIP", "RIGHT_KNEE", "LEFT_KNEE"
        )
        return self._calculate_angle_from_parts(shoulder_avg, hip_avg, knee_avg)


class BodyPartDistance:
    def __init__(self, landmarks):
        self.landmarks = landmarks
    
    def _calculate_x_distance(self, part1, part2):
        return calculate_distances(part1, part2)[0]

    def _calculate_y_distance(self, part1, part2):
        return calculate_distances(part1, part2)[1]
    
    def _calculate_z_distance(self, part1, part2):
        return calculate_distances(part1, part2)[2]

    def _get_body_parts(self, *body_part_names):
        return [detection_body_part(self.landmarks, name) for name in body_part_names]
    
    def is_knee_inward(self):
        r_knee, l_knee, r_ankle, l_ankle = self._get_body_parts(
            "RIGHT_KNEE", "LEFT_KNEE", "RIGHT_ANKLE", "LEFT_ANKLE") 
        
        knee_distance_x = self._calculate_x_distance(r_knee, l_knee)
        ankle_distance_x = self._calculate_x_distance(r_ankle, l_ankle)

        return knee_distance_x < ankle_distance_x

