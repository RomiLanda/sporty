import os 
import glob
import shutil
import numpy as np
import random

from tqdm.auto import tqdm


seed = 27
np.random.seed(seed)
random.seed(seed)

VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.15

INPUT_ROOT_DIR = os.path.join("input", "workout_classifier_resized")
OUTPUT_DIR = os.path.join("input", "workout_classifier_split")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def copy_data(video_list, split='train'):
    for video_path in video_list:
        exercise_name, video_name = video_path.split(os.path.sep)[-2:]
        data_dir = os.path.join(OUTPUT_DIR, split, exercise_name)

        os.makedirs(os.path.join(data_dir), exist_ok=True)
        
        shutil.copy(video_path, os.path.join(data_dir, video_name))


def split_data(all_videos, split_percenage):
    random.shuffle(all_videos)
    total_samples = len(all_videos)

    split_p = all_videos[0:int(total_samples * split_percenage)] # porcentage
    split_r = all_videos[int(total_samples * split_percenage):] # 1-percentage

    return split_r, split_p


def split_train_test_val(list_exersices):
    for exercise_name in tqdm(list_exersices, desc= "Split data"):
        all_videos = glob.glob(os.path.join(INPUT_ROOT_DIR, exercise_name, '*'))
        
        videos_ , test_videos = split_data(all_videos, split_percenage = TEST_SPLIT)
        train_videos , val_videos = split_data(videos_ , split_percenage = VALIDATION_SPLIT)

        for split, videos in [('data_train', train_videos), ('data_test', test_videos), ('data_val', val_videos)]:
            copy_data(videos, split)


if __name__== '__main__':
    all_exersices = os.listdir(INPUT_ROOT_DIR)
    split_train_test_val(all_exersices)
