import os
import cv2
import glob

from tqdm.auto import tqdm

INPUT_ROOT_DIR = os.path.join("input", "workout_classifier")
OUTPUT_DIR = os.path.join("input", "workout_classifier_resized")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESIZE_TO = 512


def resize(image, img_size=512):
    h, w = image.shape[:2]
    ratio = img_size / max(h, w)
    if max(h, w) > img_size:
        ratio = img_size / max(h, w)
        new_size = (int(w * ratio), int(h * ratio))
        image = cv2.resize(image, new_size)
    return image


def process_and_resize_video(video_path, output_dir, img_size=512):
    cap = cv2.VideoCapture(video_path)
    exercise_name, video_name = video_path.split(os.path.sep)[-2:]

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    status_video, frame = cap.read()
    frame = resize(frame, img_size=RESIZE_TO)
    new_h, new_w = frame.shape[:2]

    output_path = os.path.join(OUTPUT_DIR, exercise_name, video_name)
    output = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps,
                             (new_w, new_h)
                             )

    os.makedirs(os.path.join(OUTPUT_DIR, exercise_name), exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = resize(frame, img_size=RESIZE_TO)
            output.write(frame)
        else:
            break

    output.release()
    cap.release()


def process_all_videos(input_dir, output_dir, img_size=512):
    all_videos = glob.glob(os.path.join(input_dir, '*', '*'), recursive=True)

    for video_path in tqdm(all_videos, desc="Processing videos"):
        process_and_resize_video(video_path, output_dir, img_size=img_size)


if __name__== '__main__':
    process_all_videos(INPUT_ROOT_DIR, OUTPUT_DIR, img_size=RESIZE_TO)  
