import os
import torch
import cv2
import albumentations as A
import numpy as np

from src.classifier.class_names import class_names
from src.classifier.model import build_model

# from class_names import class_names
# from model import build_model

def load_model_labels():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load('output/best_model.pth')
    model = build_model(fine_tune=False, num_classes=len(class_names))
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.eval().to(device)
    return model, device


def process_frame(frame, transform):
    image = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(image=frame)['image']
    return frame


def transform_image(crop_size, resize_size):
    crop_size = tuple(crop_size)
    resize_size = tuple(resize_size)
    transform = A.Compose([
        A.Resize(resize_size[1], resize_size[0], always_apply=True),
        A.CenterCrop(crop_size[1], crop_size[0], always_apply=True),
        A.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989],
            always_apply=True
        )
    ])
    return transform


def predict_label(frames, model, device):
    with torch.no_grad():
        input_frames = np.array(frames)
        input_frames = torch.tensor(np.transpose(np.expand_dims(input_frames, axis=0), (0, 4, 1, 2, 3)), dtype=torch.float32)
        input_frames = input_frames.to(device)
        outputs = model(input_frames)
        _, preds = torch.max(outputs.data, 1)
        label = class_names[preds].strip()
    return label


 
def predict_exercise(video_input, crop_size, imgsz, clip_len) -> list:
    transform = transform_image(crop_size, imgsz)
    # Load model and labels
    model, device = load_model_labels()

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
        return None  
    
    frames = []
    predicted_labels = []  

    while cap.isOpened():
        status_video, frame = cap.read()

        if not status_video:
            break

        if status_video:
            frame = process_frame(frame, transform)
            frames.append(frame)

            if len(frames) == CLIP_LEN:
                label = predict_label(frames, model, device)
                predicted_labels.append(label)  # Store the predicted label
                frames = []  # Reset frames for the next batch of frames

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

    return predicted_labels 



if __name__ == "__main__":
    INPUT_VIDEO = "input/workout_classifier_split/data_test/squat/squat_11.mp4"
    CLIP_LEN = 16
    IMGSZ = (256, 256)
    CROP_SIZE = (224, 224)
    
    print(predict_exercise(INPUT_VIDEO, CROP_SIZE, IMGSZ, CLIP_LEN)[0])
