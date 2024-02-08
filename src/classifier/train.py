import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from tqdm.auto import tqdm

import src.classifier.preprocess_data as preprocess_data
from src.classifier.model import build_model
from src.classifier.load_data import VideoClassificationDataset
from src.classifier.utills_classifier import save_model, save_plots, SaveBestModel
from src.classifier.class_names import class_names

from torchvision.datasets.samplers import (
    RandomClipSampler, UniformClipSampler
)
from torch.utils.data.dataloader import default_collate

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


EPOCHS = 50 # Number of epochs to train our network for
LEARNING_RATE = 0.001 #Learning rate for training the model
BATCH_SIZE = 8 
FINE_TUNE = True
SCHEDULER = True
SAVE_NAME = "model" # file name of the final model to save
WORKERS = 4 # number of parallel workers for data loader
CLIP_LEN = 16 # number of frames per clip
CLIPS_PER_VIDEO = 5 # maximum number of clips per video
FRAME_RATE = 15  # the frame rate of each clip
IMGSZ = (256, 256) # image resize resolution
CROP_SIZE = (224, 224) # image cropping resolution



def collate_fn(batch):
    batch = [(d[0], d[1]) for d in batch]
    return default_collate(batch)

# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    bs_accumuator = 0
    counter = 0
    prog_bar = tqdm(
        trainloader, 
        total=len(trainloader), 
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    for i, data in enumerate(prog_bar):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        bs_accumuator += outputs.shape[0]
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / bs_accumuator)
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    bs_accumuator = 0
    counter = 0
    prog_bar = tqdm(
        testloader, 
        total=len(testloader), 
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    with torch.no_grad():
        for i, data in enumerate(prog_bar):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            bs_accumuator += outputs.shape[0]
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / bs_accumuator)
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    out_dir = os.path.join('output')
    os.makedirs(out_dir, exist_ok=True)

    ## Data Loading.
    train_crop_size = tuple(CROP_SIZE)
    train_resize_size = tuple(IMGSZ)

    transform_train = preprocess_data.VideoClassificationPreprocess(
        crop_size=train_crop_size, 
        resize_size=train_resize_size
    )
    transform_valid = preprocess_data.VideoClassificationPreprocess(
        crop_size=train_crop_size, 
        resize_size=train_resize_size, 
        hflip_prob=0.0
    )

    # Load the training and validation datasets.
    dataset_train = VideoClassificationDataset(
        'input/workout_classifier_split',
        frames_per_clip= CLIP_LEN,
        frame_rate=FRAME_RATE,
        split="data_train",
        transform=transform_train,
        extensions=(
            "mp4",
            'avi',
            'mov'
        ),
        output_format="TCHW",
        num_workers=WORKERS
    )
    dataset_valid = VideoClassificationDataset(
        'input/workout_classifier_split',
        frames_per_clip= CLIP_LEN, 
        frame_rate= FRAME_RATE,
        split="data_val",
        transform=transform_valid,
        extensions=(
            "mp4",
            'avi',
            'mov'
        ),
        output_format="TCHW",
        num_workers= WORKERS
    )
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Classes: {class_names}")


    # Load the training and validation data loaders.
    train_sampler = RandomClipSampler(
        dataset_train.video_clips, max_clips_per_video=CLIPS_PER_VIDEO
    )
    test_sampler = UniformClipSampler(
        dataset_valid.video_clips, num_clips_per_video=CLIPS_PER_VIDEO
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        num_workers=WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Learning_parameters. 
    lr = LEARNING_RATE
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {EPOCHS}\n")

    # Load the model.
    model = build_model(
        fine_tune=FINE_TUNE, 
        num_classes=len(class_names)
    ).to(device)
    print(model)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()

    # LR scheduler.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25], gamma=0.1, verbose=True
    )

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    # Start the training.
    for epoch in range(EPOCHS):
        print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, SAVE_NAME
        )
        if SCHEDULER:
            scheduler.step()
        print('-'*50)

    # Save the trained model weights.
    save_model(EPOCHS, model, optimizer, criterion, out_dir, SAVE_NAME)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir)
    print('TRAINING COMPLETE')