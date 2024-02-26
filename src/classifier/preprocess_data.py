import torch
import torch.nn as nn
from torchvision.transforms import transforms



class ConvertBCHWtoCBHW(nn.Module):
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class VideoClassificationPreprocess:
    def __init__(
        self,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        hflip_prob=0.5,
    ):
        self.transforms = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size, antialias=False),
            transforms.RandomHorizontalFlip(hflip_prob) if hflip_prob > 0 else transforms.Lambda(lambda x: x),
            transforms.Normalize(mean=mean, std=std), 
            transforms.CenterCrop(crop_size), 
            ConvertBCHWtoCBHW()
        ])

    def __call__(self, x):
        return self.transforms(x)
