import os
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.datasets.video_utils import VideoClips
from typing import Optional, Dict, List, Callable, Tuple, Union



def check_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    return filename.lower().endswith(extensions)


def cast(typ: Callable, x: Optional[Callable[[str], bool]]) -> Optional[Callable[[str], bool]]:
    return typ(x) if x is not None else None


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    if extensions is not None:
        is_valid_file = cast(Callable[[str], bool], is_valid_file) or \
                        (lambda x: check_extension(x, extensions))

    instances = []
    available_classes = set()
    
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)

        if not os.path.isdir(target_dir):
            continue

        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset."""
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class VideoClassificationDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        frames_per_clip: int,
        split: str = "train",
        frame_rate: Optional[int] = None,
        step_between_clips: int = 1,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ("avi", "mp4"),
        num_workers: int = 1,
        output_format: str = "TCHW",
    ):
        super().__init__(root, transform=transform)

        self.extensions = extensions
        self.split_folder = os.path.join(root, split)
        self.split = split

        self.classes, class_to_idx = find_classes(self.split_folder)
        self.samples = make_dataset(self.split_folder, class_to_idx, extensions)

        video_list = [x[0] for x in self.samples]

        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            num_workers=num_workers,
            output_format=output_format,
        )

        print(class_to_idx)

    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        video, _, _, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label


if __name__ == '__main__':
    dataset = VideoClassificationDataset(root='../input', frames_per_clip=12, split='train')
