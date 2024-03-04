from typing import Any, Callable, Tuple

import torch
import torchvision.transforms as tf
from omegaconf import DictConfig


class Return2Views:
    def __init__(self, transform_pipeline) -> None:
        self.tsfm = transform_pipeline

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.tsfm(x)
        x2 = self.tsfm(x)
        return torch.stack([x1, x2], dim=0)


class RandomSharpness:
    def __init__(self, sharp):
        self.sharp = sharp

    def __call__(self, x: torch.Tensor) -> Any:
        sharp_min = 1 - self.sharp
        sharp_max = 1 + self.sharp
        random_sharp = sharp_min + (sharp_max - sharp_min) * torch.rand(1)
        return tf.RandomAdjustSharpness(random_sharp)(x)


def get_augmentations_from_config(config: DictConfig) -> Tuple[Callable, Callable]:
    """
    Return transformation pipeline as per config.
    """

    transform_list, val_transforms = [], []
    if config.augmentations.random_crop != "None":
        transform_list.append(
            tf.RandomResizedCrop(
                config.augmentations.resize,
                scale=config.augmentations.random_crop,
                antialias=True,
            )
        )
        val_transforms.append(tf.Resize(config.augmentations.resize, antialias=True))

    if config.augmentations.resize != "None":
        transform_list.append(tf.Resize(config.augmentations.resize, antialias=True))
        val_transforms.append(tf.Resize(config.augmentations.resize, antialias=True))

    if config.augmentations.random_rotation != "None":
        transform_list.append(tf.RandomRotation(config.augmentations.random_rotation))
    if config.augmentations.horizontal_flip:
        transform_list.append(tf.RandomHorizontalFlip())
    if config.augmentations.vertical_flip:
        transform_list.append(tf.RandomVerticalFlip())
    if config.augmentations.random_color_jitter:
        transform_list.append(
            tf.ColorJitter(
                brightness=config.augmentations.random_color_jitter,
                contrast=config.augmentations.random_color_jitter,
                hue=0
                if config.input_channels == 1
                else config.augmentations.random_color_jitter,
                saturation=0
                if config.input_channels == 1
                else config.augmentations.random_color_jitter,
            )
        )

    if config.augmentations.random_erase_scale[0] > 0.0:
        transform_list.append(
            tf.RandomErasing(
                scale=[
                    config.augmentations.random_erase_scale[0],
                    config.augmentations.random_erase_scale[1],
                ]
            )
        )

    if config.augmentations.sharp > 0.0:
        transform_list.append(RandomSharpness(config.augmentations.sharp))

    if config.augmentations.center_crop != "None":
        transform_list.append(tf.CenterCrop(config.augmentations.center_crop))
        val_transforms.append(tf.CenterCrop(config.augmentations.center_crop))

    return tf.Compose(transform_list), tf.Compose(val_transforms)
