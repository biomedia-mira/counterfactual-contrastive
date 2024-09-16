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


class GaussianBlur(tf.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 0.7
    ):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = tf.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class NoEmptyCrop:
    def __init__(self, base_tsfm):
        self.base_tsfm = base_tsfm

    def __call__(self, x):
        xt = self.base_tsfm(x)
        if x.min() == x.max():
            print("skipping, empty image")
            return xt
        while xt.min() == xt.max():
            xt = self.base_tsfm(x)
        return xt


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        base_tsfm=None,
        use_counterfactuals: bool = False,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.base_tsfm = base_tsfm

        # random resized crop and flip
        self.geometric_augmentation_global = tf.Compose(
            [
                tf.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=tf.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                tf.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = tf.Compose(
            [
                tf.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=tf.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                tf.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = NoEmptyCrop(
            self.geometric_augmentation_local
        )

        # color distorsions / blurring
        color_jittering = tf.Compose(
            [
                tf.RandomApply(
                    [
                        tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)
                    ],  # no hue for B&W
                    p=0.8,
                ),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = tf.Compose(
            [
                GaussianBlur(p=0.1),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = tf.Compose(
            [
                tf.ToTensor(),
                # make_normalize_transform(), not needed for my datasets
            ]
        )

        self.global_transfo1 = tf.Compose([color_jittering, global_transfo1_extra])
        self.global_transfo2 = tf.Compose([color_jittering, global_transfo2_extra])
        self.local_transfo = tf.Compose([color_jittering, local_transfo_extra])
        self.use_counterfactuals = use_counterfactuals

    def __call__(self, image):
        output = {}

        # global crops:
        if self.use_counterfactuals:
            im1_base = self.geometric_augmentation_global(image[0])
            im2_base = self.geometric_augmentation_global(image[1])
        else:
            im1_base = self.geometric_augmentation_global(image)
            im2_base = self.geometric_augmentation_global(image)

        # To make sure that both global transformation are applied
        # with the same proba to both real and counterfactual images.
        if torch.rand(1).item() > 0.5:
            global_crop_1 = self.global_transfo1(im1_base)
            global_crop_2 = self.global_transfo2(im2_base)
        else:
            global_crop_1 = self.global_transfo2(im1_base)
            global_crop_2 = self.global_transfo1(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        if self.use_counterfactuals:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image[0]))
                for _ in range(self.local_crops_number // 2)
            ] + [
                self.local_transfo(self.geometric_augmentation_local(image[1]))
                for _ in range(self.local_crops_number // 2)
            ]
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image))
                for _ in range(self.local_crops_number)
            ]
        # here we could have half from counterfactual image and half from the original one?
        # typically this is like 8 in the dino config
        output["local_crops"] = local_crops
        output["offsets"] = ()

        if self.base_tsfm is not None:
            output["x"] = self.base_tsfm(image)
        else:
            output["x"] = image

        return output
