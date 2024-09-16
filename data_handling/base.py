from typing import Tuple
import copy
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data_handling.augmentations import (
    Return2Views,
    get_augmentations_from_config,
)

from classification.dinov2.utils import MaskingGenerator

"""A batch is a 3-tuple of imgs, labels, and metadata dict"""
BatchType = Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]


class BaseDataModuleClass(LightningDataModule):
    def __init__(self, config: DictConfig, shuffle: bool = True, parents=None) -> None:
        super().__init__()
        self.config = config
        self.shuffle = shuffle
        self.parents = parents
        self.train_tsfm, self.val_tsfm = get_augmentations_from_config(config.data)
        self.image_size = (
            self.config.data.augmentations.center_crop
            if self.config.data.augmentations.center_crop != "None"
            else self.config.data.augmentations.resize
        )

        if config.trainer.loss != "dino":
            if config.trainer.return_two_views and not config.data.use_counterfactuals:
                self.train_tsfm = Return2Views(self.train_tsfm)
                self.val_tsfm = Return2Views(self.val_tsfm)
            elif config.data.use_counterfactuals:
                self.val_tsfm = Return2Views(self.val_tsfm)
        else:
            from data_handling.augmentations import DataAugmentationDINO

            self.test_tsfm = copy.deepcopy(self.val_tsfm)
            self.train_tsfm = DataAugmentationDINO(
                base_tsfm=self.train_tsfm,
                global_crops_size=self.image_size,
                local_crops_number=config.trainer.crops.local_crops_number,
                local_crops_scale=config.trainer.crops.local_crops_scale,
                global_crops_scale=config.trainer.crops.global_crops_scale,
                local_crops_size=config.trainer.crops.local_crops_size,
                use_counterfactuals=config.data.use_counterfactuals,
            )

            self.val_tsfm = DataAugmentationDINO(
                base_tsfm=self.val_tsfm,
                global_crops_size=self.image_size,
                local_crops_number=config.trainer.crops.local_crops_number,
                local_crops_scale=config.trainer.crops.local_crops_scale,
                global_crops_scale=config.trainer.crops.global_crops_scale,
                local_crops_size=config.trainer.crops.local_crops_size,
            )

        if not config.trainer.use_train_augmentations:
            self.train_tsfm = self.val_tsfm
        self.sampler = None
        self.create_datasets()

    def get_collate_fn(self):
        collate_fn = None

        if self.config.trainer.loss == "dino":
            from classification.dinov2.collate import DinoCollator

            n_tokens = (self.image_size[0] // self.config.trainer.patch_size) * (
                self.image_size[1] // self.config.trainer.patch_size
            )
            mask_generator = MaskingGenerator(
                input_size=(
                    self.image_size[0] // self.config.trainer.patch_size,
                    self.image_size[1] // self.config.trainer.patch_size,
                ),
                max_num_patches=0.5 * n_tokens,
            )

            collate_fn = DinoCollator(
                mask_ratio_tuple=self.config.trainer.ibot_mask_ratio_min_max,
                mask_probability=self.config.trainer.mask_probability,
                n_tokens=n_tokens,
                mask_generator=mask_generator,
            )
        return collate_fn

    def train_dataloader(self):
        if self.sampler is not None and self.shuffle:
            return DataLoader(
                self.dataset_train,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory,
                persistent_workers=False,
                batch_sampler=self.sampler,
                collate_fn=self.get_collate_fn(),
            )
        return DataLoader(
            self.dataset_train,
            self.config.data.batch_size,
            shuffle=self.shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            collate_fn=self.get_collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            collate_fn=self.get_collate_fn(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )

    @property
    def dataset_name(self):
        raise NotImplementedError

    def create_datasets(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError
