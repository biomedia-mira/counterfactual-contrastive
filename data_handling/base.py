from typing import Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data_handling.augmentations import Return2Views, get_augmentations_from_config

"""A batch is a 3-tuple of imgs, labels, and metadata dict"""
BatchType = Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]


class BaseDataModuleClass(LightningDataModule):
    def __init__(self, config: DictConfig, shuffle: bool = True, parents=None) -> None:
        super().__init__()
        self.config = config
        self.shuffle = shuffle
        self.parents = parents
        self.train_tsfm, self.val_tsfm = get_augmentations_from_config(config.data)
        if config.trainer.return_two_views and not config.data.use_counterfactuals:
            self.train_tsfm = Return2Views(self.train_tsfm)
            self.val_tsfm = Return2Views(self.val_tsfm)
        elif config.data.use_counterfactuals:
            self.val_tsfm = Return2Views(self.val_tsfm)

        if not config.trainer.use_train_augmentations:
            self.train_tsfm = self.val_tsfm
        self.sampler = None
        self.create_datasets()

    def train_dataloader(self):
        if self.sampler is not None and self.shuffle:
            return DataLoader(
                self.dataset_train,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory,
                persistent_workers=False,
                batch_sampler=self.sampler,
            )
        return DataLoader(
            self.dataset_train,
            self.config.data.batch_size,
            shuffle=self.shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
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
