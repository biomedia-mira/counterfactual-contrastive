import io
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

from classification.losses import nt_xent_loss
from data_handling.base import BatchType


class ClassificationModule(pl.LightningModule):
    """
    A generic PL module for classification
    """

    def __init__(
        self,
        num_classes: int,
        encoder_name: str,
        pretrained: bool = False,
        lr: float = 1e-4,
        input_channels: int = 3,
        patience_for_scheduler: int = 10,
        metric_to_monitor: str = "Val/AUROC",
        metric_to_monitor_mode: str = "max",
        weight_decay: float = 0.0,
        loss: str = "ce",
        contrastive_temperature: float = 0.1,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience_scheduler = patience_for_scheduler
        self.metric_to_monitor = metric_to_monitor
        self.metric_to_monitor_mode = metric_to_monitor_mode
        self.loss = loss
        self.input_channels = input_channels
        self.model = self.get_model()

        match loss:
            case "ce":
                self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
            case "simclr":
                self.contrastive_temperature = contrastive_temperature
                self.head_criterion = torch.nn.CrossEntropyLoss(weight=weights)
                self.model.projection_head = torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=self.model.num_features, out_features=1024
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_features=1024, out_features=128),
                )
            case _:
                raise NotImplementedError

        self.save_hyperparameters()
        self.automatic_optimization = loss not in ["simclr"]

    def common_step(self, batch: BatchType, prefix: str, batch_idx: int) -> Any:  # type: ignore
        data, target = batch["x"], batch["y"].float()
        if data.ndim == 5:
            bsz, n_views, c, w, h = data.shape
            data = data.reshape((-1, c, w, h))
        else:
            if self.loss != "ce" and self.training:
                raise ValueError(
                    """
                    Your dataloader returns 1 view per image but you use SimCLR loss.
                    Not expected.
                    """
                )
            bsz, n_views = data.shape[0], 1

        if self.loss == "simclr":
            features = self.model.get_features(data)
            proj_features = self.model.projection_head(features)
            proj_features = normalize(proj_features, dim=1)
            proj_features = proj_features.reshape((bsz, n_views, -1))
            loss = 0

            if self.training or n_views > 1:
                loss += nt_xent_loss(proj_features[:, 0], proj_features[:, 1])
                self.log(f"{prefix}/simclr_loss", loss, sync_dist=True)

            output = self.model.classify_features(features.detach())

            if n_views > 1:
                target = torch.stack([target]*n_views, dim=1).reshape((bsz*n_views, -1))

            if self.training:
                # Optimizer for model encoder
                encoder_opt, head_opt = self.optimizers()
                encoder_opt.zero_grad()
                self.manual_backward(loss)
                encoder_opt.step()

            head_loss = self.head_criterion(output, target)

            if self.training:
                head_opt.zero_grad()
                self.manual_backward(head_loss)
                head_opt.step()

            self.log(f"{prefix}/head_loss", head_loss, sync_dist=True)

        else:
            output = self.model(data)
            if n_views > 1:
                target = torch.stack([target]*n_views, dim=1).reshape((bsz*n_views, -1))
            loss = self.criterion(output, target)

        self.log(f"{prefix}/loss", loss, sync_dist=True)
        if torch.isnan(loss):
            raise ValueError("Found loss Nan")

        probas = torch.softmax(output, 1)

        return loss, probas, target

    def training_step(self, batch: Any, batch_idx: int) -> Any:  # type: ignore
        loss, probas, targets = self.common_step(
            batch, prefix="Train", batch_idx=batch_idx
        )
        self.train_probas.append(probas.detach().cpu())
        self.train_targets.append(targets.detach().cpu())

        if batch_idx == 0:
            data = batch["x"]
            data = data.cpu().numpy()
            self._plot_image_and_log_img_grid(
                data, targets.cpu().numpy(), "train/inputs"
            )
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:  # type: ignore
        _, probas, targets = self.common_step(batch, prefix="Val", batch_idx=batch_idx)
        self.val_probas.append(probas.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

    def test_step(self, batch, batch_idx: int) -> None:  # type: ignore
        _, probas, targets = self.common_step(batch, prefix="Test", batch_idx=batch_idx)
        self.test_probas.append(probas.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

    def on_train_epoch_start(self) -> None:
        self.train_probas = []
        self.train_targets = []

    def _compute_metrics_at_epoch_end(
        self, targets: torch.Tensor, probas: torch.Tensor, prefix: str
    ):
        preds = torch.argmax(probas, 1)
        try:
            if self.num_classes == 2:
                self.log(
                    f"{prefix}/AUROC",
                    roc_auc_score(targets, probas[:, 1]),
                    sync_dist=True,
                )
            else:
                self.log(
                    f"{prefix}/AUROC",
                    roc_auc_score(targets, probas, average="macro", multi_class="ovr"),
                    sync_dist=True,
                )
        except ValueError:
            pass
        amax_targets = torch.argmax(targets, 1)
        self.log(f"{prefix}/Accuracy", accuracy_score(amax_targets, preds), sync_dist=True)
        self.log(
            f"{prefix}/BalAccuracy",
            balanced_accuracy_score(amax_targets, preds),
            sync_dist=True,
        )

    def on_train_epoch_end(self, unused=None) -> None:
        if len(self.train_targets) > 0:
            targets, probas = torch.cat(self.train_targets), torch.cat(
                self.train_probas
            )
            self._compute_metrics_at_epoch_end(targets, probas, "Train")
        self.train_probas = []
        self.train_targets = []

    def on_validation_epoch_start(self) -> None:
        self.val_probas = []
        self.val_targets = []

    def on_validation_epoch_end(self, unused=None) -> None:
        targets, probas = torch.cat(self.val_targets), torch.cat(self.val_probas)
        self._compute_metrics_at_epoch_end(targets, probas, "Val")
        self.val_probas = []
        self.val_targets = []

    def on_test_epoch_start(self) -> None:
        self.test_probas = []
        self.test_targets = []

    def on_test_epoch_end(self, unused=None) -> None:
        targets, probas = torch.cat(self.test_targets), torch.cat(self.test_probas)
        self._compute_metrics_at_epoch_end(targets, probas, "Test")
        self.test_probas = []
        self.test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.loss not in ["simclr"]:
            scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.patience_scheduler,
                    mode=self.metric_to_monitor_mode,
                    min_lr=1e-5,
                ),
                "monitor": self.metric_to_monitor,
            }
            return [optimizer], [scheduler]
        else:
            head_optimizer = torch.optim.Adam(
                self.model.net.fc.parameters(), lr=1e-4, weight_decay=1e-4
            )
            return optimizer, head_optimizer

    def get_model(self) -> torch.nn.Module:
        if self.encoder_name.startswith("resnet"):
            return ResNetBase(
                num_classes=self.num_classes,
                encoder_name=self.encoder_name,
                pretrained=self.pretrained,
                input_channels=self.input_channels,
            )
        else:
            raise NotImplementedError

    def _plot_image_and_log_img_grid(self, data: np.ndarray, y: np.ndarray, tag: str):
        f, ax = plt.subplots(2, 5, figsize=(15, 5))

        if data.ndim == 5:
            for i in range(min(5, data.shape[0])):
                for j in range(2):
                    img = np.transpose(data[i, j], [1, 2, 0])
                    img = (img - img.min()) / (img.max() - img.min())
                    ax[j, i].imshow(img) if img.shape[-1] == 3 else ax[j, i].imshow(
                        img, cmap="gray"
                    )
                    ax[j, i].set_title(y[i])
                    ax[j, i].axis("off")
        else:
            ax = ax.ravel()
            for i in range(min(10, data.shape[0])):
                img = np.transpose(data[i], [1, 2, 0])
                img = (img - img.min()) / (img.max() - img.min())
                ax[i].imshow(img) if img.shape[-1] == 3 else ax[i].imshow(
                    img, cmap="gray"
                )
                ax[i].set_title(y[i])
                ax[i].axis("off")

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")

        im = Image.open(img_buf)
        self.logger.experiment.log({tag: wandb.Image(im)})
        img_buf.close()
        plt.close()


class ResNetBase(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder_name: str,
        pretrained: bool,
        input_channels: int = 3,
        normalise_features: bool = False,
    ) -> None:
        super().__init__()
        match encoder_name:
            case "resnet50":
                if pretrained:
                    self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                else:
                    self.net = models.resnet50(weights=None)
            case "resnet18":
                if pretrained:
                    self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                else:
                    self.net = models.resnet18(weights=None)
            case _:
                raise ValueError(f"Encoder name {encoder_name} not recognised.")
        self.num_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(self.num_features, num_classes)
        self.num_classes = None
        self.normalise_features = normalise_features
        if input_channels != 3:
            self.net.conv1 = torch.nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x0 = self.net.maxpool(x)
        x1 = self.net.layer1(x0)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        x4 = self.net.avgpool(x4)
        x4 = torch.flatten(x4, 1)
        if self.normalise_features:
            x4 = normalize(x4, dim=1)
        return x4

    def classify_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        return self.classify_features(feats)

    def reset_classifier(self, num_classes):
        self.net.fc = torch.nn.Linear(self.num_features, num_classes)
        for p in self.net.fc.parameters():
            p.requires_grad = True
