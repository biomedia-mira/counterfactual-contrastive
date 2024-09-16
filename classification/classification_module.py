import io
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any

from classification.dinov2.dino_model import CosineScheduler
from classification.dinov2 import dino_model
from classification.resnet import ResNetBase
from classification.utils import EMAWeightUpdate, LinearWarmupCosineAnnealingLR
from classification.losses import nt_xent_loss
from data_handling.base import BatchType


class ClassificationModule(pl.LightningModule):
    """
    A generic PL module for classification
    """

    def __init__(
        self,
        config,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.config = config
        self.loss = config.trainer.loss
        self.model = self.get_model()
        weights = (
            torch.tensor(config.data.weights) if config.data.weights != "None" else None
        )

        match self.loss:
            case "ce":
                self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
            case "simclr":
                self.contrastive_temperature = config.trainer.contrastive_temperature
                self.head_criterion = torch.nn.CrossEntropyLoss(weight=weights)
                self.model.projection_head = torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=self.model.num_features, out_features=1024
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_features=1024, out_features=128),
                )
            case "dino":
                self.weight_callback = EMAWeightUpdate(n_optimizer=2)
                self.teacher_temp_scheduler = CosineScheduler(
                    base_value=self.config.trainer.teacher_temp,
                    final_value=self.config.trainer.teacher_temp,
                    warmup_iters=self.config.trainer.warmup_epochs,
                    start_warmup_value=self.config.trainer.warmup_teacher_temp,
                    total_iters=self.config.trainer.num_epochs,
                )

                self.wd_scheduler = CosineScheduler(
                    base_value=self.config.trainer.wd_start,
                    final_value=self.config.trainer.wd_end,
                    total_iters=self.config.trainer.num_epochs,
                )

                self.teacher_temp = self.config.trainer.teacher_temp
                self.model.fc = torch.nn.Linear(
                    self.model.student.embed_dim, self.num_classes
                )
                self.head_criterion = torch.nn.CrossEntropyLoss(weight=weights)

            case _:
                raise NotImplementedError

        self.save_hyperparameters()
        self.automatic_optimization = self.loss not in [
            "simclr",
            "dino",
        ]
        self.is_dino_vit = self.config.model.encoder_name.startswith("dino_vit")

    def common_step(self, batch: BatchType, prefix: str, batch_idx: int) -> Any:  # type: ignore
        data, target = batch["x"], batch["y"]
        if data.ndim == 5:
            bsz, n_views, c, w, h = data.shape
            if self.loss == "dino":
                data = data[:, 0]
            else:
                data = data.reshape((-1, c, w, h))
        else:
            if self.loss in ["simclr"] and self.training:
                raise ValueError(
                    "Your dataloader returns 1 view per image but you use a contrastive loss. Not expected."
                )
            bsz, n_views = data.shape[0], 1

        if self.loss in ["simclr", "dino"]:
            match self.loss:
                case "simclr":
                    features = self.model.get_features(data)
                    proj_features = self.model.projection_head(features)
                    proj_features = normalize(proj_features, dim=1)
                    proj_features = proj_features.reshape((bsz, n_views, -1))
                    loss = 0

                    if self.training or n_views > 1:
                        loss += nt_xent_loss(proj_features[:, 0], proj_features[:, 1])
                        self.log(f"{prefix}/simclr_loss", loss.detach(), sync_dist=True)

                    output = self.model.classify_features(features.detach())

                case "dino":
                    if self.training:
                        with torch.no_grad():
                            self.log("EMA/tau", self.weight_callback.current_tau)
                            self.weight_callback.update_target_model(
                                online_net=self.model.student,
                                target_net=self.model.teacher,
                                pl_module=self,
                                trainer=self.trainer,
                            )

                    loss, losses_dict = self.model.get_loss(batch, self.teacher_temp)

                    for k, v in losses_dict.items():
                        self.log(f"dino_{prefix}/{k}", v, sync_dist=True)

                    features = self.model.student(data)
                    output = self.model.fc(features.detach())
                    if self.training and self.current_epoch == 0 and batch_idx < 5:
                        data = batch["x"]
                        data = data.cpu().numpy()
                        self._plot_image_and_log_img_grid(data, None, "train/inputs")
                        data = batch["collated_global_crops"].reshape(
                            (
                                2,
                                -1,
                                batch["x"].shape[-3],
                                batch["x"].shape[-2],
                                batch["x"].shape[-1],
                            )
                        )
                        data = torch.permute(data, (1, 0, 2, 3, 4))
                        data = data.cpu().numpy()
                        self._plot_image_and_log_img_grid(data, None, "train/global")
                        data = batch["collated_local_crops"].reshape(
                            (
                                8,
                                -1,
                                batch["x"].shape[-3],
                                batch["collated_local_crops"].shape[-2],
                                batch["collated_local_crops"].shape[-1],
                            )
                        )[[0, 7]]
                        data = torch.permute(data, (1, 0, 2, 3, 4))
                        data = data.cpu().numpy()
                        self._plot_image_and_log_img_grid(data, None, "train/local")

            if n_views > 1 and self.loss != "dino":
                target = torch.stack([target, target], dim=1).reshape(-1)

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

            if loss is not None:
                self.log(f"{prefix}/loss", loss.detach(), sync_dist=True)
                if torch.isnan(loss):
                    raise ValueError("Found loss Nan")

            self.log(f"{prefix}/head_loss", head_loss, sync_dist=True)

        else:
            output = self.model(data)
            if n_views > 1:
                target = torch.stack([target, target], dim=1).reshape(-1)
            loss = self.criterion(output, target)

            self.log(f"{prefix}/loss", loss)

        probas = torch.softmax(output, 1)
        return loss, probas, target

    def training_step(self, batch: Any, batch_idx: int) -> Any:  # type: ignore
        loss, probas, targets = self.common_step(
            batch, prefix="Train", batch_idx=batch_idx
        )
        self.train_probas.append(probas.detach().cpu())
        self.train_targets.append(targets.detach().cpu())

        # if batch_idx == 0 and self.current_epoch < 3:
        #     data = batch["x"]
        #     data = data.cpu().numpy()
        #     self._plot_image_and_log_img_grid(
        #         data, targets.cpu().numpy(), "train/inputs"
        #     )
        # if batch_idx % 50 == 0:

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
        if self.loss in ["dino"]:
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()
            if self.loss == "dino":
                self.teacher_temp = self.teacher_temp_scheduler[self.current_epoch]

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
        self.log(f"{prefix}/Accuracy", accuracy_score(targets, preds), sync_dist=True)
        self.log(
            f"{prefix}/BalAccuracy",
            balanced_accuracy_score(targets, preds),
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
            self.model.parameters(),
            lr=self.config.trainer.lr,
            weight_decay=self.config.trainer.weight_decay,
        )
        if self.loss == "ce":
            scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.config.trainer.patience_for_scheduler,
                    mode=self.config.trainer.metric_to_monitor_mode,
                    min_lr=1e-5,
                ),
                "monitor": self.config.trainer.metric_to_monitor,
            }
            return [optimizer], [scheduler]
        else:
            head_optimizer = torch.optim.Adam(
                (
                    self.model.fc.parameters()
                    if hasattr(self.model, "fc")
                    else self.model.net.fc.parameters()
                ),
                lr=1e-4,
                weight_decay=1e-4,
            )
            if self.loss in ["dino"] and self.config.trainer.use_lr_scheduler:
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.config.trainer.warmup_epochs,
                    max_epochs=self.trainer.max_epochs,
                )
                return [optimizer, head_optimizer], [scheduler]
            return optimizer, head_optimizer

    def get_model(self) -> torch.nn.Module:
        target_size = (
            self.config.data.augmentations.center_crop
            if self.config.data.augmentations.center_crop != "None"
            else self.config.data.augmentations.resize
        )

        if self.loss == "dino":
            from classification.dinov2.dino_model import MySSLMetaArch

            return MySSLMetaArch(
                encoder_model_name=self.config.model.encoder_name,
                img_size=target_size,
                in_chans=self.config.data.input_channels,
                head_n_prototypes=self.config.trainer.head_n_prototypes,
                head_bottleneck_dim=self.config.trainer.head_bottleneck_dim,
                head_hidden_dim=self.config.trainer.head_hidden_dim,
                head_nlayers=self.config.trainer.head_nlayers,
                ibot_mask_ratio_min_max=self.config.trainer.ibot_mask_ratio_min_max,
                mask_sample_probability=self.config.trainer.mask_probability,
                koleo_loss_weight=self.config.trainer.koleo_loss_weight,
                ibot_loss_weight=self.config.trainer.ibot_loss_weight,
            )

        elif self.config.model.encoder_name.startswith("resnet"):
            return ResNetBase(
                num_classes=self.num_classes,
                encoder_name=self.config.model.encoder_name,
                pretrained=self.config.model.pretrained,
                input_channels=self.config.data.input_channels,
            )
        elif self.config.model.encoder_name.startswith("dino_vit"):
            match self.config.model.encoder_name:
                case "dino_vit_small":
                    model = dino_model.vit_small(
                        img_size=target_size,
                        patch_size=self.config.model.patch_size,
                        in_chans=self.config.data.input_channels,
                    )
                case "dino_vit_base":
                    model = dino_model.vit_base(
                        img_size=target_size,
                        patch_size=self.config.model.patch_size,
                        in_chans=self.config.data.input_channels,
                    )
                case _:
                    raise NotImplementedError

            model.fc = torch.nn.Linear(model.embed_dim, self.num_classes)
            return model
        else:
            raise ValueError

    def _plot_image_and_log_img_grid(self, data: np.ndarray, y: np.ndarray, tag: str):
        f, ax = plt.subplots(2, 5, figsize=(15, 5))

        if data.ndim == 5:
            for i in range(min(5, data.shape[0])):
                for j in range(2):
                    img = np.transpose(data[i, j], [1, 2, 0])
                    img = (img - img.min()) / (img.max() - img.min())
                    (
                        ax[j, i].imshow(img)
                        if img.shape[-1] == 3
                        else ax[j, i].imshow(img, cmap="gray")
                    )
                    if y is not None:
                        ax[j, i].set_title(y[i])
                    ax[j, i].axis("off")
        else:
            ax = ax.ravel()
            for i in range(min(10, data.shape[0])):
                img = np.transpose(data[i], [1, 2, 0])
                img = (img - img.min()) / (img.max() - img.min())
                (
                    ax[i].imshow(img)
                    if img.shape[-1] == 3
                    else ax[i].imshow(img, cmap="gray")
                )
                if y is not None:
                    ax[i].set_title(y[i])
                ax[i].axis("off")

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")

        im = Image.open(img_buf)
        self.logger.experiment.log({tag: wandb.Image(im)})
        img_buf.close()
        plt.close()
