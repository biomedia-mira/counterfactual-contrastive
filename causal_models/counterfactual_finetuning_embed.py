"""
Script to run counterfactual finetuning for EMBED model
"""

import io
import pytorch_lightning as pl
import torch
import wandb
from PIL import Image
from typing import Any

from causal_models.hps import Hparams
from causal_models.hvae import HVAE2
from causal_models.train_setup import setup_dataloaders
from causal_models.trainer import preprocess_batch
from classification.classification_module import ClassificationModule
from data_handling.base import BatchType
import matplotlib.pyplot as plt
import numpy as np


class CounterfactualTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model_dict,
        hvae_model,
        args,
    ) -> None:
        super().__init__()
        self.model_dict = torch.nn.ModuleDict(model_dict)
        for model in self.model_dict.values():
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
        self.model_dict.eval()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.hvae_model = hvae_model
        self.hvae_model.train()
        self.hvae_model.encoder.eval()
        for p in self.hvae_model.encoder.parameters():
            p.requires_grad = False
        self.args = args
        self.lmbda = torch.nn.Parameter(0 * torch.ones(1))
        self.register_buffer("eps", self.args.elbo_constraint * torch.ones(1))
        self.automatic_optimization = False
        self.save_hyperparameters()

    def common_step(self, batch: BatchType) -> Any:  # type: ignore
        col = "scanner"
        with torch.no_grad():
            batch = preprocess_batch(self.args, batch, expand_pa=False)
            _pa = torch.cat([batch[k] for k in self.args.parents_x], dim=1)
            _pa = _pa[..., None, None].repeat(1, 1, *(self.args.input_res,) * 2).float()

            cf_pa = {k: batch[k] for k in self.args.parents_x}

            if col == "scanner":
                cf_pa[col] = torch.nn.functional.one_hot(
                    torch.randint_like(torch.argmax(cf_pa[col], 1), high=5),
                    num_classes=5,
                )
            if col == "scanner_cxr":
                cf_pa[col] = torch.randint_like(torch.argmax(cf_pa[col], 1), high=2)
            _cf_pa = torch.cat([cf_pa[k] for k in self.args.parents_x], dim=1)
            _cf_pa = (
                _cf_pa[..., None, None]
                .repeat(1, 1, *(self.args.input_res,) * 2)
                .to(self.args.device)
                .float()
            )

        vae_out = self.hvae_model(batch["x"], _pa, self.args.beta)
        zs = self.hvae_model.abduct(x=batch["x"].cuda(), parents=_pa.cuda())
        if self.hvae_model.cond_prior:
            zs = [zs[j]["z"] for j in range(len(zs))]

        cf_loc, cf_scale = self.hvae_model.forward_latents(
            zs, parents=_cf_pa.cuda(), t=0.8
        )
        rec_loc, rec_scale = self.hvae_model.forward_latents(
            zs, parents=_pa.cuda(), t=0.8
        )
        u = (batch["x"] - rec_loc) / rec_scale
        cf_x = torch.clamp(cf_loc + cf_scale * u, min=-1, max=1)

        cf_x = (cf_x + 1) / 2

        predicted_intervened = self.model_dict["scanner"](cf_x)
        real_intervened = torch.argmax(cf_pa["scanner"], 1)

        aux_loss = self.criterion(predicted_intervened, real_intervened)

        with torch.no_grad():
            sg = self.eps - vae_out["elbo"]
            damp = self.args.damping * sg
        loss = aux_loss - (self.lmbda - damp) * (self.eps - vae_out["elbo"])
        return loss, aux_loss, cf_x, vae_out["elbo"]

    def training_step(self, batch: Any, batch_idx: int) -> Any:  # type: ignore
        opt, lagrange_opt = self.optimizers()
        loss, aux_loss, cf, elbo = self.common_step(batch)
        self.log("train/loss", loss)
        self.log("train/aux_loss", aux_loss)
        self.log("train/elbo", elbo)
        self.log("train/lambda", self.lmbda)
        opt.zero_grad()
        lagrange_opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        lagrange_opt.step()
        self.lmbda.data.clamp_(min=0)

        if batch_idx % 100 == 0:
            cf = cf.detach().cpu().numpy()
            self._plot_image_and_log_img_grid(cf, "train/cf")
        # return loss

    def validation_step(self, batch, batch_idx: int) -> None:  # type: ignore
        loss, aux_loss, _, elbo = self.common_step(batch)
        self.log("val/loss", loss)
        self.log("val/aux_loss", aux_loss)
        self.log("val/elbo", elbo)
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            # self.hvae_model.parameters(),
            list(self.hvae_model.decoder.parameters())
            + list(self.hvae_model.likelihood.parameters()),
            lr=5e-6,
            weight_decay=0.05,
        )
        lagrange_optimiser = torch.optim.AdamW([self.lmbda], lr=0.1, maximize=True)
        return optimizer, lagrange_optimiser

    def _plot_image_and_log_img_grid(self, data: np.ndarray, tag: str):
        f, ax = plt.subplots(2, 5, figsize=(15, 5))

        ax = ax.ravel()
        for i in range(min(10, data.shape[0])):
            img = np.transpose(data[i], [1, 2, 0])
            img = (img - img.min()) / (img.max() - img.min())
            (
                ax[i].imshow(img)
                if img.shape[-1] == 3
                else ax[i].imshow(img, cmap="gray")
            )
            ax[i].axis("off")

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")

        im = Image.open(img_buf)
        self.logger.experiment.log({tag: wandb.Image(im)})
        img_buf.close()
        plt.close()


if __name__ == "__main__":
    from pathlib import Path
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import seed_everything

    model_path = "/vol/biomedic3/mb121/causal-contrastive/outputs/scanner/beta1balanced/last_19.pt"

    seed_everything(33)

    args = Hparams()
    checkpoint = torch.load(model_path)
    args.update(checkpoint["hparams"])
    if not hasattr(args, "cond_prior"):
        args.cond_prior = False
    vae = HVAE2(args).to(args.device)
    vae.load_state_dict(checkpoint["ema_model_state_dict"])
    args.elbo_constraint = 1.1652
    args.damping = 0
    args.bs = 8

    dataloaders = setup_dataloaders(args, cache=True)

    wandb_logger = WandbLogger(
        save_dir="outputs2", project="causal_contrastive_extended"
    )

    output_dir = Path(f"outputs2/run_{wandb_logger.experiment.id}")  # type: ignore
    print("Saving to" + str(output_dir.absolute()))

    model_paths = {
        "scanner": "/vol/biomedic3/mb121/causal-contrastive/outputs2/run_vrn2dmeo/best.ckpt",
        # "density": "/vol/biomedic3/mb121/tech-demo/outputs2/run_w52hn87m/best.ckpt",
    }

    model_dict = {
        k: ClassificationModule.load_from_checkpoint(model_paths[k]).model.eval()
        for k in model_paths.keys()
    }

    model_module = CounterfactualTrainingModule(model_dict, vae, args)
    wandb_logger.watch(model_module, log="all", log_freq=100)

    callbacks = [LearningRateMonitor()]

    checkpoint_callback = ModelCheckpoint(dirpath=output_dir, filename="{epoch}")
    callbacks.append(checkpoint_callback)

    checkpoint_callback_best = ModelCheckpoint(
        dirpath=output_dir,
        monitor="val/aux_loss",
        mode="min",
        filename="best",
    )
    callbacks.append(checkpoint_callback_best)

    precision = "32-true"
    torch.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        deterministic="warn",
        accelerator="auto",
        devices=1,
        max_steps=5000,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=precision,
        val_check_interval=250,
        limit_val_batches=250,
    )

    trainer.fit(model_module, dataloaders["train"], dataloaders["valid"])
