from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from classification.load_model_and_config import get_modules


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def train_model_main(config):
    print(config)
    pl.seed_everything(config.seed, workers=True)

    wandb_logger = WandbLogger(save_dir="outputs", project=config.project_name)

    data_module, model_module = get_modules(config)

    wandb_logger.watch(model_module, log="all", log_freq=100)

    wandb_logger.log_hyperparams(config)

    output_dir = Path(f"outputs/run_{wandb_logger.experiment.id}")  # type: ignore
    print("Saving to" + str(output_dir.absolute()))
    checkpoint_callback = ModelCheckpoint(dirpath=output_dir, filename="{epoch}")
    checkpoint_callback_best = ModelCheckpoint(
        dirpath=output_dir,
        monitor=config.trainer.metric_to_monitor,
        mode=config.trainer.metric_to_monitor_mode,
        filename="best",
    )

    checkpoint_callback_bestbalacc = ModelCheckpoint(
        dirpath=output_dir,
        monitor="Val/BalAccuracy",
        mode="max",
        filename="best_balacc",
    )

    lr_monitor = LearningRateMonitor()
    early_stopping = EarlyStopping(
        monitor=config.trainer.metric_to_monitor,
        mode=config.trainer.metric_to_monitor_mode,
        patience=round(5 * config.trainer.patience_for_scheduler),
    )

    precision = "32-true"
    torch.set_float32_matmul_precision("medium")
    if config.mixed_precision:
        precision = "16-mixed"
    n_gpus = (
        config.trainer.device
        if isinstance(config.trainer.device, int)
        else len(config.trainer.device)
    )
    trainer = pl.Trainer(
        deterministic=True,
        accelerator="auto",
        devices=config.trainer.device,
        strategy="ddp_find_unused_parameters_true" if n_gpus > 1 else "auto",
        max_epochs=config.trainer.num_epochs
        if config.trainer.max_steps == "None"
        else -1,
        max_steps=config.trainer.max_steps
        if config.trainer.max_steps != "None"
        else -1,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            checkpoint_callback_best,
            lr_monitor,
            early_stopping,
            checkpoint_callback_bestbalacc,
        ],
        precision=precision,
        fast_dev_run=config.is_unit_test_config,
        val_check_interval=min(
            config.trainer.val_check_interval, len(data_module.train_dataloader())
        )
        if config.trainer.val_check_interval != "None"
        else None,
        limit_val_batches=250,
    )
    if config.trainer.finetune_path != "None":
        state_dict = torch.load(config.trainer.finetune_path, map_location="cuda:0")[
            "state_dict"
        ]
        state_dict.pop("model.net.fc.weight")
        state_dict.pop("model.net.fc.bias")
        print(model_module.load_state_dict(state_dict, strict=False))
        print("Model loaded successfully")

        if config.trainer.freeze_encoder:
            model_module = model_module.eval()
            for p in model_module.parameters():
                p.requires_grad = False
        else:
            model_module = model_module.train()

        model_module.model.reset_classifier(data_module.num_classes)

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    """
    Script to run one particular configuration.
    """
    torch.multiprocessing.set_sharing_strategy("file_system")
    train_model_main()
