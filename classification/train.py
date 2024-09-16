from pathlib import Path
from omegaconf import OmegaConf
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
    output_dir = Path(f"outputs/run_{wandb_logger.experiment.id}")  # type: ignore
    print("Saving to" + str(output_dir.absolute()))

    data_module, model_module = get_modules(config)

    wandb_logger.watch(model_module, log="all", log_freq=100)

    wandb_logger.log_hyperparams(
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )

    callbacks = [LearningRateMonitor()]

    checkpoint_callback = ModelCheckpoint(dirpath=output_dir, filename="{epoch}")
    callbacks.append(checkpoint_callback)

    checkpoint_callback_best = ModelCheckpoint(
        dirpath=output_dir,
        monitor=config.trainer.metric_to_monitor,
        mode=config.trainer.metric_to_monitor_mode,
        filename="best",
    )
    callbacks.append(checkpoint_callback_best)

    if config.trainer.loss == "ce":
        early_stopping = EarlyStopping(
            monitor=config.trainer.metric_to_monitor,
            mode=config.trainer.metric_to_monitor_mode,
            patience=round(3 * config.trainer.patience_for_scheduler),
        )
        callbacks.append(early_stopping)

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
        deterministic="warn",
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
        callbacks=callbacks,
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

        if "model.net.fc.weight" in state_dict.keys():
            state_dict.pop("model.net.fc.weight")
            state_dict.pop("model.net.fc.bias")

        if "model.student.cls_token" in state_dict.keys():
            assert config.model.encoder_name.startswith("dino_vit")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model.student"):
                    new_state_dict[k.replace("model.student", "model")] = v
            print(model_module.load_state_dict(new_state_dict, strict=False))
        else:
            print(model_module.load_state_dict(state_dict, strict=False))
        print("Model loaded successfully")

        if config.trainer.freeze_encoder:
            model_module = model_module.eval()
            for p in model_module.parameters():
                p.requires_grad = False
        else:
            model_module = model_module.train()

        model_module.model.reset_classifier(data_module.num_classes)

    elif config.model.pretrained_encoder_path != "None":
        state_dict = torch.load(
            config.model.pretrained_encoder_path, map_location="cuda:0"
        )
        if config.trainer.loss == "dino":
            state_dict_update = {}
            for k, v in state_dict.items():
                if k != "pos_embed" and not k.startswith("patch_embed"):
                    if "blocks" in k:
                        new_key = f"model.student.{k}"
                        new_key = new_key.replace("blocks.", "blocks.0.")
                        state_dict_update[new_key] = v
                        new_key = new_key.replace("student.", "teacher.")
                        state_dict_update[new_key] = v
                    else:
                        state_dict_update[f"model.student.{k}"] = v
                        state_dict_update[f"model.teacher.{k}"] = v
            print(model_module.load_state_dict(state_dict_update, strict=False))
        else:
            print(model_module.load_state_dict(state_dict, strict=False))

        print("Pretrained model loaded successfully")
    trainer.fit(
        model_module,
        data_module,
    )


if __name__ == "__main__":
    """
    Script to run one particular configuration.
    """
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    torch.multiprocessing.set_sharing_strategy("file_system")
    train_model_main()
