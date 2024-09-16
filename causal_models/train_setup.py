import logging
import os

import send2trash
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from data_handling.sampler import SamplerFactory
from causal_models.utils import linear_warmup, seed_worker
from hydra import compose, initialize

from data_handling.mammo import EmbedDataModule
from data_handling.xray import PadChestDataModule


def setup_dataloaders(args, cache: bool = True, shuffle_train=True):
    """
    Converts our pytorch lightning data module in the format
    expected by the DSCM codebase
    """
    if "padchest" in args.hps:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=padchest",
                    f"data.cache={cache}",
                ],
            )
            print(cfg)
        data_module = PadChestDataModule(config=cfg, parents=args.parents_x)

        train_loader = DataLoader(
            data_module.dataset_train,
            data_module.config.data.batch_size,
            shuffle=False,
            num_workers=data_module.config.data.num_workers,
            pin_memory=data_module.config.data.pin_memory,
        )

    elif "embed" in args.hps:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=embed",
                    "data.batch_size=16",
                    f"data.cache={cache}",
                    f"data.exclude_cviews={'cview' not in args.parents_x}",
                ],
            )
            print(cfg)
        data_module = EmbedDataModule(config=cfg, parents=args.parents_x)
        batch_size = cfg.data.batch_size

        if shuffle_train:
            class_idx = [
                np.where(data_module.dataset_train.scanner == i)[0] for i in range(5)
            ]
            n_batches = len(data_module.dataset_train) // batch_size
            print(n_batches, len(data_module.dataset_train), batch_size)

            sampler = SamplerFactory().get(
                class_idx,
                batch_size,
                n_batches,
                alpha=0.5,
                kind="random",
            )

            train_loader = DataLoader(
                data_module.dataset_train,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                batch_sampler=sampler,
                worker_init_fn=seed_worker,
            )
        else:
            train_loader = DataLoader(
                data_module.dataset_train,
                data_module.config.data.batch_size,
                shuffle=False,
                num_workers=data_module.config.data.num_workers,
                pin_memory=data_module.config.data.pin_memory,
            )

    else:
        NotImplementedError

    dataloaders = {
        "train": train_loader,
        "valid": data_module.val_dataloader(),
    }

    return dataloaders


def setup_optimizer(args, model):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=args.betas
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=linear_warmup(args.lr_warmup_steps)
    )

    return optimizer, scheduler


def setup_directories(args, ckpt_dir="outputs"):
    parents_folder = "_".join([k for k in args.parents_x])
    save_dir = os.path.join(ckpt_dir, parents_folder, args.exp_name)
    if os.path.isdir(save_dir):
        if (
            input(f"\nSave directory '{save_dir}' already exists, overwrite? [y/N]: ")
            == "y"
        ):
            if input(f"Send '{save_dir}', to Trash? [y/N]: ") == "y":
                send2trash.send2trash(save_dir)
                print("Done.\n")
            else:
                exit()
        else:
            if (
                input(f"\nResume training with save directory '{save_dir}'? [y/N]: ")
                == "y"
            ):
                pass
            else:
                exit()
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def setup_tensorboard(args, model):
    """Setup metric summary writer."""
    writer = SummaryWriter(args.save_dir)

    hparams = {}
    for k, v in vars(args).items():
        if isinstance(v, list) or isinstance(v, torch.device):
            hparams[k] = str(v)
        elif isinstance(v, torch.Tensor):
            hparams[k] = v.item()
        else:
            hparams[k] = v

    writer.add_hparams(hparams, {"hparams": 0}, run_name=os.path.abspath(args.save_dir))

    if "vae" in type(model).__name__.lower():
        z_str = []
        if hasattr(model.decoder, "blocks"):
            for i, block in enumerate(model.decoder.blocks):
                if block.stochastic:
                    z_str.append(f"z{i}_{block.res}x{block.res}")
        else:
            z_str = ["z0_" + str(args.z_dim)]

        writer.add_custom_scalars(
            {
                "nelbo": {"nelbo": ["Multiline", ["nelbo/train", "nelbo/valid"]]},
                "nll": {"kl": ["Multiline", ["nll/train", "nll/valid"]]},
                "kl": {"kl": ["Multiline", ["kl/train", "kl/valid"]]},
            }
        )
    return writer


def setup_logging(args):
    # reset root logger
    [logging.root.removeHandler(h) for h in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt")),
            logging.StreamHandler(),
        ],
        # filemode='a',  # append to file, 'w' for overwrite
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(args.exp_name)  # name the logger
    return logger
