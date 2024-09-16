import copy
import os
from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from causal_models.plotting_utils import write_images
from causal_models.utils import linear_warmup


def preprocess_batch(args, batch, expand_pa=False):
    batch["x"] = (batch["x"].to(args.device).float() * 2) - 1
    batch["pa"] = batch["pa"].to(args.device).float()
    if expand_pa:  # used for HVAE parent concatenation
        batch["pa"] = batch["pa"][..., None, None].repeat(1, 1, *(args.input_res,) * 2)
    for k in batch.keys():
        if (
            k not in ["x", "pa", "dicom_id"]
            and not isinstance(batch[k], List)
            and batch[k].ndim == 1
        ):
            batch[k] = batch[k].reshape(-1, 1)
    return batch


def trainer(args, model, ema, dataloaders, optimizer, scheduler, writer, logger):
    for k in sorted(vars(args)):
        logger.info(f"--{k}={vars(args)[k]}")
    logger.info(f"total params: {sum(p.numel() for p in model.parameters()):,}")

    def run_epoch(dataloader, viz_batch=None, training=True):
        model.train(training)
        model.zero_grad(set_to_none=True)
        stats = {k: 0 for k in ["elbo", "nll", "kl", "n"]}
        updates_skipped = 0

        mininterval = 0.1
        loader = tqdm(
            enumerate(dataloader), total=len(dataloader), mininterval=mininterval
        )

        for i, batch in loader:
            batch = preprocess_batch(args, batch, expand_pa=args.expand_pa)
            bs = batch["x"].shape[0]
            update_stats = True

            if training:
                args.iter = i + 1 + (args.epoch - 1) * len(dataloader)
                if args.beta_warmup_steps > 0:
                    args.beta = args.beta_target * linear_warmup(
                        args.beta_warmup_steps
                    )(args.iter)
                writer.add_scalar("train/beta_kl", args.beta, args.iter)

                out = model(batch["x"], batch["pa"], beta=args.beta)
                out["elbo"] = out["elbo"] / args.accu_steps
                out["elbo"].backward()

                if i % args.accu_steps == 0:  # gradient accumulation update
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                    writer.add_scalar("train/grad_norm", grad_norm, args.iter)
                    nll_nan = torch.isnan(out["nll"]).sum()
                    kl_nan = torch.isnan(out["kl"]).sum()

                    if grad_norm < args.grad_skip and nll_nan == 0 and kl_nan == 0:
                        optimizer.step()
                        scheduler.step()
                        ema.update()
                    else:
                        updates_skipped += 1
                        update_stats = False
                        logger.info(
                            f"Updates skipped: {updates_skipped}"
                            + f" - grad_norm: {grad_norm:.3f}"
                            + f" - nll_nan: {nll_nan.item()} - kl_nan: {kl_nan.item()}"
                        )

                    model.zero_grad(set_to_none=True)

            if args.iter % args.viz_freq == 0 or (args.iter in early_evals):
                with torch.no_grad():
                    ema.ema_model.train(False)
                    write_images(args, ema.ema_model, viz_batch)

            with torch.no_grad():
                ema.ema_model.train(False)
                out = ema.ema_model(batch["x"], batch["pa"], beta=args.beta)

            if update_stats:
                if training:
                    out["elbo"] *= args.accu_steps
            stats["n"] += bs  # samples seen counter
            stats["elbo"] += out["elbo"].detach().item() * bs
            stats["nll"] += out["nll"].detach().item() * bs
            stats["kl"] += out["kl"].detach().item() * bs

            split = "train" if training else "valid"
            loader.set_description(
                f' => {split} | nelbo: {stats["elbo"] / stats["n"]:.5f}'
                + f' - nll: {stats["nll"] / stats["n"]:.5f}'
                + f' - kl: {stats["kl"] / stats["n"]:.3f}'
                + f" - lr: {scheduler.get_last_lr()[0]:.6g}"
                + (f" - grad norm: {grad_norm:.2f}" if training else ""),
                refresh=False,
            )

        return {k: v / stats["n"] for k, v in stats.items() if k != "n"}

    if args.beta_warmup_steps > 0:
        args.beta_target = copy.deepcopy(args.beta)

    viz_batch = next(iter(dataloaders["valid"]))
    # expand pa to input res, used for HVAE parent concatenation
    args.expand_pa = True
    viz_batch = preprocess_batch(args, viz_batch, expand_pa=args.expand_pa)
    early_evals = set([args.iter + 1] + [args.iter + 2**n for n in range(3, 14)])

    # Start training loop
    for epoch in range(args.start_epoch, args.epochs):
        args.epoch = epoch + 1
        logger.info(f"Epoch {args.epoch}:")

        stats = run_epoch(dataloaders["train"], viz_batch, training=True)

        writer.add_scalar("nelbo/train", stats["elbo"], args.epoch)
        writer.add_scalar("nll/train", stats["nll"], args.epoch)
        writer.add_scalar("kl/train", stats["kl"], args.epoch)
        logger.info(
            f'=> train | nelbo: {stats["elbo"]:.4f}'
            + f' - nll: {stats["nll"]:.4f} - kl: {stats["kl"]:.4f}'
            + f" - steps: {args.iter}"
        )

        save_dict = {
            "epoch": args.epoch,
            "step": args.epoch * len(dataloaders["train"]),
            "best_loss": args.best_loss,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema.ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "hparams": vars(args),
        }

        if (args.epoch - 1) % args.eval_freq == 0:
            valid_stats = run_epoch(dataloaders["valid"], training=False)

            writer.add_scalar("nelbo/valid", valid_stats["elbo"], args.epoch)
            writer.add_scalar("nll/valid", valid_stats["nll"], args.epoch)
            writer.add_scalar("kl/valid", valid_stats["kl"], args.epoch)
            logger.info(
                f'=> valid | nelbo: {valid_stats["elbo"]:.4f}'
                + f' - nll: {valid_stats["nll"]:.4f} - kl: {valid_stats["kl"]:.4f}'
                + f" - steps: {args.iter}"
            )

            if valid_stats["elbo"] < args.best_loss:
                args.best_loss = valid_stats["elbo"]
                ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
                torch.save(save_dict, ckpt_path)
                logger.info(f"Model saved: {ckpt_path}")

        ckpt_path = os.path.join(args.save_dir, "last.pt")
        torch.save(save_dict, ckpt_path)
        logger.info(f"Model saved: {ckpt_path}")

    return
