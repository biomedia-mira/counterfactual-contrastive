import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


import math
import torch

import pytorch_lightning as pl


class EMAWeightUpdate:
    """
    Weight updates for moving average encoder (e.g. teacher). Pl_module is expected to contain three attributes:

    Updates the target_network params using an exponential moving average update rule weighted by tau.
    Tau parameter is increased from its base value to 1.0 with every training step scheduled with a cosine function.
    global_step correspond to the total number of sgd updates expected to happen throughout the training.

    Target network is updated at the end of each SGD update on training batch.
    """

    def __init__(self, initial_tau: float = 0.995, n_optimizer=1):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau
        # work around because PL counts one step per optimiser.
        self.n_optimizer = n_optimizer

    def update_tau(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs  # type: ignore
        real_global_step = pl_module.global_step // self.n_optimizer
        assert (real_global_step / max_steps) < 1.0
        # max_steps = trainer.max_steps
        tau = (
            1
            - (1 - self.initial_tau)
            * (math.cos(math.pi * real_global_step / max_steps) + 1)
            / 2
        )
        self.current_tau = tau
        return tau

    def update_weights(
        self, online_net: torch.nn.Module, target_net: torch.nn.Module
    ) -> None:
        # apply MA weight update
        for current_params, ma_params in zip(
            online_net.parameters(), target_net.parameters()
        ):
            up_weight, old_weight = (
                current_params.data.to(ma_params.data.device),
                ma_params.data,
            )
            ma_params.data = (
                old_weight * self.current_tau + (1 - self.current_tau) * up_weight
            )

    def update_target_model(self, online_net, target_net, pl_module, trainer):
        with torch.no_grad():
            self.update_weights(online_net, target_net)
            self.update_tau(pl_module, trainer)


# https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/optimizers/lr_scheduler.py#L12
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and
    base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> #
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 2e-6,
        eta_min: float = 5e-6,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                """
                To get the last learning rate computed by the scheduler;
                please use `get_last_lr()`.
                """,
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]
