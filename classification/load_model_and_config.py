import hydra
import torch

from classification.classification_module import ClassificationModule


def get_modules(config, shuffle_training: bool = True):
    """
    Returns model and data modules according to Hydra config.
    """
    data_module_cls = hydra.utils.get_class(config.data._target_)
    data_module = data_module_cls(config=config, shuffle=shuffle_training)

    # Create PL module
    module = ClassificationModule(
        encoder_name=config.model.encoder_name,
        pretrained=config.model.pretrained,
        num_classes=data_module.num_classes,
        lr=config.trainer.lr,
        patience_for_scheduler=config.trainer.patience_for_scheduler,
        metric_to_monitor=config.trainer.metric_to_monitor,
        metric_to_monitor_mode=config.trainer.metric_to_monitor_mode,
        weight_decay=config.trainer.weight_decay,
        loss=config.trainer.loss,
        contrastive_temperature=config.trainer.contrastive_temperature,
        input_channels=config.data.input_channels,
        weights=torch.tensor(config.data.weights)
        if config.data.weights != "None"
        else None,
    )

    return data_module, module
