import hydra

from classification.classification_module import ClassificationModule


def get_modules(config, shuffle_training: bool = True):
    """
    Returns model and data modules according to Hydra config.
    """
    data_module_cls = hydra.utils.get_class(config.data._target_)
    data_module = data_module_cls(config=config, shuffle=shuffle_training)

    module = ClassificationModule(
        config=config,
        num_classes=data_module.num_classes,
    )

    return data_module, module
