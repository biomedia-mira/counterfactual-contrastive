import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from hydra import compose, initialize
from omegaconf import DictConfig
from classification.classification_module import ClassificationModule
from data_handling.xray import PadChestDataModule
from evaluation.helper_functions import run_inference

os.chdir("/vol/biomedic3/mb121/causal-contrastive/evaluation")

# Mapping from human readable run name to Weights&Biases run_id.

# Human readable name should be in format:
# for finetuning:
# {simclr/simclrcf/simclrcfaug}-{train_prop}-{seed}
# for linear probing
# {simclr/simclrcf/simclrcfaug}head-{train_prop}-{seed}

model_dict_normal = {
    "simclrhead-0.25-33": "sjeg48br",
    "simclrhead-1.0-33": "maq90iwt",
    "simclrhead-0.25-22": "cc2cdd76",
    "simclrhead-0.1-22": "6hh5p3wu",
    "simclrhead-1.0-22": "ujg8b6kr",
    "simclrhead-0.25-11": "w5c1uilu",
    "simclrhead-0.1-11": "kjy8kd4j",
    "simclrhead-1.0-11": "h7b924ww",
    "simclrhead-0.05-33": "qjb0j9cp",
    "simclrhead-0.05-22": "wzcayezk",
    "simclrhead-0.05-11": "0t8nbhc7",
    "simclrsexcfhead-0.05-22": "welaofyb",
    "simclrsexcfhead-0.05-33": "b7khph21",
    "simclrsexcfhead-0.1-22": "whi6zwoe",
    "simclrsexcfhead-0.1-11": "zs6kud02",
    "simclrsexcfhead-0.1-33": "aai24ln0",
    "simclrsexcfhead-0.25-11": "ayfkmow2",
    "simclrsexcfhead-0.25-22": "qxknwepo",
    "simclrsexcfhead-0.25-33": "qojgi8gc",
    "simclrsexcfhead-1.0-33": "i1212pwe",
    "simclrsexcfhead-1.0-11": "zelpwy5n",
    "simclrsexcfhead-1.0-22": "9qesem2g",
}

with initialize(version_base=None, config_path="../configs"):
    cfg: DictConfig = compose(
        config_name="config.yaml",
        overrides=["experiment=base_padchestpneumo.yaml", "data.cache=True"],
    )
    print(cfg)
    data_module = PadChestDataModule(config=cfg)

test_dataloader = data_module.test_dataloader()


df = pd.read_csv("../outputs/padchest_fair.csv")
for run_name, run_id in model_dict_normal.items():
    already_in_df: bool = run_name in df.run_name.values
    if run_id != "" and not already_in_df:
        print(run_name)
        try:
            model_to_evaluate = f"../outputs/run_{run_id}/best.ckpt"
            classification_model = ClassificationModule.load_from_checkpoint(
                model_to_evaluate, map_location="cuda:0", strict=False, config=cfg
            ).model.eval()
        except:  # noqa
            model_to_evaluate = f"../outputs2/run_{run_id}/best.ckpt"
            classification_model = ClassificationModule.load_from_checkpoint(
                model_to_evaluate,
                map_location="cuda:0",
                strict=False,
            ).model.eval()
        classification_model.cuda()

        inference_results = run_inference(test_dataloader, classification_model)
        sexs = inference_results["sexs"]
        for sc in range(2):
            sc_idx = np.where(sexs == sc)
            targets = inference_results["targets"][sc_idx]
            preds = np.argmax(inference_results["confs"], 1)[sc_idx]
            confs = inference_results["confs"][sc_idx][:, 1]
            res = {}
            res["N_test"] = [targets.shape[0]]
            res["Sex"] = [sc]
            res["run_name"] = run_name
            res["ROC"] = [roc_auc_score(targets, confs)]
            print(res)
            df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)

        df.to_csv("../outputs/padchest_fair.csv", index=False)
