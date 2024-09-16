import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from hydra import compose, initialize
from classification.classification_module import ClassificationModule
from data_handling.xray import RSNAPneumoniaDataModule
from evaluation.helper_functions import run_inference
from pathlib import Path

os.chdir("/vol/biomedic3/mb121/causal-contrastive/evaluation")

# Mapping from human readable run name to Weights&Biases run_id.

# Human readable name should be in format:
# for finetuning:
# {simclr/simclrcf/simclrcfaug}-{train_prop}-{seed}
# for linear probing
# {simclr/simclrcf/simclrcfaug}head-{train_prop}-{seed}

model_dict_normal = {
    "simclrsexcfhead-0.1-22": "jm1b3rby",
    "simclrsexcfhead-0.1-33": "33wb4uqj",
    "simclrsexcfhead-0.1-11": "1r23eocl",
    "simclrsexcfhead-0.25-22": "h83ozgx1",
    "simclrsexcfhead-0.25-33": "q2yy4f49",
    "simclrsexcfhead-0.25-11": "r7ba7d45",
    "simclrsexcfhead-1.0-11": "18y2o53d",
    "simclrsexcfhead-1.0-22": "juno434t",
    "simclrsexcfhead-1.0-33": "rv7szigt",
    "simclrhead-0.1-11": "08xblha2",
    "simclrhead-0.1-22": "m4ty8avm",
    "simclrhead-0.1-33": "3rhzl1ec",
    "simclrhead-0.25-33": "0heae8us",
    "simclrhead-0.25-22": "vfpa4o6r",
    "simclrhead-0.25-11": "joh56v4h",
    "simclrhead-1.0-22": "186lx9pj",
    "simclrhead-1.0-33": "660mel80",
    "simclrhead-1.0-11": "tim7eazh",
}

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=["experiment=base_rsna", "data.cache=False"],
    )
    data_module = RSNAPneumoniaDataModule(config=cfg)

test_dataloader = data_module.test_dataloader()


filename = "../outputs/rsna_fair.csv"
df = pd.read_csv(filename)
for run_name, run_id in model_dict_normal.items():
    already_in_df = run_name in df.run_name.values
    if run_id != "" and not already_in_df:
        print(run_name)
        if Path(f"../outputs/run_{run_id}/best.ckpt").exists():
            model_to_evaluate = f"../outputs/run_{run_id}/best.ckpt"
            classification_model = ClassificationModule.load_from_checkpoint(
                model_to_evaluate, map_location="cuda:0", strict=False, config=cfg
            ).model.eval()
        else:
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
        df.to_csv(filename, index=False)
