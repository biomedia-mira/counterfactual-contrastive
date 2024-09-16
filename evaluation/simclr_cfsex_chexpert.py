import os
import pandas as pd
from hydra import compose, initialize
from sklearn.metrics import roc_auc_score
from data_handling.xray import CheXpertDataModule
from classification.classification_module import ClassificationModule
from evaluation.helper_functions import run_inference
import numpy as np

os.chdir("/vol/biomedic3/mb121/causal-contrastive/evaluation")

# Mapping from human readable run name to Weights&Biases run_id.

# Human readable name should be in format:
# for finetuning:
# {simclr/simclrcf/simclrcfaug}-{train_prop}-{seed}
# for linear probing
# {simclr/simclrcf/simclrcfaug}head-{train_prop}-{seed}

model_dict_normal: dict[str, str] = {
    "simclrhead-0.25-33": "85yznxgy",
    "simclrhead-1.0-33": "95qyx64t",
    "simclrhead-0.1-33": "nlvsmcne",
    "simclrhead-0.25-22": "uxu4dwzu",
    "simclrhead-1.0-22": "9m18itid",
    "simclrhead-0.25-11": "6iwtu99y",
    "simclrhead-0.1-11": "hdimroq2",
    "simclrhead-1.0-11": "8o5owv7w",
    "simclrsexcfhead-0.1-33": "39x0pt5a",
    "simclrsexcfhead-0.1-22": "md93vzp5",
    "simclrsexcfhead-0.1-11": "zi5vrlum",
    "simclrsexcfhead-0.25-11": "6dprf6l2",
    "simclrsexcfhead-0.25-33": "03lq2wer",
    "simclrsexcfhead-0.25-22": "lyj4z6yt",
    "simclrsexcfhead-1.0-11": "xu0n688b",
    "simclrsexcfhead-1.0-22": "an1986jd",
    "simclrsexcfhead-1.0-33": "clzv291f",
}

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=[
            "experiment=base_padchestpneumo",
            "data=chexpert",
            "data.label=Pneumonia",
            "data.cache=False",
        ],
    )
    data_module = CheXpertDataModule(config=cfg)

test_dataloader = data_module.test_dataloader()

df = pd.read_csv("../outputs/chexpert_fair.csv")

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

        df.to_csv("../outputs/chexpert_fair.csv", index=False)
