import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from hydra import compose, initialize


from classification.classification_module import ClassificationModule
from data_handling.xray import RSNAPneumoniaDataModule
from evaluation.helper_functions import run_inference

os.chdir("/vol/biomedic3/mb121/causal-contrastive/evaluation")

# Mapping from human readable run name to Weights&Biases run_id.

# Human readable name should be in format:
# for finetuning:
# {simclr/simclrcf/simclrcfaug}-{train_prop}-{seed}
# for linear probing
# {simclr/simclrcf/simclrcfaug}head-{train_prop}-{seed}

model_dict_normal = {
    "supervised-0.1-11": "lnofy8px",
    "supervised-0.1-22": "ck3nhurh",
    "supervised-0.1-33": "fq85b7yv",
    "supervised-1.0-11": "iasr1ngi",
    "supervised-1.0-22": "aq28vr4n",
    "supervised-1.0-33": "jzecmpej",
    "supervised-0.25-11": "vgbe3kaw",
    "supervised-0.25-22": "tb0gu42b",
    "supervised-0.25-33": "uc7ew0bk",
    "simclrcfaug-1.0-33": "johrigco",
    "simclrcfaug-0.1-33": "mwj1n15o",
    "simclrcfaug-1.0-22": "urupcr8p",
    "simclrcfaug-0.1-22": "zhqlwbiv",
    "simclrcfaug-1.0-11": "u065scjn",
    "simclrcfaug-0.1-11": "10ik4o6r",
    "simclrcfaug-0.25-11": "ia70qkb8",
    "simclrcfaug-0.25-22": "u4npk2ti",
    "simclrcfaug-0.25-33": "wjx8atvt",
    "simclrcf-0.1-11": "dmhnp0bi",
    "simclrcf-1.0-11": "exuhveyd",
    "simclrcf-0.1-22": "79sei22s",
    "simclrcf-1.0-22": "pslj9cw7",
    "simclrcf-0.1-33": "q3poikhz",
    "simclrcf-1.0-33": "fjb1jc33",
    "simclrcf-0.25-11": "drpdrih7",
    "simclrcf-0.25-22": "ejxf43d7",
    "simclrcf-0.25-33": "nvgwji8u",
    "simclr-0.25-11": "4o9d80gc",
    "simclr-0.25-22": "o3vvj758",
    "simclr-0.25-33": "r3iwi9qf",
    "simclr-0.1-11": "ie25zeoi",
    "simclr-0.1-22": "wkgvu1ij",
    "simclr-0.1-33": "r6ga87wc",
    "simclr-1.0-11": "tgdoaxmj",
    "simclr-1.0-22": "gxhens59",
    "simclr-1.0-33": "hsxljb6i",
    "simclrcfhead-0.25-33": "lp4urzn2",
    "simclrcfhead-0.25-22": "iv9ew6eu",
    "simclrcfhead-0.25-11": "si567ykr",
    "simclrcfhead-0.1-11": "cq6t04x6",
    "simclrcfhead-0.1-22": "pwzw0imf",
    "simclrcfhead-0.1-33": "27woawwy",
    "simclrcfhead-1.0-33": "qeejzwbk",
    "simclrcfhead-1.0-22": "9mkozfi6",
    "simclrcfhead-1.0-11": "7qb4671t",
    "simclrcfaughead-0.25-22": "0h644xfp",
    "simclrcfaughead-0.25-33": "bctcb405",
    "simclrcfaughead-0.25-11": "pyal1u9w",
    "simclrcfaughead-0.1-11": "m4t67j2z",
    "simclrcfaughead-0.1-22": "73m2zbrr",
    "simclrcfaughead-0.1-33": "2w7tf1q4",
    "simclrcfaughead-1.0-33": "v5ginyi5",
    "simclrcfaughead-1.0-22": "1408m35b",
    "simclrcfaughead-1.0-11": "bb4gq5r2",
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
        config_name="config.yaml", overrides=["experiment=base_rsna", "data.cache=True"]
    )
    data_module = RSNAPneumoniaDataModule(config=cfg)

test_dataloader = data_module.test_dataloader()

df = pd.read_csv("../outputs/classification_rsna_results.csv")
df = pd.DataFrame()
for run_name, run_id in model_dict_normal.items():
    already_in_df = False  # run_name in df.run_name.values
    if run_id != "" and not already_in_df:
        print(run_name)
        model_to_evaluate = f"../outputs/run_{run_id}/best.ckpt"
        classification_model = ClassificationModule.load_from_checkpoint(
            model_to_evaluate, map_location="cuda:0", strict=False
        ).model.eval()
        classification_model.cuda()
        inference_results = run_inference(test_dataloader, classification_model)
        print("\nEvaluating RSNA")
        res = {}
        res["N_test"] = [inference_results["targets"].shape[0]]
        res["Scanner"] = ["RSNA"]
        res["run_name"] = run_name
        res["ROC"] = [
            roc_auc_score(
                inference_results["targets"], inference_results["confs"][:, 1]
            )
        ]
        print(res)
        df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)
        df.to_csv("../outputs/classification_rsna_results.csv", index=False)
