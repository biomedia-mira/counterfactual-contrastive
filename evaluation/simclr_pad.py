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
    "supervised-0.1-11": "8vouetwb",
    "supervised-0.1-22": "96rhabqe",
    "supervised-0.1-33": "fyismsp9",
    "supervised-1.0-22": "4oaaikt6",
    "supervised-1.0-33": "n8h9oapi",
    "supervised-1.0-11": "ghopk3ju",
    "supervised-0.05-11": "yhhb9ytj",
    "supervised-0.05-22": "bx6gi5ib",
    "supervised-0.05-33": "wbhd0e7x",
    "supervised-0.25-11": "jtwn5ni9",
    "supervised-0.25-22": "yll5f08y",
    "supervised-0.25-33": "mcyuc9hr",
    "simclr-0.1-22": "n4q4a33f",
    "simclr-0.1-33": "3z4co0f5",
    "simclr-0.1-11": "u2hzkzi8",
    "simclr-1.0-22": "4pe40g5x",
    "simclr-1.0-33": "fph8u80n",
    "simclr-1.0-11": "hqvqp4mh",
    "simclrcfaug-0.1-33": "zqv28q1c",
    "simclrcfaug-1.0-33": "hu8wg6s4",
    "simclrcfaug-0.1-22": "8xf5b6ts",
    "simclrcfaug-1.0-22": "e4k7i59o",
    "simclrcfaug-0.1-11": "n7rp8bvc",
    "simclrcfaug-1.0-11": "ghtxoeph",
    "simclrcf-0.1-11": "597enhyw",
    "simclrcf-0.1-22": "erjrh75z",
    "simclrcf-1.0-22": "axb7nhet",
    "simclrcf-1.0-33": "gcn4pmb5",
    "simclrcf-1.0-11": "bjurudm1",
    "simclrcf-0.1-33": "givxkg9r",
    "simclrcf-0.05-11": "yuw1hej0",
    "simclrcf-0.25-11": "zzr4utap",
    "simclrcf-0.05-22": "46orb30r",
    "simclrcf-0.25-22": "sg6h109z",
    "simclrcf-0.05-33": "ues31163",
    "simclrcf-0.25-33": "x20kayvz",
    "simclr-0.05-11": "xuat4sf9",
    "simclr-0.25-11": "03r51wrk",
    "simclr-0.05-22": "noeakj31",
    "simclr-0.25-22": "9k2na4p0",
    "simclr-0.05-33": "qoum861b",
    "simclr-0.25-33": "9ofn816x",
    "simclrcfaug-0.05-11": "84ijd323",
    "simclrcfaug-0.25-11": "2mveqyje",
    "simclrcfaug-0.05-22": "9sg7nlcn",
    "simclrcfaug-0.25-22": "7r316b1j",
    "simclrcfaug-0.05-33": "edk3wfn3",
    "simclrcfaug-0.25-33": "yuwmsfoz",
    "simclrhead-0.25-33": "sjeg48br",
    "simclrcfaughead-0.25-33": "n3r9fiz3",
    "simclrcfaughead-0.1-33": "zjcf6740",
    "simclrcfaughead-1.0-33": "hhmtknho",
    "simclrhead-1.0-33": "maq90iwt",
    "simclraughead-0.25-22": "6to2f7a0",
    "simclrcfaughead-0.1-22": "4nfbi6tl",
    "simclrcfaughead-1.0-22": "skb167qn",
    "simclrhead-0.25-22": "cc2cdd76",
    "simclrcfaughead-0.25-11": "ozkj5npc",
    "simclrhead-0.1-22": "6hh5p3wu",
    "simclrcfaughead-0.1-11": "g9i9fshy",
    "simclrhead-1.0-22": "ujg8b6kr",
    "simclrcfaughead-1.0-11": "fzb2gg18",
    "simclrhead-0.25-11": "w5c1uilu",
    "simclrhead-0.1-11": "kjy8kd4j",
    "simclrhead-1.0-11": "h7b924ww",
    "simclrcfhead-0.25-33": "8fbh0up5",
    "simclrcfhead-0.1-33": "sfqu07v6",
    "simclrcfhead-0.25-11": "crn58bt4",
    "simclrcfhead-0.1-11": "ipxh9y6b",
    "simclrcfhead-0.25-22": "yj99p9gk",
    "simclrcfhead-1.0-11": "3cj3p5ha",
    "simclrcfaughead-0.05-11": "w8o2l2bp",
    "simclrcfaughead-0.05-22": "98wsobp9",
    "simclrhead-0.05-33": "qjb0j9cp",
    "simclrcfaughead-0.05-33": "93uflh60",
    "simclrcfhead-0.05-33": "c2ssrqf7",
    "simclrhead-0.05-22": "wzcayezk",
    "simclrcfhead-0.05-22": "lvjitftj",
    "simclrhead-0.05-11": "0t8nbhc7",
    "simclrcfhead-0.05-11": "e7kw2tuf",
    "simclrcfhead-1.0-33": "5et2yzej",
    "simclrcfhead-1.0-22": "t0zmf1lp",
}

with initialize(version_base=None, config_path="../configs"):
    cfg: DictConfig = compose(
        config_name="config.yaml",
        overrides=["experiment=base_padchestpneumo.yaml", "data.cache=True"],
    )
    print(cfg)
    data_module = PadChestDataModule(config=cfg)

test_dataloader = data_module.test_dataloader()


df = pd.read_csv("../outputs/classification_padchestfinetunepneumo_results.csv")
for run_name, run_id in model_dict_normal.items():
    already_in_df: bool = run_name in df.run_name.values
    if run_id != "" and not already_in_df:
        print(run_name)
        model_to_evaluate = f"../outputs/run_{run_id}/best.ckpt"
        classification_model = ClassificationModule.load_from_checkpoint(
            model_to_evaluate, map_location="cuda:0", strict=False
        ).model.eval()
        classification_model.cuda()

        inference_results = run_inference(test_dataloader, classification_model)
        scanners = inference_results["scanners"]
        for sc in range(2):
            sc_idx = np.where(scanners == sc)
            targets = inference_results["targets"][sc_idx]
            preds = np.argmax(inference_results["confs"], 1)[sc_idx]
            confs = inference_results["confs"][sc_idx][:, 1]
            res = {}
            res["N_test"] = [targets.shape[0]]
            res["Scanner"] = [sc]
            res["run_name"] = run_name
            res["ROC"] = [roc_auc_score(targets, confs)]
            print(res)
            df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)

        df.to_csv(
            "../outputs/classification_padchestfinetunepneumo_results.csv", index=False
        )
