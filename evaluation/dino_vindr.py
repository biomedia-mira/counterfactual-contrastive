import os
import pandas as pd
import numpy as np
from hydra import compose, initialize
from sklearn.metrics import roc_auc_score
from classification.classification_module import ClassificationModule
from data_handling.mammo import VinDrDataModule
from evaluation.helper_functions import run_inference

os.chdir("/vol/biomedic3/mb121/causal-contrastive/evaluation")
# Mapping from human readable run name to Weights&Biases run_id.

# Human readable name should be in format:
# for finetuning:
# {simclr/simclrcf/simclrcfaug}-{train_prop}-{seed}
# for linear probing
# {simclr/simclrcf/simclrcfaug}head-{train_prop}-{seed}

model_dict = {
    # "dinocfhead-0.05-22": "i4eth85t",
    # "dinocfhead-0.05-33": "im28fn12",
    # "dinocfhead-0.05-11": "lcpg7rnn",
    # "dinocfhead-0.1-22": "mswrbsea",
    # "dinocfhead-0.1-11": "af4o3ihe",
    # "dinocfhead-0.1-33": "fu95fjt5",
    # "dinocfhead-0.25-22": "erd5zjdt",
    # "dinocfhead-0.25-11": "tz5gmyjl",
    # "dinocfhead-0.25-33": "x67rheef",
    # "dinocfhead-1.0-11": "9zoghcgc",
    # "dinocfhead-1.0-22": "013cf1zb",
    # "dinocfhead-1.0-33": "ljq44n4i",
    "dinohead-0.05-33": "c9lzh8oj",
    "dinohead-0.05-22": "o8aua0gx",
    "dinohead-0.05-11": "dyeztdt4",
    "dinohead-0.1-33": "qqakceol",
    "dinohead-0.1-11": "rfinh8nm",
    "dinohead-0.1-22": "kufojxof",
    "dinohead-0.25-22": "svztu2s6",
    "dinohead-0.25-11": "kz0vssve",
    "dinohead-0.25-33": "ek9g54xr",
    "dinohead-1.0-11": "b1mkjvb7",
    "dinohead-1.0-22": "ryuojun7",
    "dinohead-1.0-33": "zfpilsrx",
    "dino-0.05-22": "35xadmb7",
    "dino-0.05-11": "1u6hffjd",
    "dino-0.05-33": "rwe0z551",
    "dino-0.1-11": "p62vcx7k",
    "dino-0.1-22": "yz8awyov",
    "dino-0.1-33": "v8icf6km",
    "dino-0.25-11": "2xtbi141",
    "dino-0.25-22": "3fant9n6",
    "dino-0.25-33": "896pqhu9",
    "dino-1.0-11": "tsvrxhfn",
    "dino-1.0-22": "2bl7x4lu",
    "dino-1.0-33": "uv6xmj5r",
    # "dinocf-1.0-22": "lcu4odxy",
    # "dinocf-1.0-33": "jc2t0g8w",
    # "dinocf-0.25-33": "gnozmx3g",
    # "dinocf-1.0-11": "qlag4oiv",
    # "dinocf-0.05-33": "apz6q31a",
    # "dinocf-0.05-22": "bjrowkfl",
    # "dinocf-0.05-11": "ot767q3g",
    # "dinocf-0.1-33": "4bh6v83w",
    # "dinocf-0.1-22": "aicyjwma",
    # "dinocf-0.1-11": "fbak5svf",
    # "dinocf-0.25-22": "2uyermd7",
    # "dinocf-0.25-11": "u3gdzs8h",
    "dinocfaughead-0.05-11": "g3zi15us",
    "dinocfaughead-0.1-11": "bbyj2xn0",
    "dinocfaughead-0.25-11": "izgwz9vw",
    "dinocfaughead-0.05-22": "wsuicj4u",
    "dinocfaughead-1.0-11": "rk5h3yov",
    "dinocfaughead-0.1-22": "xbhstdqi",
    "dinocfaughead-0.25-22": "838irpqm",
    "dinocfaughead-0.05-33": "coaldixc",
    "dinocfaughead-0.1-33": "dqk9l4ju",
    "dinocfaughead-1.0-22": "96z2qfsf",
    "dinocfaughead-0.25-33": "j7dc0fk3",
    "dinocfaughead-1.0-33": "4ooecna8",
    "dinocfaug-0.1-11": "61hqmz7p",
    "dinocfaug-0.05-11": "y5sb3paz",
    "dinocfaug-0.05-22": "dn2q1fdr",
    "dinocfaug-0.25-11": "m662l3lk",
    "dinocfaug-0.1-22": "z9nmbjol",
    "dinocfaug-1.0-11": "s6aafmor",
    "dinocfaug-0.25-22": "ljoa9sgr",
    "dinocfaug-0.25-33": "jypjee2x",
    "dinocfaug-0.05-33": "6r3cxehs",
    "dinocfaug-0.1-33": "mqrhu3sy",
    "dinocfaug-1.0-33": "ak9jzocg",
    "dinocfaug-1.0-22": "bklmml4j",
    "dinocf2-1.0-33": "oip5uin2",
    "dinocf2-0.25-33": "f779rck9",
    "dinocf2-0.1-11": "1ot0yut3",
    "dinocf2-0.05-11": "0yd25jx9",
    "dinocf2head-0.05-11": "fw25hqqb",
    "dinocf2head-0.25-22": "uzcy437u",
    "dinocf2head-1.0-22": "poy96510",
    "dinocf2head-0.25-11": "rbq3829w",
    "dinocf2-0.05-11": "0yd25jx9",
    "dinocf2-0.05-22": "0eu2sfpv",
    "dinocf2-0.25-11": "rcf1q6h3",
    "dinocf2head-0.1-11": "0hum57la",
    "dinocf2-1.0-22": "ce9z8j5e",
    "dinocf2-0.1-33": "og7ee269",
    "dinocf2-0.05-33": "79925xbr",
    "dinocf2head-1.0-33": "hczr2xun",
    "dinocf2-0.1-22": "ablx9oyp",
    "dinocf2-0.25-22": "k45zjvrl",
    "dinocf2-1.0-11": "wshwpzba",
    "dinocf2head-0.05-33": "lakb4tlo",
    "dinocf2head-0.05-22": "lhcwfjxj",
    "dinocf2head-0.1-22": "fikh4j1y",
    "dinocf2-1.0-22": "ce9z8j5e",
    "dinocf2-0.1-33": "og7ee269",
    "dinocf2head-1.0-11": "gybohu26",
    "dinocf2head-0.25-33": "fpymsjqq",
    "dinocf2head-0.1-33": "xxpdnf3h",
}


with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=["experiment=base_density", "data=vindr", "data.cache=False"],
    )
    data_module = VinDrDataModule(config=cfg)
test_loader = data_module.test_dataloader()


filename = "../outputs/dino_vindr2.csv"


df = pd.read_csv(filename)
for run_name, run_id in model_dict.items():
    already_in_df = run_name in df.run_name.values
    if run_id != "" and not already_in_df:
        print(run_name)
        model_to_evaluate = f"../outputs2/run_{run_id}/best.ckpt"
        classification_model = ClassificationModule.load_from_checkpoint(
            model_to_evaluate, map_location="cuda"
        ).model.eval()
        classification_model.cuda()

        # ID evaluation
        inference_results = run_inference(test_loader, classification_model)
        scanners = inference_results["scanners"]

        for i in np.unique(scanners):
            sc_idx = np.where(scanners == i)
            targets = inference_results["targets"][sc_idx]
            preds = np.argmax(inference_results["confs"], 1)[sc_idx]
            confs = inference_results["confs"][sc_idx]
            if np.unique(targets).shape[0] == 4:
                res = {}
                res["N_test"] = [targets.shape[0]]
                res["ROC"] = [roc_auc_score(targets, confs, multi_class="ovr")]
                res["Model Name"] = [f"(OOD) VinDr\n{i}"]
                res["run_name"] = run_name
                print(res)
                df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)

        targets = inference_results["targets"]
        preds = np.argmax(inference_results["confs"], 1)
        confs = inference_results["confs"]
        res = {}
        res["N_test"] = [targets.shape[0]]
        res["ROC"] = [roc_auc_score(targets, confs, multi_class="ovr")]
        res["Model Name"] = ["(OOD) VinDr"]
        res["run_name"] = run_name
        print(res)
        df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)
        df.to_csv(filename, index=False)
