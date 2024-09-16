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
    "supervised-0.1-33": "p6vg0bqi",
    "supervised-0.5-33": "cwi8xrg8",
    "supervised-1.0-33": "14q2njao",
    "supervised-0.1-22": "s8miws7d",
    "supervised-0.5-22": "x2muh4t0",
    "supervised-1.0-22": "mbdkezyb",
    "supervised-0.1-11": "exzxnswf",
    "supervised-0.5-11": "lvusaac3",
    "supervised-1.0-11": "tnzfk6l2",
    "supervised-0.05-11": "2ms8yvko",
    "supervised-0.05-22": "pobuespp",
    "supervised-0.05-33": "a81wveb4",
    "supervised-0.25-11": "thaje7u1",
    "supervised-0.25-22": "wq8yo3j8",
    "supervised-0.25-33": "pvs3glhw",
    "simclrcfaug-0.05-11": "rysd390e",
    "simclr-0.05-11": "9l7y5ut7",
    "simclrcfaug-0.1-11": "znaz1mde",
    "simclrcf-0.05-11": "p95njer4",
    "simclrcf-0.1-11": "iw7tlt92",
    "simclrcfaug-1.0-11": "nyvzr6ob",
    "simclrcfaug-0.05-22": "fhcjybmj",
    "simclr-0.1-11": "mof52mfw",
    "simclrcfaug-0.1-22": "vg3xsy4o",
    "simclrcfaug-0.25-22": "hx1na3m8",
    "simclrcf-0.25-11": "ct68hj4b",
    "simclr-0.25-11": "2458sxig",
    "simclrcfaug-0.5-22": "31sfch4u",
    "simclrcf-0.5-11": "keifisvj",
    "simclrcfaug-1.0-22": "nldvcmd9",
    "simclr-0.5-11": "x4h890xg",
    "simclrcf-1.0-11": "snfp6z9e",
    "simclr-1.0-11": "98a4nxz6",
    "simclrcf-0.05-22": "9shpau2m",
    "simclrcf-0.1-22": "7g2mzjji",
    "simclr-0.05-22": "bisabus9",
    "simclr-0.1-22": "0gczhnmx",
    "simclrcf-0.25-22": "av41b9aw",
    "simclr-0.25-22": "ehunbb4k",
    "simclrcf-0.5-22": "z28lpj3b",
    "simclr-0.5-22": "ic2srqqw",
    "simclrcf-1.0-22": "f3yuxs0n",
    "simclr-1.0-22": "n5w3srew",
    "simclrcf-0.05-33": "mvccg1m1",
    "simclr-0.05-33": "c73xfh8k",
    "simclrcf-0.1-33": "jgh63ufx",
    "simclrcf-0.25-33": "yzsrtgah",
    "simclrcf-0.5-33": "4r6q203i",
    "simclrcfaug-0.05-33": "byblnw9g",
    "simclrcfaug-0.1-33": "4ibb1j7n",
    "simclr-0.25-33": "cf5trd5v",
    "simclrcfaug-0.25-33": "b1qmartk",
    "simclrcfaug-0.5-33": "gkgfkfui",
    "simclrcf-1.0-33": "exxhffot",
    "simclr-0.1-33": "knuou61x",
    "simclrcfaug-1.0-33": "bozj3y17",
    "simclr-0.5-33": "qfbtg6f9",
    "simclr-1.0-33": "wh3lxhxz",
    "simclrcfaug-0.25-11": "ciucf48v",
    "simclrcfaug-0.5-11": "d98hdcn0",
    "simclrcfaughead-0.05-33": "9kmm950b",
    "simclrcfaughead-0.1-33": "zqt0uenu",
    "simclrcfhead-0.05-33": "mqj29vzy",
    "simclrhead-0.05-33": "beh79tru",
    "simclrcfaughead-0.25-33": "07njirrz",
    "simclrcfhead-0.1-33": "kyof9klp",
    "simclrhead-0.1-33": "4j616g3p",
    "simclrcfaughead-0.5-33": "z4jyae0t",
    "simclrcfhead-0.25-33": "a6pjsv40",
    "simclrhead-0.25-33": "hd5xx0wy",
    "simclrcfhead-0.5-33": "uuigz04q",
    "simclrcfaughead-1.0-33": "7zj3kpzs",
    "simclrhead-0.5-33": "j9z30v0i",
    "simclrcfhead-1.0-33": "eaybins7",
    "simclrhead-1.0-33": "4iopv07k",
    "simclrcfaughead-0.25-22": "cekqxksm",
    "simclrcfhead-0.05-22": "bhb010nf",
    "simclrcfaughead-0.1-22": "7gks2qd4",
    "simclrhead-0.05-22": "2wfjbbyn",
    "simclrcfhead-0.1-22": "1cfux2ew",
    "simclrhead-0.1-22": "fxuauquy",
    "simclrcfhead-0.25-22": "uualms6e",
    "simclrcfaughead-0.5-22": "xz9g395o",
    "simclrhead-0.25-22": "o5b0g7qi",
    "simclrcfhead-0.5-22": "4ktb86ju",
    "simclrhead-0.5-22": "w2vhotqs",
    "simclrcfaughead-1.0-22": "9k6krmms",
    "simclrcfhead-1.0-22": "yc7z4b0b",
    "simclrhead-1.0-22": "f8ph0afl",
    "simclrcfaughead-0.05-11": "f9rww309",
    "simclrcfhead-0.05-11": "hd022e7m",
    "simclrhead-0.05-11": "bniozjof",
    "simclrcfhead-0.1-11": "mzuzuhoc",
    "simclrcfaughead-0.1-11": "b9ronasb",
    "simclrhead-0.1-11": "ej5dcv7c",
    "simclrcfaughead-0.25-11": "9p621l6t",
    "simclrcfhead-0.25-11": "ox1quvu6",
    "simclrhead-0.25-11": "ek2uddv2",
    "simclrcfhead-0.5-11": "qc7qilc7",
    "simclrcfaughead-0.5-11": "w0s6otk8",
    "simclrhead-0.5-11": "8oedepno",
    "simclrcfhead-1.0-11": "yn1ip3i6",
    "simclrcfaughead-1.0-11": "5ut4ads9",
    "simclrhead-1.0-11": "5ysv6mvj",
}

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=["experiment=base_density", "data=vindr", "data.cache=False"],
    )
    data_module = VinDrDataModule(config=cfg)
test_loader = data_module.test_dataloader()

df = pd.read_csv("../outputs/vindr2.csv")
for run_name, run_id in model_dict.items():
    already_in_df = run_name in df.run_name.values
    if run_id != "" and not already_in_df:
        print(run_name)
        model_to_evaluate = f"../outputs/run_{run_id}/best.ckpt"
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
        df.to_csv("../outputs/vindr2.csv", index=False)
