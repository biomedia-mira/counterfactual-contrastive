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
    "simclr-0.05-11": "9l7y5ut7",
    "simclrcf-0.05-11": "p95njer4",
    "simclrcf-0.1-11": "iw7tlt92",
    "simclr-0.1-11": "mof52mfw",
    "simclrcf-0.25-11": "ct68hj4b",
    "simclr-0.25-11": "2458sxig",
    "simclrcf-0.5-11": "keifisvj",
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
    "simclr-0.25-33": "cf5trd5v",
    "simclrcf-1.0-33": "exxhffot",
    "simclr-0.1-33": "knuou61x",
    "simclr-0.5-33": "qfbtg6f9",
    "simclr-1.0-33": "wh3lxhxz",
    "simclrcfhead-0.05-33": "mqj29vzy",
    "simclrhead-0.05-33": "beh79tru",
    "simclrcfhead-0.1-33": "kyof9klp",
    "simclrhead-0.1-33": "4j616g3p",
    "simclrcfhead-0.25-33": "a6pjsv40",
    "simclrhead-0.25-33": "hd5xx0wy",
    "simclrcfhead-0.5-33": "uuigz04q",
    "simclrhead-0.5-33": "j9z30v0i",
    "simclrcfhead-1.0-33": "eaybins7",
    "simclrhead-1.0-33": "4iopv07k",
    "simclrcfhead-0.05-22": "bhb010nf",
    "simclrhead-0.05-22": "2wfjbbyn",
    "simclrcfhead-0.1-22": "1cfux2ew",
    "simclrhead-0.1-22": "fxuauquy",
    "simclrcfhead-0.25-22": "uualms6e",
    "simclrhead-0.25-22": "o5b0g7qi",
    "simclrcfhead-0.5-22": "4ktb86ju",
    "simclrhead-0.5-22": "w2vhotqs",
    "simclrcfhead-1.0-22": "yc7z4b0b",
    "simclrhead-1.0-22": "f8ph0afl",
    "simclrcfhead-0.05-11": "hd022e7m",
    "simclrhead-0.05-11": "bniozjof",
    "simclrcfhead-0.1-11": "mzuzuhoc",
    "simclrhead-0.1-11": "ej5dcv7c",
    "simclrcfhead-0.25-11": "ox1quvu6",
    "simclrhead-0.25-11": "ek2uddv2",
    "simclrcfhead-0.5-11": "qc7qilc7",
    "simclrhead-0.5-11": "8oedepno",
    "simclrcfhead-1.0-11": "yn1ip3i6",
    "simclrhead-1.0-11": "5ysv6mvj",
    "simclrcffinehead-0.05-11": "m4cm0bmp",
    "simclrcffinehead-0.1-11": "9spmuwrm",
    "simclrcffinehead-0.25-11": "060ghtcs",
    "simclrcffinehead-1.0-11": "0mpag0jm",
    "simclrcffinehead-0.05-22": "hczz8sve",
    "simclrcffinehead-0.1-22": "kx38nojr",
    "simclrcffinehead-0.25-22": "k07jro9y",
    "simclrcffinehead-1.0-22": "54wqu4ck",
    "simclrcffinehead-0.05-33": "o2m4e3m0",
    "simclrcffinehead-0.1-33": "mgn2gpms",
    "simclrcffinehead-0.25-33": "bkg0ru8k",
    "simclrcffinehead-1.0-33": "jsxrpmwt",
    "simclrcffine-0.05-11": "yob5ucx7",
    "simclrcffine-0.1-11": "104ndv44",
    "simclrcffine-0.25-11": "7kdx66dq",
    "simclrcffine-1.0-11": "06sdk2df",
    "simclrcffine-0.05-22": "8mtmubc9",
    "simclrcffine-0.1-22": "4bok6phm",
    "simclrcffine-0.25-22": "u9lwc3fq",
    "simclrcffine-1.0-22": "hmb4jw9i",
    "simclrcffine-0.05-33": "lldhuvc3",
    "simclrcffine-0.1-33": "5h4m04p6",
    "simclrcffine-0.25-33": "rlah94jn",
    "simclrcffine-1.0-33": "kk6xm7ri",
    "simclrcfbadhead-0.05-11": "mb1bpzcs",
    "simclrcfbadhead-0.1-11": "9udor0x2",
    "simclrcfbadhead-0.25-11": "0zxwumpo",
    "simclrcfbadhead-1.0-11": "hjgfnky2",
    "simclrcfbadhead-0.05-22": "x0v27icc",
    "simclrcfbadhead-0.1-22": "qzw7ihcw",
    "simclrcfbadhead-0.25-22": "nhoc9qb9",
    "simclrcfbadhead-1.0-22": "p20l9gs3",
    "simclrcfbadhead-0.05-33": "1meoso4u",
    "simclrcfbadhead-0.1-33": "hrwk8qwl",
    "simclrcfbadhead-0.25-33": "owakhq01",
    "simclrcfbadhead-1.0-33": "acvjswou",
    "simclrcfbad-0.05-11": "ndh4i7a7",
    "simclrcfbad-0.1-11": "fzzvny9a",
    "simclrcfbad-0.25-11": "lky7tbjo",
    "simclrcfbad-1.0-11": "h81cgjn3",
    "simclrcfbad-0.05-22": "iegggrx1",
    "simclrcfbad-0.1-22": "7bfrs4fc",
    "simclrcfbad-0.25-22": "l8dgmhe2",
    "simclrcfbad-1.0-22": "0s5nvg7z",
    "simclrcfbad-0.05-33": "85s3tyfc",
    "simclrcfbad-0.1-33": "4kj07ejd",
    "simclrcfbad-0.25-33": "x3kfjfh0",
    "simclrcfbad-1.0-33": "4oud22tv",
}

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=["experiment=base_density", "data=vindr", "data.cache=False"],
    )
    data_module = VinDrDataModule(config=cfg)
test_loader = data_module.test_dataloader()

df = pd.read_csv("../outputs/vindr_ablation.csv")
for run_name, run_id in model_dict.items():
    already_in_df = run_name in df.run_name.values
    if run_id != "" and not already_in_df:
        print(run_name)
        model_to_evaluate = f"../outputs2/run_{run_id}/best.ckpt"
        classification_model = ClassificationModule.load_from_checkpoint(
            model_to_evaluate, map_location="cuda", config=cfg
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
        df.to_csv("../outputs/vindr_ablation.csv", index=False)
