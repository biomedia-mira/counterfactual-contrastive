from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from classification.classification_module import ClassificationModule
from hydra import compose, initialize
from data_handling.mammo import modelname_map, EmbedOODDataModule
from evaluation.helper_functions import run_inference


import os

os.chdir("/vol/biomedic3/mb121/causal-contrastive/evaluation")

rev_model_map = {v: k for k, v in modelname_map.items()}

# Mapping from human readable run name to Weights&Biases run_id.

# Human readable name should be in format:
# for finetuning:
# {simclr/simclrcf/simclrcfaug}-{train_prop}-{seed}
# for linear probing
# {simclr/simclrcf/simclrcfaug}head-{train_prop}-{seed}

model_dict = {
    "simclrcfaughead-0.1-33": "sfjtep0q",
    "simclrcfhead-0.05-33": "cbfkz1sl",
    "simclrhead-0.05-33": "x9l2xljv",
    "simclrcfaughead-0.25-33": "x3ygrgw8",
    "simclrcfhead-0.1-33": "3vz13efk",
    "simclrhead-0.1-33": "bjpjjsf8",
    "simclrcfaug-0.1-33": "7akh9trb",
    "simclrcf-0.1-33": "dvb0d52s",
    "simclrcfhead-0.25-33": "2q9vqnnf",
    "simclrhead-0.25-33": "t8826i37",
    "simclrcfaughead-0.5-33": "a2t86qgm",
    "simclrcfaug-0.25-33": "pnucxhp6",
    "simclrcf-0.25-33": "018sy9al",
    "simclrcfhead-0.5-33": "ocuvdahw",
    "simclrhead-0.5-33": "huze07kx",
    "simclrcfaughead-1.0-33": "fq5i6et5",
    "simclrcfaug-0.5-33": "3r963q78",
    "simclrcf-0.5-33": "llmb6gq8",
    "simclrhead-1.0-33": "yywbsyi5",
    "simclrcf-1.0-33": "l2jlzbzt",
    "simclr-0.1-33": "jyn1jzku",
    "simclr-0.25-33": "hgcfhz1t",
    "simclrcfhead-1.0-33": "in9dzbn3",
    "simclrhead-0.05-22": "lnj3yydc",
    "simclrcfaug-1.0-33": "1zwvsq0s",
    "simclrcfaughead-0.05-22": "ymguomdu",
    "simclrhead-0.1-22": "jugy7nw2",
    "simclr-0.5-33": "g8aq5em5",
    "simclrcfaug-0.05-22": "1cm0ul2o",
    "simclrcfaughead-0.1-22": "5ubg393w",
    "simclrcf-0.05-22": "kp8u3s2m",
    "simclrhead-0.25-22": "n6osuheo",
    "simclrcfhead-0.05-22": "nvelflfg",
    "simclrcfaug-0.1-22": "fiixmtqq",
    "simclrhead-0.5-22": "4z5nyl5u",
    "simclrcfhead-0.1-22": "mqv4rv7l",
    "simclrcfaughead-0.25-22": "1st31efx",
    "simclrhead-1.0-22": "iqrs8bpw",
    "simclrcfaug-0.25-22": "5gk0bf4h",
    "simclrcf-0.1-22": "np0nb3fk",
    "simclrhead-0.05-11": "pejx08s9",
    "simclrhead-0.1-11": "jbun6pv1",
    "simclrcfaug-0.5-22": "lymnrsi4",
    "simclrcf-0.25-22": "y5d9twi9",
    "simclrcfhead-0.25-22": "2w8dd9sl",
    "simclrhead-0.25-11": "nzh0uxqw",
    "simclrcfaughead-0.5-22": "ock5l9ii",
    "simclrcfaug-1.0-22": "x0zkov3t",
    "simclrcf-0.5-22": "8222inz6",
    "simclr-1.0-33": "g57vd7lg",
    "simclrhead-0.5-11": "8td74xhh",
    "simclr-0.05-22": "jdh3xf7t",
    "simclrcfhead-0.5-22": "tmxnvhml",
    "simclrcfaug-0.05-11": "tdjrer5o",
    "simclrhead-1.0-11": "by1h6jp3",
    "simclrcf-1.0-22": "5dq2a3wx",
    "simclrcfaughead-1.0-22": "3jhxlv5v",
    "simclr-0.1-22": "8tja5u0s",
    "simclrcfaug-0.1-11": "c6652p4m",
    "simclrcf-0.05-11": "u8mkxtk2",
    "simclrcf-0.1-11": "3nca3s09",
    "simclrcfaughead-0.05-11": "fdvbf3yj",
    "simclrcfhead-1.0-22": "fr6nh3px",
    "simclrcfaug-0.5-11": "d5usym29",
    "simclrcfhead-0.05-11": "4rbg4h1d",
    "simclrcf-0.25-11": "8tp6vnjr",
    "simclr-0.25-22": "kx3owiy4",
    "simclrcfaughead-0.1-11": "apwbewir",
    "simclrcfhead-0.1-11": "sas2eztu",
    "simclrcfaughead-0.25-11": "cjxptv82",
    "simclrcfaug-1.0-11": "952t5sbz",
    "simclrcf-0.5-11": "2agax8fw",
    "simclrcfhead-0.25-11": "6an1yp1y",
    "simclrcfaughead-0.5-11": "6euv46vl",
    "simclr-0.5-22": "vum2olh1",
    "simclrcf-1.0-11": "mtcukk2x",
    "supervised-0.05-33": "01doh1to",
    "simclrcfhead-0.5-11": "2cfjko4w",
    "supervised-0.1-33": "vy8qveyt",
    "simclr-0.05-11": "lpeu81ya",
    "supervised-0.25-33": "j46z0201",
    "simclr-1.0-11": "fd4r9de5",
    "supervised-1.0-33": "trj4rjoh",
    "simclr-0.1-11": "7y72cm3b",
    "simclrcfaughead-1.0-11": "zcefxazw",
    "simclrcfhead-1.0-11": "tzwrpyqf",
    "simclr-1.0-22": "kcloy93c",
    "supervised-0.05-22": "ckyww6k4",
    "simclr-0.05-33": "m4i603ks",
    "simclr-0.25-11": "xala1ohh",
    "supervised-0.1-22": "k4utvrmu",
    "supervised-1.0-11": "a7brvlk5",
    "supervised-0.25-11": "yhvlr7ww",
    "supervised-0.1-11": "ma2ss7jh",
    "supervised-0.05-11": "8ttg7aty",
    "supervised-1.0-22": "dixs9enj",
}


with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=["experiment=base_density", "data=embed_ood", "data.cache=False"],
    )
    print(cfg)
    data_module = EmbedOODDataModule(config=cfg)
test_loader = data_module.test_dataloader()


df = pd.read_csv(f"../outputs/classification_{cfg.data.label}_results_ood.csv")
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
        scanners = np.argmax(inference_results["scanners"], 1)

        for i in np.unique(scanners):
            print(f"\nEvaluating scanner {i}")
            sc_idx = np.where(scanners == i)
            targets = inference_results["targets"][sc_idx]
            preds = np.argmax(inference_results["confs"], 1)[sc_idx]
            confs = inference_results["confs"][sc_idx]
            res = {}
            res["N_test"] = [targets.shape[0]]
            res["ROC"] = [roc_auc_score(targets, confs, multi_class="ovr")]
            res["Model Name"] = [rev_model_map[i]]
            res["run_name"] = run_name
            print(res)
            df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)

        df.to_csv(
            f"../outputs/classification_{cfg.data.label}_results_ood.csv",
            index=False,
        )
