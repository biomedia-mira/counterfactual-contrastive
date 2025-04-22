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
    "simclrcfhead-0.05-33": "cbfkz1sl",
    "simclrhead-0.05-33": "x9l2xljv",
    "simclrcfhead-0.1-33": "3vz13efk",
    "simclrhead-0.1-33": "bjpjjsf8",
    "simclrcf-0.1-33": "dvb0d52s",
    "simclrcfhead-0.25-33": "2q9vqnnf",
    "simclrhead-0.25-33": "t8826i37",
    "simclrcf-0.25-33": "018sy9al",
    "simclrcfhead-0.5-33": "ocuvdahw",
    "simclrhead-0.5-33": "huze07kx",
    "simclrcf-0.5-33": "llmb6gq8",
    "simclrhead-1.0-33": "yywbsyi5",
    "simclrcf-1.0-33": "l2jlzbzt",
    "simclr-0.1-33": "jyn1jzku",
    "simclr-0.25-33": "hgcfhz1t",
    "simclrcfhead-1.0-33": "in9dzbn3",
    "simclrhead-0.05-22": "lnj3yydc",
    "simclrhead-0.1-22": "jugy7nw2",
    "simclr-0.5-33": "g8aq5em5",
    "simclrcf-0.05-22": "kp8u3s2m",
    "simclrhead-0.25-22": "n6osuheo",
    "simclrcfhead-0.05-22": "nvelflfg",
    "simclrhead-0.5-22": "4z5nyl5u",
    "simclrcfhead-0.1-22": "mqv4rv7l",
    "simclrhead-1.0-22": "iqrs8bpw",
    "simclrcf-0.1-22": "np0nb3fk",
    "simclrhead-0.05-11": "pejx08s9",
    "simclrhead-0.1-11": "jbun6pv1",
    "simclrcf-0.25-22": "y5d9twi9",
    "simclrcfhead-0.25-22": "2w8dd9sl",
    "simclrhead-0.25-11": "nzh0uxqw",
    "simclrcf-0.5-22": "8222inz6",
    "simclr-1.0-33": "g57vd7lg",
    "simclrhead-0.5-11": "8td74xhh",
    "simclr-0.05-22": "jdh3xf7t",
    "simclrcfhead-0.5-22": "tmxnvhml",
    "simclrhead-1.0-11": "by1h6jp3",
    "simclrcf-1.0-22": "5dq2a3wx",
    "simclr-0.1-22": "8tja5u0s",
    "simclrcf-0.05-11": "u8mkxtk2",
    "simclrcf-0.1-11": "3nca3s09",
    "simclrcfhead-1.0-22": "fr6nh3px",
    "simclrcfhead-0.05-11": "4rbg4h1d",
    "simclrcf-0.25-11": "8tp6vnjr",
    "simclr-0.25-22": "kx3owiy4",
    "simclrcfhead-0.1-11": "sas2eztu",
    "simclrcf-0.5-11": "2agax8fw",
    "simclrcfhead-0.25-11": "6an1yp1y",
    "simclr-0.5-22": "vum2olh1",
    "simclrcf-1.0-11": "mtcukk2x",
    "simclrcfhead-0.5-11": "2cfjko4w",
    "simclr-0.05-11": "lpeu81ya",
    "simclr-1.0-11": "fd4r9de5",
    "simclr-0.1-11": "7y72cm3b",
    "simclrcfhead-1.0-11": "tzwrpyqf",
    "simclr-1.0-22": "kcloy93c",
    "simclr-0.05-33": "m4i603ks",
    "simclr-0.25-11": "xala1ohh",
    "simclrcffinehead-0.01-11": "2p9wxgk8",
    "simclrcffinehead-0.05-11": "q40louov",
    "simclrcffinehead-0.1-11": "lap3fc4m",
    "simclrcffinehead-0.25-11": "frqq0a3x",
    "simclrcffinehead-1.0-11": "61bpeh0b",
    "simclrcffinehead-0.01-22": "el29wm96",
    "simclrcffinehead-0.05-22": "6kxutiaf",
    "simclrcffinehead-0.1-22": "xja1cf9a",
    "simclrcffinehead-0.25-22": "pq3pdxg3",
    "simclrcffinehead-1.0-22": "4fmo5aav",
    "simclrcffinehead-0.01-33": "tkkop1en",
    "simclrcffinehead-0.05-33": "2nal6ufp",
    "simclrcffinehead-0.1-33": "yl1cfyem",
    "simclrcffinehead-0.25-33": "q5yirfwz",
    "simclrcffinehead-1.0-33": "ypwcc5f0",
    "simclrcffine-0.01-11": "9d6vlsg9",
    "simclrcffine-0.05-11": "n6g5spc9",
    "simclrcffine-0.1-11": "0hn769m4",
    "simclrcffine-0.25-11": "a79kgvpd",
    "simclrcffine-1.0-11": "w4uow5tw",
    "simclrcffine-0.01-22": "q7uekstb",
    "simclrcffine-0.05-22": "ggtbsugi",
    "simclrcffine-0.1-22": "wfwxt29y",
    "simclrcffine-0.25-22": "g1olxkd7",
    "simclrcffine-1.0-22": "ygzfqnrz",
    "simclrcffine-0.01-33": "ta1wvdmy",
    "simclrcffine-0.05-33": "czsnebkj",
    "simclrcffine-0.1-33": "sg6dipfo",
    "simclrcffine-0.25-33": "5aj4sp9v",
    "simclrcffine-1.0-33": "mis91jcm",
    "simclrcfbadhead-0.01-11": "o3xqtgk7",
    "simclrcfbadhead-0.05-11": "i1lvx3hf",
    "simclrcfbadhead-0.1-11": "f99qmldw",
    "simclrcfbadhead-0.25-11": "3ze02lzd",
    "simclrcfbadhead-1.0-11": "ejuhow0c",
    "simclrcfbadhead-0.01-22": "o7ykuvto",
    "simclrcfbadhead-0.05-22": "yxjkyv9r",
    "simclrcfbadhead-0.1-22": "22w52118",
    "simclrcfbadhead-0.25-22": "l6gb3ll9",
    "simclrcfbadhead-1.0-22": "bdi7fchx",
    "simclrcfbadhead-0.01-33": "jzlkf2hp",
    "simclrcfbadhead-0.05-33": "jp2skr28",
    "simclrcfbadhead-0.1-33": "m3mca7r3",
    "simclrcfbadhead-0.25-33": "q6ettexy",
    "simclrcfbadhead-1.0-33": "ou005dxh",
    "simclrcfbad-0.01-11": "h8ti0j1k",
    "simclrcfbad-0.05-11": "bpsx5ksy",
    "simclrcfbad-0.1-11": "w7tq2zwg",
    "simclrcfbad-0.25-11": "x98b93ck",
    "simclrcfbad-1.0-11": "610teohh",
    "simclrcfbad-0.01-22": "276nqcab",
    "simclrcfbad-0.05-22": "83h5whpo",
    "simclrcfbad-0.1-22": "ijk2g3oq",
    "simclrcfbad-0.25-22": "pyzggrav",
    "simclrcfbad-1.0-22": "9jpoaapx",
    "simclrcfbad-0.01-33": "xlb8ezp3",
    "simclrcfbad-0.05-33": "rksgsbi5",
    "simclrcfbad-0.1-33": "ev2u6mbe",
    "simclrcfbad-0.25-33": "wxt5s9ev",
    "simclrcfbad-1.0-33": "eoaijj8r",
}


with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=["experiment=base_density", "data=embed_ood", "data.cache=False"],
    )
    print(cfg)
    data_module = EmbedOODDataModule(config=cfg)
test_loader = data_module.test_dataloader()


df = pd.read_csv("../outputs/embed_results_ood_ablation.csv")
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

        targets = inference_results["targets"]
        preds = np.argmax(inference_results["confs"], 1)
        confs = inference_results["confs"]
        res = {}
        res["N_test"] = [targets.shape[0]]
        res["ROC"] = [roc_auc_score(targets, confs, multi_class="ovr")]
        res["Model Name"] = [rev_model_map[5]]
        res["run_name"] = run_name
        print(res)
        df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)

        df.to_csv(
            "../outputs/embed_results_ood_ablation.csv",
            index=False,
        )
