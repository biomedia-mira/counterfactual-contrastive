from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from classification.classification_module import ClassificationModule
from hydra import compose, initialize
from data_handling.mammo import EmbedDataModule, modelname_map
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
    "simclrcf-0.01-11": "um1y3iln",
    "simclrcf-0.05-22": "stoqfdld",
    "simclr-0.05-22": "pf0uqv2e",
    "simclrcf-0.1-22": "jk0xk1yq",
    "simclr-0.05-11": "j7qg2zcf",
    "simclrcf-0.05-11": "ju9xfu6z",
    "simclr-0.1-22": "mh4exarz",
    "simclr-0.1-11": "3a1wft8u",
    "simclrcf-0.25-22": "w1muwi2v",
    "simclrcf-0.1-11": "2c8x9gr9",
    "simclr-0.25-22": "io9ukhfm",
    "simclrcf-0.05-33": "kkcbrb3i",
    "simclr-0.25-11": "tkr8tw5e",
    "simclrcf-0.25-11": "e8w1vxmn",
    "simclrcf-0.1-33": "bzyqmycb",
    "simclrcf-1.0-22": "0cf2wpg6",
    "simclr-1.0-22": "qz7qx7u3",
    "simclrcf-1.0-11": "012vx3f7",
    "simclr-1.0-11": "md4ymp1g",
    "simclrcf-0.25-33": "3k1lu1oy",
    "simclr-0.01-33": "hsyrkrpi",
    "simclr-0.01-22": "nsr3ixey",
    "simclr-0.01-11": "cpqdq4gs",
    "simclrcf-0.01-22": "qxvdehkn",
    "simclrcf-0.01-33": "nrz507vz",
    "simclrcf-1.0-33": "lhaovv2s",
    "simclr-0.05-33": "eissogv0",
    "simclr-0.1-33": "1wwpkejb",
    "simclr-0.25-33": "gx6u8o0b",
    "simclr-1.0-33": "d1zvpt88",
    "simclrcfhead-0.01-33": "z2zjgali",
    "simclrhead-0.05-33": "o81utb35",
    "simclrcfhead-0.05-33": "fz8292xk",
    "simclrhead-0.1-33": "91al13pn",
    "simclrcfhead-0.1-33": "hhd6tlvt",
    "simclrhead-0.25-33": "lmnz6cqa",
    "simclrcfhead-0.25-33": "dlny28wi",
    "simclrhead-0.01-33": "ydmewf9b",
    "simclrhead-1.0-33": "pbgsgaxc",
    "simclrcfhead-0.01-22": "as4948zr",
    "simclrhead-0.05-22": "51zpigal",
    "simclrcfhead-0.05-22": "zuc0cjjx",
    "simclrhead-0.1-22": "mt2wkl00",
    "simclrcfhead-0.1-22": "giznbw9f",
    "simclrhead-0.25-22": "3rxog9d2",
    "simclrcfhead-0.25-22": "7qvf5uo5",
    "simclrhead-0.01-22": "rxk3y37s",
    "simclrcfhead-1.0-22": "dwb7zktd",
    "simclrhead-1.0-22": "e62b8afn",
    "simclrhead-0.05-11": "f4dvy97m",
    "simclrcfhead-0.05-11": "2pe5jd0v",
    "simclrhead-0.1-11": "hax3h4uo",
    "simclrcfhead-0.1-11": "udbj16e9",
    "simclrhead-0.25-11": "vb1al6q2",
    "simclrcfhead-0.25-11": "eclgsmot",
    "simclrhead-0.01-11": "xqja0tjy",
    "simclrcfhead-1.0-11": "j6jbom7s",
    "simclrhead-1.0-11": "ui1zuhsu",
    "simclrcffinehead-0.01-11": "o7yhdam4",
    "simclrcffinehead-0.05-11": "y1f77dei",
    "simclrcffinehead-0.25-11": "akmzw596",
    "simclrcffinehead-1.0-11": "8yv9arq3",
    "simclrcffinehead-0.01-22": "iv3smjhd",
    "simclrcffinehead-0.05-22": "k6wtfg1i",
    "simclrcffinehead-0.1-22": "9pkj8ply",
    "simclrcffinehead-0.25-22": "p821k1lx",
    "simclrcffinehead-1.0-22": "2080piaj",
    "simclrcffinehead-0.01-33": "7qiy9wrl",
    "simclrcffinehead-0.05-33": "t03puj43",
    "simclrcffinehead-0.1-11": "5191wwfh",
    "simclrcffinehead-0.25-33": "xxuqq23t",
    "simclrcffinehead-0.1-33": "rhg3ih53",
    "simclrcffinehead-1.0-33": "ltfm098z",
    "simclrcffine-0.01-11": "8xblvv7l",
    "simclrcffine-0.05-11": "z5hhvwr7",
    "simclrcffine-0.1-11": "y4dvbld5",
    "simclrcffine-0.25-11": "a8pgsrhq",
    "simclrcffine-1.0-11": "qcravxcx",
    "simclrcffine-0.01-22": "4809eb0x",
    "simclrcffine-0.1-22": "c0rwr0ps",
    "simclrcffine-0.25-22": "f3s7uf6d",
    "simclrcffine-1.0-22": "wj6gbrng",
    "simclrcffine-0.01-33": "kjyleedx",
    "simclrcffine-0.05-33": "wqc1s07y",
    "simclrcffine-0.1-33": "resi88eb",
    "simclrcffine-0.25-33": "t1f7bwxf",
    "simclrcffine-1.0-33": "43av73mq",
    "simclrcfbadhead-0.01-11": "l8vrh7om",
    "simclrcfbadhead-0.05-11": "xlzp3h5b",
    "simclrcfbadhead-0.1-11": "o4t7d0y7",
    "simclrcfbadhead-0.25-11": "bu5j4d5r",
    "simclrcfbadhead-1.0-11": "skdciscy",
    "simclrcfbadhead-0.01-22": "fiixu455",
    "simclrcfbadhead-0.1-22": "ffle65u1",
    "simclrcfbadhead-0.25-22": "7k2nu5u0",
    "simclrcfbadhead-1.0-22": "t4hsvcvo",
    "simclrcfbadhead-0.01-33": "ajn5tfhw",
    "simclrcfbadhead-0.05-33": "ofwd9w2o",
    "simclrcfbadhead-0.1-33": "usi4zzrz",
    "simclrcfbadhead-0.25-33": "bnnq7zmr",
    "simclrcfbadhead-1.0-33": "kae9nvyr",
    "simclrcfbad-0.01-11": "zslupbl2",
    "simclrcfbad-0.05-11": "119rb5xn",
    "simclrcfbad-0.1-11": "05nwodvq",
    "simclrcfbad-0.25-11": "flb5bfzf",
    "simclrcfbad-1.0-11": "od6koxbo",
    "simclrcfbad-0.01-22": "i2n1jxo1",
    "simclrcfbad-0.05-22": "aawkgoek",
    "simclrcfbad-0.1-22": "q3hq47tr",
    "simclrcfbad-0.25-22": "m4mj7g00",
    "simclrcfbad-1.0-22": "umetkfep",
    "simclrcfbad-0.01-33": "rw1ut4qy",
    "simclrcfbad-0.05-33": "dd4siy80",
    "simclrcfbad-0.1-33": "xpdwtnbh",
    "simclrcfbad-0.25-33": "pwm2dt18",
    "simclrcfbad-1.0-33": "vzgzitk7",
}

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=["experiment=base_density", "data.cache=False"],
    )
    print(cfg)
    data_module = EmbedDataModule(config=cfg, parents=["scanner"])
test_loader = data_module.test_dataloader()

df = pd.read_csv(f"../outputs/classification_{cfg.data.label}_results_ablation.csv")
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
            f"../outputs/classification_{cfg.data.label}_results_ablation.csv",
            index=False,
        )
