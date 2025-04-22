import os
import pandas as pd
from hydra import compose, initialize
from sklearn.metrics import roc_auc_score
from data_handling.xray import CheXpertDataModule
from classification.classification_module import ClassificationModule
from evaluation.helper_functions import run_inference

os.chdir("/vol/biomedic3/mb121/causal-contrastive/evaluation")

# Mapping from human readable run name to Weights&Biases run_id.

# Human readable name should be in format:
# for finetuning:
# {simclr/simclrcf/simclrcfaug}-{train_prop}-{seed}
# for linear probing
# {simclr/simclrcf/simclrcfaug}head-{train_prop}-{seed}

model_dict_normal: dict[str, str] = {
    # "dinocfhead-0.1-11": "s8y6rkkx",
    # "dinocfhead-0.1-22": "ucnyujha",
    # "dinocfhead-0.1-33": "cwz2lvcu",
    # "dinocfhead-0.25-22": "cnx7x2w1",
    # "dinocfhead-0.25-33": "schu7ga6",
    # "dinocfhead-0.25-11": "o2yrwwwy",
    # "dinocfhead-1.0-11": "9auoh7tn",
    # "dinocfhead-1.0-22": "lci2doov",
    # "dinocfhead-1.0-33": "0jnka5je",
    "dinohead-0.1-22": "ugoz1tki",
    "dinohead-0.1-33": "fgct8q3r",
    "dinohead-0.1-11": "b8dv7f0f",
    "dinohead-0.25-33": "jxdrgsz7",
    "dinohead-0.25-22": "30m2kvkk",
    "dinohead-0.25-11": "6a7jyw7u",
    "dinohead-1.0-33": "eydmd5ky",
    "dinohead-1.0-22": "pdcspxje",
    "dinohead-1.0-11": "9kppc8f6",
    # "dinohead-0.25-55": "l5lte6ze",
    # "dinocfhead-0.25-55": "93ncqw6w",
    # "dinocf-1.0-33": "yvr5k0uh",
    # "dinocf-1.0-22": "96sywryt",
    # "dinocf-0.25-33": "y25i4ybs",
    # "dinocf-0.1-33": "wipr0twi",
    # "dinocf-0.25-22": "i3av1q8f",
    # "dinocf-0.1-22": "2u8koexo",
    # "dinocf-1.0-11": "21txiyfc",
    # "dinocf-0.25-11": "dsfp7cld",
    # "dinocf-0.1-11": "q2dhoia9",
    "dino-1.0-33": "njzdlp0s",
    "dino-1.0-22": "5md166ed",
    "dino-0.25-33": "eduxeday",
    "dino-0.1-33": "vq8nt0fm",
    "dino-0.25-22": "xdkrzzqd",
    "dino-1.0-11": "j61znfwv",
    "dino-0.1-22": "ntigwlpe",
    "dino-0.25-11": "3n32aq5y",
    "dino-0.1-11": "sumtvxtb",
    # 'dinocfaughead-1.0-33': 'oi29wrcp',
    # 'dinocfaughead-0.25-33': 'zfeyldyp',
    # 'dinocfaughead-0.1-33': 'z1me3aky',
    # 'dinocfaughead-0.25-22': 'srjhy0nl',
    # 'dinocfaughead-1.0-22': 'l4qgqsau',
    # 'dinocfaughead-0.1-11': '55uza3l6',
    # 'dinocfaughead-0.1-22': 'hsv5n7ee',
    # 'dinocfaughead-0.25-11': 'o5xl2fuc',
    # 'dinocfaughead-1.0-11': '6i9ovr4z',
    # 'dinocfaug-1.0-11': '71q211gv',
    # 'dinocfaug-1.0-22': 'ac0e1s9g',
    # 'dinocfaug-1.0-33': 'o95wtopw',
    # 'dinocfaug-0.25-33': '4fcnc3vv',
    # 'dinocfaug-0.1-11': 'ai74edot',
    # 'dinocfaug-0.25-22': 'vmdmwita',
    # 'dinocfaug-0.05-33': 'kfqnrt0g',
    # 'dinocfaug-0.05-22': 'shqa4gdt',
    # 'dinocfaug-0.1-22': 'om35rfxw',
    # 'dinocfaug-0.25-11': 'gcxgfmp2',
    # 'dinocfaug-0.1-11': 'ave9vk2z',
    # 'dinocfaug-0.05-11': 'gogzr69a',
    # 'dinocfaug-0.25-33': '4fcnc3vv',
    # 'dinocfaug-0.1-33': 'ai74edot',
    # 'dinocfaug-0.25-22': 'vmdmwita',
    "dinocfhead-0.1-11": "ihi30y9a",
    "dinocfhead-1.0-11": "1qg8xusb",
    "dinocfhead-0.1-33": "3uvzowyq",
    "dinocfhead-1.0-22": "qbrjlivm",
    "dinocfhead-0.1-22": "w0lc0hip",
    "dinocfhead-1.0-33": "q9rjkc1i",
    "dinocfhead-0.25-11": "0el98eux",
    "dinocfhead-0.25-33": "wkqw8unq",
    "dinocfhead-0.25-22": "e1bm2o4m",
    "dinocf-1.0-33": "rb5c9db7",
    "dinocf-1.0-11": "3qyv23dt",
    "dinocf-1.0-22": "t11s0ugp",
    "dinocf-0.25-33": "pogdsxbi",
    "dinocf-0.25-22": "9jct4bci",
    "dinocf-0.1-22": "ln9ygn4z",
    "dinocf-0.25-11": "cubpedy2",
    "dinocf-0.1-11": "4yzj0tc5",
    "dinocf-0.1-33": "2dp2a6is",
    # 'dinocf-1.0-55': 'uv5tqo4o',
    # 'dinocf-1.0-44': 'atcztz7r',
    # 'dinocfaug-1.0-44': 'luwujgt4',
    # 'dinocfaug-1.0-55': '83cw3ffm',
    # 'dinocf-0.1-44': '7eqlfsnw',
    # 'dinocf-0.1-55': 'yw5gji65',
    # 'dinocfaug-0.1-44': 'rvwirm10',
    # 'dinocfaug-0.1-55': 'd20u1p13',
    "dinocfaug2-0.1-11": "h6oavktm",
    "dinocfaug2-0.1-22": "yift5k0i",
    "dinocfaug2-0.25-11": "io37roro",
    "dinocfaug2-0.25-22": "xpiqt12b",
    "dinocfaug2-1.0-11": "4u67e42d",
    "dinocfaug2-1.0-22": "3nq3vmwq",
    "dinocfaug2-1.0-33": "qxpimbsk",
    "dinocfaug2head-1.0-22": "pfnmccns",
    "dinocfaug2head-0.1-22": "0bn2coll",
    "dinocfaug2head-0.25-22": "otbwkxu4",
    "dinocfaug2-0.1-33": "cixk0mzd",
    "dinocfaug2-0.25-33": "yd17dmq3",
    "dinocfaug2head-0.25-11": "70wz9n2t",
    "dinocfaug2head-0.1-11": "ff4xywv2",
    "dinocfaug2head-0.1-33": "5npf83am",
    "dinocfaug2head-0.25-33": "t3ni32cs",
    "dinocfaug2head-1.0-11": "48d6jxas",
    "dinocfaug2head-1.0-33": "5p9f52xe",
    # 'dinocfhead-1.0-55': '5efo7zov',
    # 'dinocfhead-1.0-44': '4eq4thdr',
    # 'dinocfhead-0.25-55': '9e4b5v29',
    # 'dinocfhead-0.25-44': 'jxpmajme',
    # 'dinocfhead-1.0-66': 'd1pi55en',
    # 'dinocfhead-0.25-66': 'zykd7ncm',
    # 'dinocfhead-1.0-77': 'fgc10746',
    # 'dinocfhead-0.25-77': 'b9yecgmn'
    # 'dinocf3head-1.0-11': 'ho4y2ypb',
    # 'dinocf3head-1.0-22': '1dx6fnb1',
    # 'dinocf3head-1.0-33': '5085eg0e',
    # 'dinocf3head-0.1-11': 'avzkh6vo',
    # 'dinocf3head-0.1-33': 'lq5kp01l',
    # 'dinocf3head-0.1-22': 'apazezap',
    # 'dinocf3head-0.1-11': 'chpjanzw',
    # 'dinocf3head-0.1-33': 'poj1cuni',
}

filename = "../outputs/dino_chexpert2.csv"

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=[
            "experiment=base_padchestpneumo",
            "data=chexpert",
            "data.label=Pneumonia",
            "data.cache=True",
        ],
    )
    data_module = CheXpertDataModule(config=cfg)

test_dataloader = data_module.test_dataloader()

df = pd.read_csv(filename)
for run_name, run_id in model_dict_normal.items():
    already_in_df = run_name in df.run_name.values
    if run_id != "" and not already_in_df:
        print(run_name)
        model_to_evaluate = f"../outputs2/run_{run_id}/best.ckpt"
        classification_model = ClassificationModule.load_from_checkpoint(
            model_to_evaluate, map_location="cuda:0", strict=False
        ).model.eval()
        classification_model.cuda()
        inference_results = run_inference(test_dataloader, classification_model)
        print("\nEvaluating CheXpert")
        res = {}
        res["N_test"] = [inference_results["targets"].shape[0]]
        res["Scanner"] = ["CheXpert"]
        res["run_name"] = run_name
        res["ROC"] = [
            roc_auc_score(
                inference_results["targets"], inference_results["confs"][:, 1]
            )
        ]
        print(res)
        df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)
        df.to_csv(filename, index=False)
