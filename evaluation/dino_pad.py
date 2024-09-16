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
    "dinohead-0.05-11": "h3ve6rxq",
    "dinohead-0.05-33": "gvsnx7zk",
    "dinohead-0.05-22": "awmnt776",
    "dinohead-0.1-22": "hdj5vwiw",
    "dinohead-0.1-11": "wsckm688",
    "dinohead-0.1-33": "pjvvxjek",
    "dinohead-0.25-22": "yn6fbkro",
    "dinohead-0.25-11": "0lhnka4p",
    "dinohead-0.25-33": "jnxocd5x",
    "dinohead-1.0-22": "1t4u5xlp",
    "dinohead-1.0-33": "jsmfit3g",
    "dinohead-1.0-11": "cixwfhrv",
    # "dinocfhead-0.05-11": "ef8n7rta",
    # "dinocfhead-0.05-22": "tfd0movu",
    # "dinocfhead-0.05-33": "utjuzxh4",
    # "dinocfhead-0.1-11": "a4hxmpvm",
    # "dinocfhead-0.1-22": "is2p1ku2",
    # "dinocfhead-0.1-33": "xvycinnc",
    # "dinocfhead-0.25-11": "z018kskq",
    # "dinocfhead-0.25-22": "kgzscynr",
    # "dinocfhead-0.25-33": "sb1f5xze",
    # "dinocfhead-1.0-22": "pn7prwew",
    # "dinocfhead-1.0-11": "edczfxg3",
    # "dinocf-1.0-33": "g1s1yrmm",
    # "dinocf-1.0-22": "eoupmesc",
    # "dinocf-1.0-11": "k248hzc5",
    # "dinocf-0.25-33": "27rfpqen",
    # "dinocf-0.25-22": "d3r4mehb",
    # "dinocf-0.25-11": "yl7fwf2d",
    # "dinocf-0.1-33": "vh8xt4si",
    # "dinocf-0.1-22": "3hcpkqjl",
    # "dinocf-0.1-11": "cjxhmqqz",
    # "dinocf-0.05-33": "bib2i6mw",
    # "dinocf-0.05-11": "bdlb5qz0",
    # "dinocf-0.05-22": "7tidwayt",
    "dino-1.0-33": "yi9husbb",
    "dino-1.0-22": "l0psdilz",
    "dino-1.0-11": "z0e2irla",
    "dino-0.25-33": "4pgz4ttt",
    "dino-0.25-22": "87mw0v28",
    "dino-0.25-11": "g5lgl8ww",
    "dino-0.1-33": "nxqew3sq",
    "dino-0.1-22": "brczzz3q",
    "dino-0.1-11": "nb9qlblx",
    "dino-0.05-11": "0r7b7m08",
    "dino-0.05-33": "ekivutz7",
    "dino-0.05-22": "4mvm4um9",
    # 'dinocfaughead-0.05-11': 'xmynvzxq',
    # 'dinocfaug-0.1-11': 'd3zkew8l',
    # 'dinocfaug-0.05-11': 'aywzajky',
    # 'dinocfaug-0.05-22': 'wi4i7xj6',
    # 'dinocfaug-0.25-22': '4yoznon9',
    # 'dinocfaug-0.1-22': 'yd303pd0',
    # 'dinocfaug-1.0-11': 'j43qhqhx',
    # 'dinocfaug-1.0-22': '98x38udm',
    # 'dinocfaug-1.0-33': 'kw7qyf4l',
    # 'dinocfaug-0.25-33': 'd3wapavu',
    # 'dinocfaug-0.1-33': 'oo7y8hu1',
    # 'dinocfaug-0.05-33': 'kj0uirnh',
    # 'dinocfaug-0.25-11': 'f19dc7dr',
    # 'dinocfaughead-0.1-11': '4g8qplpj',
    # 'dinocfaughead-0.25-11': '5tkscbtk',
    # 'dinocfaughead-0.1-22': 'kcmod7w3',
    # 'dinocfaughead-0.05-22': 'bl9gbh24',
    # 'dinocfaughead-0.1-33': 'irkvceno',
    # 'dinocfaughead-0.25-22': 'fvy9314t',
    # 'dinocfaughead-0.05-33': 'mks19v2s',
    # 'dinocfaughead-0.25-33': '0lq13tj5',
    # 'dinocfaughead-1.0-22': '2ag9e5nx',
    # 'dinocfaughead-1.0-11': 'xj15t7i4',
    # 'dinocfaughead-1.0-33': 'g6ruzisy',
    "dinocfaug2head-1.0-11": "7st8mo0v",
    "dinocfaug2head-1.0-33": "nikl980l",
    "dinocfaug2head-1.0-22": "kjctouvs",
    "dinocfaug2head-0.1-33": "rffpq2wx",
    "dinocfaug2-1.0-11": "226o2k5i",
    "dinocfaug2-1.0-22": "40u7dj5x",
    "dinocfaug2-1.0-33": "rd2eb23l",
    "dinocfaug2head-0.25-33": "3e96g4b1",
    "dinocfaug2head-0.1-22": "wm2gv4yd",
    "dinocfaug2head-0.25-11": "w0l12n1v",
    "dinocfaug2head-0.25-22": "hf86clcr",
    "dinocfaug2head-0.1-11": "fcrqj4v1",
    "dinocfaug2-0.1-33": "r9azkcn7",
    "dinocfaug2-0.25-33": "y5sxsi1g",
    "dinocfaug2-0.25-22": "12fqxs9a",
    "dinocfaug2-0.25-11": "wru1kyrc",
    "dinocfaug2-0.1-11": "ckwr0jux",
    "dinocfaug2-0.1-22": "gczixmjb",
    "dinocf2-0.25-11": "xt1mhgm8",
    "dinocf2-0.1-11": "6sg15idj",
    "dinocf2-0.25-22": "v7mbsny7",
    "dinocf2-0.1-33": "bl003zob",
    "dinocf2-0.1-22": "3kwin8h1",
    "dinocf2-0.25-33": "l1p8jlt1",
    "dinocf2-1.0-22": "bbg1drf7",
    "dinocf2-1.0-11": "y1l186be",
    "dinocf2-1.0-33": "rpik199r",
    "dinocf2head-0.1-11": "hlxqh36o",
    "dinocf2head-0.25-22": "4ys83obt",
    "dinocf2head-0.1-33": "mx2se5wi",
    "dinocf2head-0.1-22": "jl8cg7ur",
    "dinocf2head-0.25-11": "nbmkrxj1",
    "dinocf2head-1.0-33": "ikqhbrvh",
    "dinocf2head-1.0-22": "6q3bddw0",
    "dinocf2head-0.25-33": "i18c3pot",
    "dinocf2head-1.0-11": "xf2hcnft",
    "dinocf2head-0.05-33": "e557a0yo",
    "dinocf2head-0.05-11": "936yue2y",
    "dinocf2head-0.05-22": "0e056r6j",
    "dinocf2-0.05-33": "18z0wm7c",
    "dinocf2-0.05-11": "xpxl2yfl",
    "dinocf2-0.05-22": "szvwkkmj",
    "dinocfaug2head-0.05-11": "zxchc7n7",
    "dinocfaug2head-0.05-22": "30l0ae95",
    "dinocfaug2head-0.05-33": "ynupqkxc",
    "dinocfaug2-0.05-33": "5zsc5kpi",
    "dinocfaug2-0.05-11": "kdpuixtl",
    "dinocfaug2-0.05-22": "g6igowri",
}


with initialize(version_base=None, config_path="../configs"):
    cfg: DictConfig = compose(
        config_name="config.yaml",
        overrides=["experiment=base_padchestpneumo.yaml", "data.cache=False"],
    )
    print(cfg)
    data_module = PadChestDataModule(config=cfg)

test_dataloader = data_module.test_dataloader()


filename = "../outputs/dino_padchest2.csv"


df = pd.read_csv(filename)
for run_name, run_id in model_dict_normal.items():
    already_in_df: bool = run_name in df.run_name.values
    if run_id != "" and not already_in_df:
        print(run_name)
        model_to_evaluate = f"../outputs2/run_{run_id}/best.ckpt"
        classification_model = ClassificationModule.load_from_checkpoint(
            model_to_evaluate, map_location="cuda:0", strict=True
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

        df.to_csv(filename, index=False)
