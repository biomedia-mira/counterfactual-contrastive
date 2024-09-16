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
    "dinohead-0.01-11": "oahatuyu",
    "dinohead-0.01-22": "3m9m0obt",
    "dinohead-0.01-33": "i8jshuh4",
    "dinohead-0.05-11": "o8du1wbl",
    "dinohead-0.05-22": "qb8by6as",
    "dinohead-0.05-33": "10256yfl",
    "dinohead-0.1-11": "z7v7gyjm",
    "dinohead-0.1-22": "gdoy3mx2",
    "dinohead-0.1-33": "othw6zry",
    "dinohead-0.25-11": "lxsyvtyq",
    "dinohead-0.25-22": "xnj5sjoi",
    "dinohead-0.25-33": "vjchbh7a",
    "dinohead-1.0-11": "4tzsd1zx",
    "dinohead-1.0-33": "tg7joka1",
    "dinohead-1.0-22": "nsr22zsh",
    # "dinocfhead-0.01-33": "3nzk7vyx",
    # "dinocfhead-0.01-22": "yjwl1hne",
    # "dinocfhead-0.01-11": "2ura4dyw",
    # "dinocfhead-0.05-11": "4knaifm1",
    # "dinocfhead-0.05-22": "93cejfnv",
    # "dinocfhead-0.05-33": "ux2b07g1",
    # "dinocfhead-0.1-11": "8prc6hln",
    # "dinocfhead-0.1-22": "2jz4nn16",
    # "dinocfhead-0.1-33": "27l4ygl9",
    # "dinocfhead-0.25-11": "sznihnmh",
    # "dinocfhead-0.25-22": "egtg5x8w",
    # "dinocfhead-0.25-33": "lq4bbqac",
    # "dinocfhead-1.0-22": "2ac09o5d",
    # "dinocfhead-1.0-33": "gqun3yxw",
    # "dinocfhead-1.0-11": "vunr7kh6",
    "dino-0.01-11": "gkctgwbu",
    "dino-0.01-22": "bwqmlm5n",
    "dino-0.01-33": "w6cy02qm",
    "dino-0.05-22": "jpsvcxvk",
    "dino-0.05-33": "4eejhehl",
    "dino-0.05-11": "l2crzr10",
    "dino-0.1-22": "yhoqtlbg",
    "dino-0.1-33": "p3ad7eih",
    "dino-0.1-11": "tvuvpxr8",
    "dino-0.25-22": "xpj91ptv",
    "dino-0.25-11": "i7eds672",
    "dino-0.25-33": "sxbolewg",
    "dino-1.0-11": "njgpgkbm",
    "dino-1.0-33": "5plfejhr",
    "dino-1.0-22": "rf8dcfj3",
    # "dinocf-0.01-33": "jjm8dep2",
    # "dinocf-0.01-22": "6imqbhy5",
    # "dinocf-0.01-11": "4pr5vqlb",
    # "dinocf-0.1-33": "23wldcff",
    # "dinocf-0.1-22": "hizz5ivw",
    # "dinocf-0.1-11": "fms3npwm",
    # "dinocf-0.05-11": "at8bqsfv",
    # "dinocf-0.05-33": "i7nvmze9",
    # "dinocf-0.05-22": "1hk4pbgp",
    # "dinocf-1.0-33": "wkt22zwo",
    # "dinocf-1.0-11": "451iwst9",
    # "dinocf-1.0-22": "gv24sci9",
    # "dinocf-0.25-22": "m1a4g9g5",
    # "dinocf-0.25-11": "7jq58l9d",
    # "dinocf-0.25-33": "phucy2zm",
    "dinocfaughead-0.05-11": "joopo3jp",
    "dinocfaughead-0.01-11": "a6nrvioo",
    "dinocfaughead-0.1-11": "3wazgrbi",
    "dinocfaughead-0.01-22": "86tdpmmy",
    "dinocfaughead-0.25-11": "0mws6jn2",
    "dinocfaughead-0.05-22": "ba2i7eoc",
    "dinocfaughead-0.1-22": "knlqo7rv",
    "dinocfaughead-0.25-22": "0zb21zfn",
    "dinocfaughead-0.01-33": "qud58gtm",
    "dinocfaughead-0.05-33": "wkap5o7k",
    "dinocfaughead-0.1-33": "weynxvhr",
    "dinocfaughead-0.25-33": "cvksjlsd",
    "dinocfaughead-1.0-11": "6qv0ku9z",
    "dinocfaughead-1.0-22": "95hsjw59",
    "dinocfaughead-1.0-33": "zxipvul4",
    "dinocfaug-0.05-11": "xkekulrp",
    "dinocfaug-0.1-22": "ghpb8vpn",
    "dinocfaug-0.05-22": "jclb1g8q",
    "dinocfaug-0.1-11": "wf5j3pcm",
    "dinocfaug-1.0-11": "ncco5zra",
    "dinocfaug-0.05-33": "4s0ni59n",
    "dinocfaug-0.25-11": "twqqf8ni",
    "dinocfaug-0.25-22": "tfrsb6ef",
    "dinocfaug-1.0-22": "uh104rkn",
    "dinocfaug-0.1-33": "0gdbcki3",
    "dinocfaug-0.01-22": "l9q2h7zf",
    "dinocfaug-0.25-33": "lnshyc7y",
    "dinocfaug-0.01-11": "yaf2sjkd",
    "dinocfaug-0.01-33": "a3hgfisq",
    "dinocfaug-1.0-33": "6lrah7s0",
    "dinocf2-0.25-33": "5fcsslnz",
    "dinocf2-0.25-22": "c112yhkg",
    "dinocf2-0.1-22": "aq3g35rh",
    "dinocf2-0.1-33": "02jwpzxr",
    "dinocf2-0.01-33": "og5v6x4y",
    "dinocf2-0.01-22": "bj4tvibz",
    "dinocf2-0.01-11": "rikim7nk",
    "dinocf2head-0.01-22": "uy3vlmgo",
    "dinocf2head-0.01-11": "rudaqz35",
    "dinocf2head-0.01-33": "weagk5o9",
    "dinocf2head-0.1-11": "45cowjnk",
    "dinocf2head-0.1-22": "djarr2j9",
    "dinocf2head-0.1-33": "ok425idd",
    "dinocf2head-0.25-33": "a60yd6so",
    "dinocf2head-0.25-22": "zbqwemvg",
    "dinocf2head-0.25-11": "sw0q1uny",
    "dinocf2head-0.05-22": "nr8h8dax",
    "dinocf2head-0.05-33": "s4yi5zrw",
    "dinocf2head-0.05-11": "5ogvjhr0",
    "dinocf2-0.05-11": "4bnn4her",
    "dinocf2head-0.05-22": "nr8h8dax",
    "dinocf2-0.05-33": "w23n4w8g",
    "dinocf2-0.05-22": "utx7shga",
    "dinocf2-1.0-11": "eel2lds9",
    "dinocf2-1.0-22": "nco7tji6",
    "dinocf2-1.0-33": "sser6vab",
    "dinocf2head-1.0-11": "nwho8tpk",
    "dinocf2head-1.0-22": "0d93bik3",
    "dinocf2head-1.0-33": "oafqei2m",
}

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=["experiment=base_density", "data.cache=False"],
    )
    print(cfg)
    data_module = EmbedDataModule(config=cfg)
test_loader = data_module.test_dataloader()

filename = "../outputs/dino_embed2.csv"
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
            filename,
            index=False,
        )
df
