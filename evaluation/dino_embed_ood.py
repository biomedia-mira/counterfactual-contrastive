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
    # "dinocfhead-0.05-11": "nszzsr29",
    # "dinocfhead-0.05-22": "i8a34upf",
    # "dinocfhead-0.05-33": "84dfz0ti",
    # "dinocfhead-0.1-22": "mnf1awb3",
    # "dinocfhead-0.1-33": "nc37fz57",
    # "dinocfhead-0.1-11": "ct57gn02",
    # "dinocfhead-0.25-33": "grw4ab4z",
    # "dinocfhead-0.25-11": "5w2jfmdf",
    # "dinocfhead-0.25-22": "o2bya0t9",
    # "dinocfhead-1.0-11": "ept1eapv",
    # "dinocfhead-1.0-22": "0a0fso4f",
    # "dinocfhead-1.0-33": "litr163j",
    "dinohead-0.05-33": "7ip1kx05",
    "dinohead-0.05-22": "d3dinkun",
    "dinohead-0.05-11": "t0yfokhv",
    "dinohead-0.1-11": "9dnnio2m",
    "dinohead-0.1-33": "09px8hgi",
    "dinohead-0.1-22": "6291x198",
    "dinohead-0.25-11": "m7ifga29",
    "dinohead-0.25-22": "mnlnwo2u",
    "dinohead-0.25-33": "x9hy1yh6",
    "dinohead-1.0-33": "ld2qm49z",
    "dinohead-1.0-11": "usdjwgnk",
    "dinohead-1.0-22": "vkdjp7gx",
    "dino-0.05-11": "ttq6wlqc",
    "dino-0.05-22": "pwzyp2su",
    "dino-0.05-33": "ecwwk630",
    "dino-0.1-11": "7l7gxyt0",
    "dino-0.1-22": "4bm2o9yi",
    "dino-0.1-33": "ezvxlfv2",
    "dino-0.25-11": "uyowac1y",
    "dino-0.25-22": "5inuk6u3",
    "dino-0.25-33": "rl4kt3kr",
    "dino-1.0-22": "6bdsdq7u",
    "dino-1.0-33": "cadow7o4",
    "dino-1.0-11": "225ud2qy",
    # "dinocf-0.05-11": "zln89yjz",
    # "dinocf-0.05-22": "rkuidk3d",
    # "dinocf-0.05-33": "gd4inbxa",
    # "dinocf-0.1-11": "cg4f6ief",
    # "dinocf-0.1-22": "w5kd67cc",
    # "dinocf-0.1-33": "vzxa5639",
    # "dinocf-0.25-11": "nte2fct4",
    # "dinocf-0.25-22": "jwalmg1a",
    # "dinocf-0.25-33": "0ckg1t0j",
    # "dinocf-1.0-22": "udjfb37c",
    "dinocfaughead-0.05-11": "asp1323w",
    "dinocfaughead-0.1-11": "vrsctfhl",
    "dinocfaughead-0.25-11": "1ue36c58",
    "dinocfaughead-0.05-22": "kbyqrp18",
    "dinocfaughead-0.1-22": "o631ci6t",
    "dinocfaughead-0.25-22": "3pgfwa7v",
    "dinocfaughead-0.05-33": "m7dr9au0",
    "dinocfaughead-1.0-11": "tm431jem",
    "dinocfaughead-0.1-33": "osfijj6s",
    "dinocfaughead-1.0-22": "tbsy7onp",
    "dinocfaughead-0.25-33": "vk93yip6",
    "dinocfaughead-1.0-33": "8b8xqgju",
    "dinocfaug-0.05-11": "9vhvvbcm",
    "dinocfaug-0.25-11": "psl920yl",
    "dinocfaug-1.0-33": "si5hb129",
    "dinocfaug-0.1-33": "2f6lnxz0",
    "dinocfaug-0.25-33": "nouga1jp",
    "dinocfaug-0.25-22": "t9ik6wt1",
    "dinocfaug-0.05-33": "uolakjh2",
    "dinocfaug-1.0-22": "3xpq4ypv",
    "dinocfaug-0.1-22": "2jb5e50l",
    "dinocfaug-1.0-11": "tqznx4ua",
    "dinocfaug-0.05-22": "yar17mof",
    "dinocfaug-0.1-11": "rhaqa3yz",
    "dinocf2head-0.25-22": "0ngnck56",
    "dinocf2head-0.1-22": "8zysq2d5",
    "dinocf2head-0.1-11": "kp98ohuh",
    "dinocf2head-0.05-11": "iexzxcab",
    "dinocf2-0.05-33": "mogm3tze",
    "dinocf2-0.1-33": "lhb3rrk1",
    "dinocf2-0.1-11": "2otr2sz4",
    "dinocf2-0.05-22": "xu2292og",
    "dinocf2-0.05-11": "lkhlk3ns",
    "dinocf2head-1.0-22": "f5n8uxrx",
    "dinocf2-1.0-33": "51ktbhdz",
    "dinocf2head-0.25-22": "0ngnck56",
    "dinocf2head-0.1-22": "8zysq2d5",
    "dinocf2head-0.1-11": "kp98ohuh",
    "dinocf2head-0.05-11": "iexzxcab",
    "dinocf2-0.05-33": "mogm3tze",
    "dinocf2head-0.25-11": "4orf2ukz",
    "dinocf2-0.1-22": "spnj9g2n",
    "dinocf2-0.25-22": "ep077zti",
    "dinocf2-0.25-11": "pdaex49y",
    "dinocf2head-1.0-33": "quv5ahxe",
    "dinocf2-1.0-22": "ho92jcg6",
    "dinocf2head-0.25-33": "654v6eg8",
    "dinocf2-1.0-11": "po1hmolz",
    "dinocf2head-0.1-33": "ifs431pi",
    "dinocf2head-0.05-22": "9bi5zahg",
    "dinocf2head-0.05-33": "bb1gsi3m",
    "dinocf2head-0.25-11": "4orf2ukz",
    "dinocf2-0.25-33": "b84swdx3",
    "dinocf2-0.1-22": "spnj9g2n",
    "dinocf2-0.25-22": "ep077zti",
    "dinocf2-0.25-11": "pdaex49y",
    "dinocf2head-1.0-11": "5nc3x9co",
}

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=["experiment=base_density", "data=embed_ood", "data.cache=False"],
    )
    print(cfg)
    data_module = EmbedOODDataModule(config=cfg)
test_loader = data_module.test_dataloader()

filename = "../outputs/dino_embed_ood.csv"

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
