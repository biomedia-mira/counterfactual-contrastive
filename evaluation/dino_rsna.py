import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from hydra import compose, initialize
from classification.classification_module import ClassificationModule
from data_handling.xray import RSNAPneumoniaDataModule
from evaluation.helper_functions import run_inference

os.chdir("/vol/biomedic3/mb121/causal-contrastive/evaluation")

# Mapping from human readable run name to Weights&Biases run_id.

# Human readable name should be in format:
# for finetuning:
# {simclr/simclrcf/simclrcfaug}-{train_prop}-{seed}
# for linear probing
# {simclr/simclrcf/simclrcfaug}head-{train_prop}-{seed}

model_dict_normal = {
    # "dinocfhead-0.1-11": "cldniww9",
    # "dinocfhead-0.1-22": "5v8j0u7z",
    # "dinocfhead-0.1-33": "x8b6s9w3",
    # "dinocfhead-0.25-11": "w4yjd1u4",
    # "dinocfhead-0.25-22": "yvqrflhu",
    # "dinocfhead-0.25-33": "6y77hu2k",
    # "dinocfhead-1.0-11": "e2b7vfk5",
    # "dinocfhead-1.0-22": "i8u0buao",
    # "dinocfhead-1.0-33": "55yuuwqh",
    "dinohead-0.1-11": "jy033hbb",
    "dinohead-0.1-22": "arzbg3l3",
    "dinohead-0.1-33": "q3akd0w4",
    "dinohead-0.25-11": "m1sqn6cm",
    "dinohead-0.25-22": "zaf1qmr3",
    "dinohead-0.25-33": "hg88eqls",
    "dinohead-1.0-33": "y7y7ypc8",
    "dinohead-1.0-22": "6nj3ops1",
    "dinohead-1.0-11": "nijkwzsz",
    # "dinocf-1.0-33": "93wejvof",
    # "dinocf-1.0-22": "64ex86x1",
    # "dinocf-1.0-11": "gan839v4",
    # "dinocf-0.25-33": "wtvrx9pr",
    # "dinocf-0.25-22": "0u9h5e5o",
    # "dinocf-0.25-11": "mzpzmj7y",
    # "dinocf-0.1-22": "mhto1bu4",
    # "dinocf-0.1-33": "0ds5cq7j",
    # "dinocf-0.1-11": "xlseexyc",
    "dino-1.0-33": "faxy1go6",
    "dino-1.0-22": "rlgtjgyf",
    "dino-0.25-33": "a7g0kxgd",
    "dino-0.25-22": "yscsyiiy",
    "dino-0.1-33": "qbek79ek",
    "dino-0.1-22": "l0e5u8eh",
    "dino-1.0-11": "z90w3obw",
    "dino-0.25-11": "vrbjkhju",
    "dino-0.1-11": "s9t6hc59",
    # 'dinocfaughead-1.0-33': 'wbkyfkyk',
    # 'dinocfaughead-0.1-33': 'b2v3sxwl',
    # 'dinocfaughead-0.25-33': 'ukgqdsmp',
    # 'dinocfaughead-1.0-22': 'if2johxy',
    # 'dinocfaughead-0.1-22': 'jimppwfn',
    # 'dinocfaughead-0.25-22': 'ny83v1af',
    # 'dinocfaughead-1.0-11': '2rp5nvdr',
    # 'dinocfaughead-0.1-11': '1hm04por',
    # 'dinocfaughead-0.25-11': '3ajnn3ga',
    # 'dinocfaug-0.1-33': '4ubzhprm',
    # 'dinocfaug-1.0-11': 'qrcl0xyx',
    # 'dinocfaug-0.25-22': 'ucge88sm',
    # 'dinocfaug-0.1-22': '1i48qcss',
    # 'dinocfaug-0.05-33': 'ngypk1bl',
    # 'dinocfaug-0.05-22': '2axnd2dp',
    # 'dinocfaug-0.1-11': 'kw8ru1i5',
    # 'dinocfaug-0.25-11': 'uuq51rx1',
    # 'dinocfaug-0.05-11': 'h4988zuu',
    # 'dinocfaug-1.0-33': '0uuphqxf',
    # 'dinocfaug-0.25-33': 'du1dobk6',
    # 'dinocfaug-1.0-22': '63fz9q2i',
    "dinocfaug2-0.1-22": "4vhox2z9",
    "dinocfaug2-0.25-22": "j348o6wt",
    "dinocfaug2-0.25-11": "lc2sn03j",
    "dinocfaug2-0.1-11": "ejyhaomc",
    "dinocfaug2-1.0-33": "u5ftk89t",
    "dinocfaug2-0.25-33": "u9zhe3oj",
    "dinocfaug2-0.1-33": "lzk2k2vi",
    "dinocfaug2-1.0-22": "d3lcggp4",
    "dinocfaug2-1.0-11": "5ydcxu5o",
    "dinocfhead-1.0-11": "fpk62d9s",
    "dinocfhead-1.0-33": "lhbkbrzd",
    "dinocfhead-0.25-22": "k9p1voyv",
    "dinocfhead-0.25-33": "l8nio6j6",
    "dinocfhead-1.0-22": "38rdgpx4",
    "dinocfhead-0.1-22": "bhogylza",
    "dinocfhead-0.1-33": "gxw4owxm",
    "dinocfhead-0.25-11": "dy47gsly",
    "dinocfhead-0.1-11": "4222skp8",
    # 'dinocf3head-1.0-33': 'jifxctzd',
    # 'dinocf3head-0.25-33': 'au1bbea6',
    # 'dinocf3head-1.0-22': 'ub2gqklg',
    # 'dinocf3head-0.1-33': 'wep8c51k',
    # 'dinocf3head-0.25-22': 'iv9epcev',
    # 'dinocf3head-0.25-11': 'p0tvv31w',
    # 'dinocf3head-1.0-11': '4b7z3jzt',
    # 'dinocf3head-0.1-22': '8b5o5km0',
    # 'dinocf3head-0.1-11': 'gnc5jyzd',
    "dinocf-1.0-11": "rcgrmvh3",
    "dinocf-1.0-22": "bg159ond",
    "dinocf-1.0-33": "z8conjpm",
    "dinocf-0.25-22": "9ovptzfj",
    "dinocf-0.25-11": "8b7flh1l",
    "dinocf-0.25-33": "4y43iqe9",
    "dinocf-0.1-33": "uo0nu8a9",
    "dinocf-0.1-22": "k8e7205s",
    "dinocf-0.1-11": "xwxbs2f2",
    "dinocfaug2head-1.0-33": "qcrkuly0",
    "dinocfaug2head-0.1-33": "8gicnv2h",
    "dinocfaug2head-1.0-22": "ko1yi04h",
    "dinocfaug2head-0.25-33": "vqd0vkok",
    "dinocfaug2head-0.25-11": "n1d2b2be",
    "dinocfaug2head-0.25-22": "rln7inrz",
    "dinocfaug2head-0.1-11": "a3d3z14c",
    "dinocfaug2head-0.1-22": "oak45z6p",
}

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="config.yaml",
        overrides=["experiment=base_rsna", "data.cache=False"],
    )
    data_module = RSNAPneumoniaDataModule(config=cfg)

test_dataloader = data_module.test_dataloader()

filename = "../outputs/dino_rsna2.csv"


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
        print("\nEvaluating RSNA")
        res = {}
        res["N_test"] = [inference_results["targets"].shape[0]]
        res["Scanner"] = ["RSNA"]
        res["run_name"] = run_name
        res["ROC"] = [
            roc_auc_score(
                inference_results["targets"], inference_results["confs"][:, 1]
            )
        ]
        print(res)
        df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)
        df.to_csv(filename, index=False)
