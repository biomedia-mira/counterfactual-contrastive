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
    "simclr-1.0-33": "84lv0t6h",
    "simclr-1.0-22": "yp8kgxkn",
    "simclr-1.0-11": "wmbptk3z",
    "simclr-0.1-33": "wtces7tl",
    "simclr-0.1-22": "c0qmmgx1",
    "simclr-0.1-11": "ouucdcl5",
    "simclr-0.25-11": "np395uhs",
    "simclr-0.25-22": "ypyw4lrc",
    "simclr-0.25-33": "nrep0c0s",
    "supervised-0.1-11": "1kd02vig",
    "supervised-1.0-22": "u6mm38zx",
    "supervised-1.0-33": "wgq57ak4",
    "supervised-1.0-11": "1e18dovk",
    "supervised-0.1-33": "1njtbkre",
    "supervised-0.1-22": "yl2qnv2s",
    "supervised-0.25-11": "af029hmt",
    "supervised-0.25-22": "r5rzknzo",
    "supervised-0.25-33": "yms2a9pj",
    "simclrcfaug-1.0-33": "toecr61u",
    "simclrcfaug-0.1-33": "vvkk0rqq",
    "simclrcfaug-1.0-22": "msyiln0g",
    "simclrcfaug-0.1-22": "tmjl2u4z",
    "simclrcfaug-1.0-11": "enhr8r5w",
    "simclrcfaug-0.1-11": "528qi412",
    "simclrcfaug-0.25-11": "vos0kkys",
    "simclrcfaug-0.25-22": "p0ic7hbb",
    "simclrcfaug-0.25-33": "dxyr38as",
    "simclrcf-0.1-11": "gxzgrx09",
    "simclrcf-1.0-11": "mydv7jl9",
    "simclrcf-0.1-22": "7fmc7i5u",
    "simclrcf-1.0-22": "4nep2y5i",
    "simclrcf-0.1-33": "rth6ff0x",
    "simclrcf-1.0-33": "0ra6hohi",
    "simclrcf-0.25-11": "gf75s11v",
    "simclrcf-0.25-33": "tgot0avm",
    "simclrcf-0.25-22": "dqozk1f5",
    "simclrhead-0.25-33": "85yznxgy",
    "simclrcfaughead-0.25-33": "o29t3fk8",
    "simclrhead-1.0-33": "95qyx64t",
    "simclrhead-0.1-33": "nlvsmcne",
    "simclrcfaughead-0.1-33": "z96xslzy",
    "simclrcfaughead-1.0-33": "52e65qwx",
    "simclrcfaughead-0.25-22": "m5sn2pr8",
    "simclrhead-0.25-22": "uxu4dwzu",
    "simclrcfaughead-0.1-22": "xhqtu3m2",
    "simclrcfaughead-1.0-22": "of3d28jb",
    "simclrcfaughead-0.1-11": "w6tbzaly",
    "simclrcfaughead-1.0-11": "4jvrmbpw",
    "simclrhead-1.0-22": "9m18itid",
    "simclrhead-0.25-11": "6iwtu99y",
    "simclrhead-0.1-11": "hdimroq2",
    "simclrhead-1.0-11": "8o5owv7w",
    "simclrcfhead-0.25-33": "hvq81iwv",
    "simclrcfhead-0.1-33": "2uigw3n5",
    "simclrcfhead-1.0-33": "qx3bvg8m",
    "simclrcfhead-0.25-22": "4rquh0fm",
    "simclrcfhead-0.1-22": "bhayhjze",
    "simclrcfhead-1.0-22": "f5xtz51t",
    "simclrcfhead-0.25-11": "8nofqdss",
    "simclrcfhead-0.1-11": "tck22uuf",
    "simclrcfhead-1.0-11": "p9hoda5n",
}


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


df = pd.read_csv("../outputs/classification_chexfinetunepneumo_results.csv")
for run_name, run_id in model_dict_normal.items():
    already_in_df = run_name in df.run_name.values
    if run_id != "" and not already_in_df:
        print(run_name)
        model_to_evaluate = f"../outputs/run_{run_id}/best.ckpt"
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
        df.to_csv(
            "../outputs/classification_chexfinetunepneumo_results.csv", index=False
        )
