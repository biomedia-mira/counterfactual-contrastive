import re
import numpy as np
import torch
from tqdm import tqdm


def extract_train_label_prop(run_name):
    pattern = r"-(\d+\.\d+)-"
    matches = re.search(pattern, run_name)
    if matches:
        extracted_data = matches.group(1)
        return float(extracted_data)
    return np.nan


def extract_run_type(run_name):
    # Define the regex pattern to split the string at the first dash
    pattern = r"-"

    # Use re.split to split the string at the first dash
    parts = re.split(pattern, run_name, maxsplit=1)

    # Extract the substring before the first dash
    if len(parts) > 1:
        substring_before_first_dash = parts[0]
        return substring_before_first_dash
    print("No dash found in the input string.")
    return ""


def extract_pretraining_type(run_name):
    run_name = re.split(r"-", run_name, maxsplit=1)[0]
    if "simclrcfaug" in run_name:
        return "SimCLR with CF\nin training set"
    if "dinocfaug" in run_name:
        return "DINO+"
    if "dinocf" in run_name:
        return "CF-DINO"
    if "dino" in run_name:
        return "DINO"
    if "simclrsexcf" in run_name:
        return "SexCF-SimCLR"
    if "simclrcf" in run_name:
        return "CF-SimCLR"
    elif "simclr" in run_name:
        return "SimCLR"
    elif "supervised" in run_name:
        return "ImageNet"
    return run_name


def extract_finetuning_type(run_name):
    run_name = re.split(r"-", run_name, maxsplit=1)[0]
    if "head" in run_name:
        return "Linear Probing"
    if "supervisedcf" in run_name or "cffine" in run_name:
        return "Finetuning with CF"
    if "transfer" in run_name or "trf" in run_name:
        return "Transfer"
    return "Finetuning"


def run_inference(dataloader, classification_model):
    confs = []
    true_label = []
    scanners = []
    sexs = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch["x"].float()
            logits = classification_model(inputs.cuda()).cpu()
            probas = torch.softmax(logits, 1).numpy()
            confs.append(probas)
            true_label.append(batch["y"].numpy())
            if "scanner" in batch.keys():
                if isinstance(batch["scanner"], torch.Tensor):
                    scanners.append(batch["scanner"].numpy())
                else:
                    scanners.append(batch["scanner"])
            if "sex" in batch.keys():
                if isinstance(batch["sex"], torch.Tensor):
                    sexs.append(batch["sex"].numpy())
                else:
                    sexs.append(batch["sex"])
    if len(sexs) > 0:
        sexs = np.concatenate(sexs)
    print(sexs)
    targets_true = np.concatenate(true_label)
    if targets_true.ndim == 2:
        if targets_true.shape[1] == 1:
            targets_true = targets_true.reshape(-1)
        else:
            targets_true = np.argmax(targets_true, 1)

    confs = np.concatenate(confs)
    if len(scanners) > 0:
        scanners = np.concatenate(scanners)
    return {"confs": confs, "targets": targets_true, "scanners": scanners, "sexs": sexs}


def run_get_embeddings(dataloader, classification_model, max_batch=100):
    all_feats = []
    true_label = []
    scanners = []
    paths = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            inputs = batch["x"].float()
            feats = classification_model.get_features(inputs.cuda()).cpu()
            all_feats.append(feats)
            true_label.append(batch["y"].numpy())
            paths.append(batch["shortpath"])
            if "scanner" in batch.keys():
                if isinstance(batch["scanner"], torch.Tensor):
                    scanners.append(batch["scanner"].numpy())
                else:
                    scanners.append(batch["scanner"])
            if i > max_batch:
                break

    targets_true = np.concatenate(true_label)
    if targets_true.ndim == 2:
        if targets_true.shape[1] == 1:
            targets_true = targets_true.reshape(-1)
        else:
            targets_true = np.argmax(targets_true, 1)

    feats = np.concatenate(all_feats)
    if len(scanners) > 0:
        scanners = np.concatenate(scanners)
    return {
        "targets": targets_true,
        "scanners": scanners,
        "feats": feats,
        "paths": np.concatenate(paths),
    }
