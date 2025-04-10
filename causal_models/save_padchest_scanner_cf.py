from pathlib import Path
from tqdm import tqdm
from causal_models.train_setup import load_vae_and_module

import torch
import numpy as np
from causal_models.trainer import preprocess_batch
import argparse
from PIL import Image


def scanner_counterfactuals(
    model,
    batch,
    args,
    partial_abduct=None,
    u_t=1.0,
    t=0.8,
    apply_sex_cf=False,
    apply_scanner_cf=True,
):
    model.cuda()
    results = {}
    pa = {k: batch[k] for k in args.parents_x}
    _pa = torch.cat([pa[k] for k in args.parents_x], dim=1)
    _pa = (
        _pa[..., None, None]
        .repeat(1, 1, *(args.input_res,) * 2)
        .to(args.device)
        .float()
    )

    zs = model.abduct(x=batch["x"].cuda(), parents=_pa.cuda(), t=1e-5)
    if model.cond_prior:
        zs = [zs[j]["z"] for j in range(len(zs))]
    if partial_abduct is not None:
        zs = zs[:partial_abduct]

    px_loc, px_scale = model.forward_latents(zs, parents=_pa, t=t)

    cf_pa = {k: pa[k] for k in args.parents_x}
    if apply_scanner_cf:
        cf_pa["scanner"] = 1 - cf_pa["scanner"]

    if apply_sex_cf:
        cf_pa["sex"] = 1 - cf_pa["sex"]

    _cf_pa = torch.cat([cf_pa[k] for k in args.parents_x], dim=1)
    _cf_pa = (
        _cf_pa[..., None, None]
        .repeat(1, 1, *(args.input_res,) * 2)
        .to(args.device)
        .float()
    )

    cf_loc, cf_scale = model.forward_latents(zs, parents=_cf_pa.cuda(), t=t)
    px_loc, px_scale = model.forward_latents(latents=zs, parents=_pa)
    cf_loc, cf_scale = model.forward_latents(latents=zs, parents=_cf_pa)

    u = (batch["x"] - px_loc) / px_scale.clamp(min=1e-12)
    cf_scale = cf_scale * u_t
    cf_x = torch.clamp(cf_loc + cf_scale * u, min=-1, max=1)
    results["DE"] = cf_x.cpu()
    results["REC"] = px_loc.cpu()
    results["GT"] = batch["x"].cpu()
    results["cf_col_values"] = cf_pa["scanner"]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vae_path",
        default="/vol/biomedic3/mb121/causal-contrastive/outputs/sex_scanner/beta3fixed/checkpoint.pt",  # noqa
    )
    parser.add_argument(
        "--folder_for_counterfactuals",
        default="padchest_v2_images",
    )

    parser.add_argument("--loader", default="train", type=str)

    parsed_args = parser.parse_args()

    vae, dataloader, args = load_vae_and_module(parsed_args.vae_path)

    if parsed_args.loader == "all":
        splits = ["train", "valid", "test"]
    else:
        assert parsed_args.loader in ["train", "valid", "test"]
        splits = [parsed_args.loader]

    cf_dir = Path(parsed_args.folder_for_counterfactuals)
    cf_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        loader = dataloader[split]
        for apply_scanner_cf in [False, True]:
            for apply_sex_cf in [False, True]:
                if apply_sex_cf or apply_scanner_cf:
                    print(apply_scanner_cf, apply_sex_cf)
                    with torch.no_grad():
                        for i, batch in enumerate(tqdm(loader)):
                            if apply_sex_cf:
                                if apply_scanner_cf:
                                    all_filename = [
                                        cf_dir / f"{x[:-4]}_sc_cf_sex_cf.png"
                                        for x in batch["shortpath"]
                                    ]
                                else:
                                    all_filename = [
                                        cf_dir / f"{x[:-4]}_sex_cf.png"
                                        for x in batch["shortpath"]
                                    ]
                            else:
                                all_filename = [
                                    cf_dir / f"{x[:-4]}_sc_cf.png"
                                    for x in batch["shortpath"]
                                ]
                            f_exist = [x for x in all_filename if x.exists()]
                            if len(all_filename) != len(f_exist):
                                print(
                                    len(all_filename), len(f_exist), print(all_filename)
                                )
                                batch = preprocess_batch(args, batch, expand_pa=False)
                                out_d = scanner_counterfactuals(
                                    model=vae,
                                    batch=batch,
                                    args=args,
                                    u_t=1.0,
                                    apply_scanner_cf=apply_scanner_cf,
                                    apply_sex_cf=apply_sex_cf,
                                )
                                cf_x = (out_d["DE"] + 1) / 2.0
                                for j in range(cf_x.shape[0]):
                                    short_path = batch["shortpath"][j][:-4]
                                    if apply_sex_cf:
                                        if apply_scanner_cf:
                                            filename = (
                                                cf_dir
                                                / f"{short_path}_sc_cf_sex_cf.png"
                                            )
                                        else:
                                            filename = (
                                                cf_dir / f"{short_path}_sex_cf.png"
                                            )
                                    else:
                                        filename = cf_dir / f"{short_path}_sc_cf.png"
                                    filename.parent.mkdir(parents=True, exist_ok=True)
                                    img = Image.fromarray(
                                        np.uint8(cf_x[j, 0] * 255), "L"
                                    )
                                    img.save(str(filename))
