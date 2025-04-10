from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from causal_models.trainer import preprocess_batch

from torchvision.transforms import CenterCrop
from causal_models.hps import Hparams
from causal_models.hvae import HVAE2
from causal_models.train_setup import (
    setup_dataloaders,
    load_finetuned_vae,
    load_vae_and_module,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    from PIL import Image

    parser.add_argument(
        "--vae_path",
        default="/vol/biomedic3/mb121/causal-contrastive/outputs/scanner/beta1balanced/last_19.pt",
    )
    parser.add_argument("--loader", default="all", type=str)
    parser.add_argument(
        "--folder_for_counterfactuals",
        default="cf_beta1balanced_scanner",
    )
    parsed_args = parser.parse_args()
    # model_path = '/vol/biomedic3/mb121/causal-contrastive/causal_models/counterfactual_evaluation/cf_finetune.ckpt'
    # model, args = load_finetuned_vae(model_path)
    # model_path = '/vol/biomedic3/mb121/causal-contrastive/outputs/scanner/bad/checkpoint.pt'

    model, _, args = load_vae_and_module(parsed_args.vae_path)
    dataloader = setup_dataloaders(args, cache=False, shuffle_train=False)

    if parsed_args.loader == "all":
        splits = ["train", "valid", "test"]
    else:
        assert parsed_args.loader in ["train", "valid", "test"]
        splits = [parsed_args.loader]

    cf_dir = Path(parsed_args.folder_for_counterfactuals)

    cf_dir.mkdir(parents=True, exist_ok=True)
    model.cuda()
    u_t = 1.0
    t = 0.8

    for split in splits:
        print(f"######### Starting {split} ######### ")
        loader = dataloader[split]
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):
                batch = preprocess_batch(args, batch, expand_pa=False)
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

                px_loc, px_scale = model.forward_latents(
                    latents=zs, parents=_pa.cuda(), t=t
                )

                for s in range(5):
                    cf_pa = {k: pa[k] for k in args.parents_x}
                    cf_pa["scanner"] = torch.nn.functional.one_hot(
                        torch.ones_like(torch.argmax(cf_pa["scanner"], 1)) * s,
                        num_classes=5,
                    )

                    _cf_pa = torch.cat([cf_pa[k] for k in args.parents_x], dim=1)
                    _cf_pa = (
                        _cf_pa[..., None, None]
                        .repeat(1, 1, *(args.input_res,) * 2)
                        .to(args.device)
                        .float()
                    )

                    cf_loc, cf_scale = model.forward_latents(
                        zs, parents=_cf_pa.cuda(), t=t
                    )

                    u = (batch["x"] - px_loc) / px_scale.clamp(min=1e-12)
                    cf_scale = cf_scale * u_t
                    cf_x = torch.clamp(cf_loc + cf_scale * u, min=-1, max=1)
                    cf_x = (cf_x + 1) / 2.0
                    cf_x = CenterCrop((256, 192))(cf_x).cpu()

                    for j in range(cf_x.shape[0]):
                        if torch.argmax(batch["scanner"][j]) != s:
                            short_path = batch["shortpath"][j][:-4]
                            filename = cf_dir / f"{short_path}_s{s}.png"
                            filename.parent.mkdir(parents=True, exist_ok=True)
                            img = Image.fromarray(np.uint8(cf_x[j, 0] * 255), "L")
                            img.save(str(filename))
