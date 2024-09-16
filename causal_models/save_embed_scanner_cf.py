from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from causal_models.trainer import preprocess_batch

from torchvision.transforms import CenterCrop
from causal_models.hps import Hparams
from causal_models.hvae import HVAE2
from causal_models.train_setup import setup_dataloaders


def load_vae_and_module(vae_path):
    args = Hparams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(vae_path, map_location=device)
    args.update(checkpoint["hparams"])
    if not hasattr(args, "cond_prior"):
        args.cond_prior = False
    args.device = device
    vae = HVAE2(args).to(args.device)
    vae.load_state_dict(checkpoint["ema_model_state_dict"])
    vae = vae.eval()
    dataloaders = setup_dataloaders(args, cache=False, shuffle_train=False)
    return vae, dataloaders, args


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    from PIL import Image

    parser.add_argument(
        "--vae_path",
        default="/vol/biomedic3/mb121/causal-contrastive/outputs/scanner/beta1balanced/last_19.pt",
    )
    parser.add_argument("--loader", default="train", type=str)
    parser.add_argument(
        "--folder_for_counterfactuals",
        default="cf_beta1balanced_scanner",
    )
    parsed_args = parser.parse_args()
    model, dataloader, args = load_vae_and_module(parsed_args.vae_path)

    if parsed_args.loader == "all":
        splits = ["train", "valid", "test"]
    else:
        assert parsed_args.loader in ["train", "valid", "test"]
        splits = [parsed_args.loader]

    cf_dir = Path(args.folder_for_counterfactuals)

    cf_dir.mkdir(parents=True, exist_ok=True)
    model.cuda()
    u_t = 0.5
    t = 0.1

    for split in splits:
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
                zs = model.abduct(x=batch["x"].cuda(), parents=_pa.cuda(), t=t)

                if model.cond_prior:
                    zs = [zs[j]["z"] for j in range(len(zs))]

                px_loc, px_scale = model.forward_latents(
                    latents=zs, parents=_pa.cuda(), t=t
                )

                for s in range(5):
                    cf_pa = {k: pa[k] for k in args.parents_x}
                    cf_pa["scanner"] = torch.nn.functional.one_hot(
                        torch.ones_like(torch.argmax(cf_pa["scanner"], 1)) * s,
                        num_classes=6,
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
