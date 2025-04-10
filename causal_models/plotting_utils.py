import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_cxr_grid(x, fig=None, ax=None, nrows=1, cmap="Greys_r", norm=None, cbar=False):
    m, n = nrows, x.shape[0] // nrows
    if ax is None:
        fig, ax = plt.subplots(m, n, figsize=(n * 4, 8))
    im = []
    for i in range(m):
        for j in range(n):
            idx = (i, j) if m > 1 else j
            ax = [ax] if n == 1 else ax
            _x = x[i * n + j].squeeze()
            if _x.shape[0] == 3:
                _x = np.transpose(_x, [1, 2, 0]).int()
            _im = ax[idx].imshow(_x, cmap=cmap, norm=norm)
            im.append(_im)
            ax[idx].axes.xaxis.set_ticks([])
            ax[idx].axes.yaxis.set_ticks([])

    plt.tight_layout()

    if cbar:
        if fig:
            fig.subplots_adjust(wspace=-0.3, hspace=0.3)
        for i in range(m):
            for j in range(n):
                idx = [i, j] if m > 1 else j
                cbar_ax = fig.add_axes(
                    [
                        ax[idx].get_position().x0,
                        ax[idx].get_position().y0 - 0.02,
                        ax[idx].get_position().width,
                        0.01,
                    ]
                )
                cbar = plt.colorbar(
                    im[i * n + j], cax=cbar_ax, orientation="horizontal"
                )
                _x = x[i * n + j].squeeze()

                d = 20
                _vmin, _vmax = _x.min().abs().item(), _x.max().item()
                _vmin = -(_vmin - (_vmin % d))
                _vmax = _vmax - (_vmax % d)

                lt = [_vmin, 0, _vmax]

                if (np.abs(_vmin) - 0) > d:
                    lt.insert(1, _vmin // 2)
                if (_vmax - 0) > d:
                    lt.insert(-2, _vmax // 2)

                cbar.set_ticks(lt)
                cbar.outline.set_visible(False)
    return fig, ax


def undo_norm(pa):
    # reverse [-1,1] parent preprocessing back to original range
    for k, v in pa.items():
        if k == "age":
            pa[k] = (v + 1) / 2 * 100  # [-1,1] -> [0,100]
    return pa


@torch.no_grad()
def plot_counterfactual_viz_mimic(args, x, cf_x, pa, cf_pa, do, rec_loc, save=True):
    fs = 15
    m, s = 6, 3
    n = 8
    fig, ax = plt.subplots(m, n, figsize=(n * s - 2, m * s))
    x = (x[:n].detach().cpu() + 1) * 127.5
    _, _ = plot_cxr_grid(x, ax=ax[0])

    cf_x = (cf_x[:n].detach().cpu() + 1) * 127.5
    rec_loc = (rec_loc[:n].detach().cpu() + 1) * 127.5
    _, _ = plot_cxr_grid(rec_loc, ax=ax[1])
    _, _ = plot_cxr_grid(cf_x, ax=ax[2])
    _, _ = plot_cxr_grid(
        rec_loc - x,
        ax=ax[3],
        fig=fig,
        cmap="RdBu_r",
        cbar=True,
        norm=MidpointNormalize(midpoint=0),
    )
    _, _ = plot_cxr_grid(
        cf_x - x,
        ax=ax[4],
        fig=fig,
        cmap="RdBu_r",
        cbar=True,
        norm=MidpointNormalize(midpoint=0),
    )
    _, _ = plot_cxr_grid(
        cf_x - rec_loc,
        ax=ax[5],
        fig=fig,
        cmap="RdBu_r",
        cbar=True,
        norm=MidpointNormalize(midpoint=0),
    )

    for j in range(n):
        msg = ""
        for i, (k, v) in enumerate(do.items()):
            if k == "sex":
                sex_categories = ["male", "female"]  # 0,1
                vv = sex_categories[int(v[j].item())]
                kk = "s"
            if k == "scanner":
                vv = str(int(v[j].item()))
                kk = "sca"
            else:
                kk = k
                vv = str(v[j])
            msg += kk + "{{=}}" + vv
            msg += ", " if (i + 1) < len(list(do.keys())) else ""
        ax[1, j].set_title("rec_loc")
        ax[2, j].set_title(rf"do(${msg}$)", fontsize=fs - 2, pad=8)
        ax[3, j].set_title("rec_loc - x")
        ax[4, j].set_title(
            "cf_loc - x",
            pad=8,
            fontsize=fs - 5,
            multialignment="center",
            linespacing=1.5,
        )
        ax[5, j].set_title("cf_loc - rec_loc")

    if save:
        fig.savefig(
            os.path.join(args.save_dir, f"viz-{args.iter}.png"), bbox_inches="tight"
        )
        return

    return fig


@torch.no_grad()
def plot_counterfactual_viz_embed(args, x, cf_x, pa, cf_pa, do, rec_loc, save=True):
    # do = undo_norm(do)
    pa = undo_norm(pa)
    cf_pa = undo_norm(cf_pa)

    fs = 15
    m, s = 6, 3
    n = 8
    fig, ax = plt.subplots(m, n, figsize=(n * s - 2, m * s))
    x = (x[:n].detach().cpu() + 1) * 127.5
    _, _ = plot_cxr_grid(x, ax=ax[0])

    cf_x = (cf_x[:n].detach().cpu() + 1) * 127.5
    rec_loc = (rec_loc[:n].detach().cpu() + 1) * 127.5
    _, _ = plot_cxr_grid(rec_loc, ax=ax[1])
    _, _ = plot_cxr_grid(cf_x, ax=ax[2])
    _, _ = plot_cxr_grid(
        rec_loc - x,
        ax=ax[3],
        fig=fig,
        cmap="RdBu_r",
        cbar=True,
        norm=MidpointNormalize(midpoint=0, vmin=-80, vmax=80),
    )
    _, _ = plot_cxr_grid(
        cf_x - x,
        ax=ax[4],
        fig=fig,
        cmap="RdBu_r",
        cbar=True,
        norm=MidpointNormalize(midpoint=0, vmin=-80, vmax=80),
    )
    _, _ = plot_cxr_grid(
        cf_x - rec_loc,
        ax=ax[5],
        fig=fig,
        cmap="RdBu_r",
        cbar=True,
        norm=MidpointNormalize(midpoint=0, vmin=-80, vmax=80),
    )
    scanner_categories = [
        "Selenia Dimensions",
        "Senographe Pristina",
        "Senograph 2000D",
        "Lorad Selenia",
        "Clearview CSm",
        "Senographe Essential",
    ]

    for j in range(n):
        msg = ""
        for i, (k, v) in enumerate(do.items()):
            if k == "scanner":
                if isinstance(v[j], str):
                    vv = v[j]
                else:
                    vv = scanner_categories[int(torch.argmax(v[j], dim=-1))]
                kk = "sc"
            else:
                kk = k
                vv = str(v[j])
            msg += kk + "{{=}}" + vv
            msg += ",\n" if (i + 1) < len(list(do.keys())) else ""

        title = ""
        if "scanner" in pa.keys():
            scan = scanner_categories[int(torch.argmax(pa["scanner"][j], dim=-1))]
            title += f"scn={scan}\n"
        ax[0, j].set_title(
            title,
            pad=8,
            fontsize=fs - 5,
            multialignment="center",
            linespacing=1.5,
        )
        ax[1, j].set_title("rec_loc")
        ax[2, j].set_title(rf"do(${msg}$)", fontsize=fs - 2, pad=8)
        ax[3, j].set_title("rec_loc - x")
        ax[4, j].set_title(
            "cf_loc - x",
            pad=8,
            fontsize=fs - 5,
            multialignment="center",
            linespacing=1.5,
        )
        ax[5, j].set_title("cf_loc - rec_loc")

    if save:
        fig.savefig(
            os.path.join(args.save_dir, f"viz-{args.iter}.png"), bbox_inches="tight"
        )
        return
    else:
        return fig


def write_images(args, model, batch):
    # reconstructions, first abduct z from q(z|x,pa)
    zs = model.abduct(x=batch["x"], parents=batch["pa"])
    if model.cond_prior:
        zs = [zs[j]["z"] for j in range(len(zs))]
    if "padchest" in args.hps:
        pa = {k: batch[k] for k in args.parents_x}
        _pa = torch.cat([batch[k] for k in args.parents_x], dim=1)
        _pa = (
            _pa[..., None, None]
            .repeat(1, 1, *(args.input_res,) * 2)
            .to(args.device)
            .float()
        )

        rec_loc, _ = model.forward_latents(zs, parents=_pa)
        # counterfactuals (focus on changing scanner)
        cf_pa = copy.deepcopy(pa)
        cf_pa = {k: batch[k] for k in args.parents_x}
        cf_pa["scanner"] = 1 - cf_pa["scanner"]
        do = {"scanner": cf_pa["scanner"]}
        _cf_pa = torch.cat([cf_pa[k] for k in args.parents_x], dim=1)
        _cf_pa = (
            _cf_pa[..., None, None]
            .repeat(1, 1, *(args.input_res,) * 2)
            .to(args.device)
            .float()
        )
        cf_loc, _ = model.forward_latents(zs, parents=_cf_pa)

        # plot this figure
        return plot_counterfactual_viz_mimic(
            args, batch["x"], cf_loc, pa, cf_pa, do, rec_loc
        )

    elif "embed" in args.hps:
        pa = {k: batch[k] for k in args.parents_x}
        _pa = torch.cat([batch[k] for k in args.parents_x], dim=1)
        _pa = (
            _pa[..., None, None]
            .repeat(1, 1, *(args.input_res,) * 2)
            .to(args.device)
            .float()
        )

        rec_loc, _ = model.forward_latents(zs, parents=_pa)
        # counterfactuals (focus on changing sex)
        cf_pa = copy.deepcopy(pa)
        cf_pa = {k: batch[k] for k in args.parents_x}
        cf_pa["scanner"] = torch.nn.functional.one_hot(
            torch.randint_like(torch.argmax(cf_pa["scanner"], 1), high=5), num_classes=5
        )
        do = {"scanner": cf_pa["scanner"]}
        _cf_pa = torch.cat([cf_pa[k] for k in args.parents_x], dim=1)
        _cf_pa = (
            _cf_pa[..., None, None]
            .repeat(1, 1, *(args.input_res,) * 2)
            .to(args.device)
            .float()
        )
        cf_loc, _ = model.forward_latents(zs, parents=_cf_pa)

        # plot this figure
        plot_counterfactual_viz_embed(args, batch["x"], cf_loc, pa, cf_pa, do, rec_loc)
        del rec_loc, cf_loc
        return
    else:
        raise NotImplementedError
