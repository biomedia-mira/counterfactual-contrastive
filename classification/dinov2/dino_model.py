# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial

import torch
from torch import nn

from classification.dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss

try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")

import copy

import numpy as np

from typing import Sequence, Tuple, Union, Callable

import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from classification.dinov2.layers import (
    Mlp,
    PatchEmbed,
    MemEffAttention,
    DINOHead,
    NestedTensorBlock as Block,
)


class MySSLMetaArch(nn.Module):
    def __init__(
        self,
        encoder_model_name,
        img_size,
        in_chans,
        patch_size=16,
        head_n_prototypes=65536,
        dino_loss_weight=1,
        koleo_loss_weight=1,
        ibot_loss_weight=1,
        head_hidden_dim=2048,
        head_bottleneck_dim=256,
        head_nlayers=3,
        ibot_mask_ratio_min_max=(0.1, 0.5),
        mask_sample_probability=0.5,
    ):
        super().__init__()

        match encoder_model_name:
            case "dino_vit_small":
                self.student = vit_small(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans
                )
            case "dino_vit_base":
                self.student = vit_base(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans
                )
            case _:
                raise NotImplementedError

        self.teacher = copy.deepcopy(self.student)

        embed_dim = self.student.embed_dim

        self.embed_dim = embed_dim
        self.dino_out_dim = head_n_prototypes

        self.do_dino = dino_loss_weight > 0
        self.do_koleo = koleo_loss_weight > 0
        self.do_ibot = ibot_loss_weight > 0

        self.dino_loss_weight = dino_loss_weight
        self.koleo_loss_weight = koleo_loss_weight

        dino_head = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=head_n_prototypes,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
            nlayers=head_nlayers,
        )

        self.dino_loss = DINOLoss(self.dino_out_dim)

        if self.do_koleo:
            self.koleo_loss = KoLeoLoss()

        if self.do_ibot:
            self.student.dino_head = dino_head()
            self.teacher.dino_head = dino_head()

        if self.do_ibot:
            self.ibot_loss_weight = ibot_loss_weight
            self.ibot_out_dim = head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

    def get_loss(self, images, teacher_temp):
        n_global_crops = 2

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)
        batch_size = int(global_crops.shape[0] / n_global_crops)
        n_local_crops = int(local_crops.shape[0] / batch_size)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)

            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat(
                (teacher_cls_tokens[1], teacher_cls_tokens[0])
            )
            ibot_teacher_patch_tokens = teacher_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if self.do_ibot:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                    upperbound + n_cls_tokens, _dim
                )
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[
                        n_cls_tokens : n_cls_tokens + n_masked_patches
                    ],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]

            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(
                    teacher_cls_tokens
                )
                masked_teacher_ibot_softmaxed_centered = None

            # just keep centering option from original code
            teacher_dino_softmaxed_centered_list = (
                self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(
                    n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:]
                )
            )
            self.dino_loss.update_center(teacher_cls_tokens_after_head)

            if self.do_ibot:
                masked_teacher_patch_tokens_after_head = (
                    masked_teacher_patch_tokens_after_head.unsqueeze(0)
                )
                masked_teacher_ibot_softmaxed_centered = (
                    self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches],
                        teacher_temp=teacher_temp,
                    )
                )
                masked_teacher_ibot_softmaxed_centered = (
                    masked_teacher_ibot_softmaxed_centered.squeeze(0)
                )
                self.ibot_patch_loss.update_center(
                    masked_teacher_patch_tokens_after_head[:n_masked_patches]
                )

            return (
                teacher_dino_softmaxed_centered_list,
                masked_teacher_ibot_softmaxed_centered,
            )

        (
            teacher_dino_softmaxed_centered_list,
            masked_teacher_ibot_softmaxed_centered,
        ) = get_teacher_output()

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        (
            student_global_backbone_output_dict,
            student_local_backbone_output_dict,
        ) = self.student(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict[
            "x_norm_clstoken"
        ]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if self.do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(
                upperbound, _dim
            )
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(
                    ibot_student_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )

            inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(
            inputs_for_student_head_list
        )
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if self.do_ibot:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(
                0
            )[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(
                    n_local_crops
                ),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if self.do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if self.do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        return loss_accumulator, loss_dict


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.0,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "identity":
            print("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.fc = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.img_size = img_size
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = [
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
        ]  # Recover the number of patches in each dimension
        assert N == M[0] * M[1]
        kwargs = {}

        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error
            #  in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are
            # using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M[0]
            sy = float(h0 + self.interpolate_offset) / M[1]
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M[0], M[1], dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.fc(ret["x_norm_clstoken"])

    def reset_classifier(self, num_classes):
        self.fc = torch.nn.Linear(self.embed_dim, num_classes)
        for p in self.fc.parameters():
            p.requires_grad = True


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
