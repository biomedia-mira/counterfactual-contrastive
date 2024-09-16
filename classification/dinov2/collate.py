# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


class DinoCollator(object):
    def __init__(
        self,
        mask_ratio_tuple,
        mask_probability,
        dtype=torch.float32,
        n_tokens=None,
        mask_generator=None,
    ):
        self.mask_ratio_tuple = mask_ratio_tuple
        self.mask_probability = mask_probability
        self.dtype = dtype
        self.n_tokens = n_tokens
        self.mask_generator = mask_generator

    def __call__(self, samples_list):
        n_global_crops = len(samples_list[0]["global_crops"])
        n_local_crops = len(samples_list[0]["local_crops"])

        collated_global_crops = torch.stack(
            [s["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )

        collated_local_crops = torch.stack(
            [s["local_crops"][i] for i in range(n_local_crops) for s in samples_list]
        )

        base_images = torch.stack([s["x"] for s in samples_list])
        if not isinstance(samples_list[0]["y"], torch.Tensor):
            y = torch.stack([torch.tensor(s["y"]) for s in samples_list])
        else:
            y = torch.stack([s["y"] for s in samples_list])

        B = len(collated_global_crops)
        N = self.n_tokens
        n_samples_masked = int(B * self.mask_probability)
        probs = torch.linspace(*self.mask_ratio_tuple, n_samples_masked + 1)
        upperbound = 0
        masks_list = []
        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            masks_list.append(
                torch.BoolTensor(
                    self.mask_generator(int(N * random.uniform(prob_min, prob_max)))
                )
            )
            upperbound += int(N * prob_max)
        for i in range(n_samples_masked, B):
            masks_list.append(torch.BoolTensor(self.mask_generator(0)))

        random.shuffle(masks_list)

        collated_masks = torch.stack(masks_list).flatten(1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

        masks_weight = (
            (1 / collated_masks.sum(-1).clamp(min=1.0))
            .unsqueeze(-1)
            .expand_as(collated_masks)[collated_masks]
        )

        return {
            "collated_global_crops": collated_global_crops.to(self.dtype),
            "collated_local_crops": collated_local_crops.to(self.dtype),
            "collated_masks": collated_masks,
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "upperbound": upperbound,
            "n_masked_patches": torch.full(
                (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
            ),
            "x": base_images,
            "y": y,
        }
