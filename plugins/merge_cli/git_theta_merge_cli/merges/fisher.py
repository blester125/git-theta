"""FisherMerging. https://arxiv.org/abs/2111.09832"""

import numpy as np
import torch
from git_theta_merge_cli.merges import utils
from git_theta_merge_cli.merges.base import (
    LoRAMixin,
    PyTorchGPUMixin,
    PyTorchMixin,
    RWVariadicMerge,
)


class FisherMerge(PyTorchMixin, RWVariadicMerge):
    """Merge models by finding a model with the highest joint probability according to each model's posterior."""

    name = "fisher"

    def __init__(self, *args, merge_lambda: float = 1.0, **kwargs):
        """Do fisher merging.

        Args:
          merge_lambda: Scales the diagonal fisher approximation.
        """
        super().__init__(*args, **kwargs)
        self.merge_lambda = merge_lambda

    def merge(self, params, aux_data, ancestor, **kwargs):
        # Convert the merged lambda to one that can be applied to each fisher.
        merging_lambdas = utils.get_merge_lambdas(self.merge_lambda, len(params))
        # Scale the diagonal fisher approximation.
        fishers = [f * l for f, l in zip(aux_data, merging_lambdas)]
        # Scale the parameters by the fisher (as out parameter is not stored as a
        # vector, we use elementwise multiplication instead of diag matrix-vector
        # multiplication).
        fisher_weighted_params = [p * f for p, f in zip(params, fishers)]
        # Sum the weighted parameters.
        merged_param = utils.interpolate(fisher_weighted_params)
        # Sum the fishers.
        all_fishers = utils.interpolate(fishers)
        # Divide, while avoiding a divide by zero.
        merged_param = merged_param / torch.where(
            all_fishers == 0, torch.ones_like(all_fishers), all_fishers
        )
        return merged_param


class FisherMergeGPU(PyTorchGPUMixin, FisherMerge):
    """Do it on the GPU."""

    name = "fisher-gpu"


class FisherMergeLoRA(LoRAMixin, FisherMerge):
    """Combine LoRA B @ A before merging."""

    name = "lora-fisher"


class FisherMergeLoRAGPU(LoRAMixin, FisherMergeGPU):
    """Combine LoRA B @ A before merging on the GPU."""

    name = "lora-fisher-gpu"
