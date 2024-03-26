"""Merging via TIES. https://arxiv.org/abs/2306.01708"""

import numpy as np
import torch
from git_theta_merge_cli.merges import utils
from git_theta_merge_cli.merges.base import (
    LoRAMixin,
    PyTorchGPUMixin,
    PyTorchMixin,
    RWVariadicMerge,
)


class TIES(PyTorchMixin, RWVariadicMerge):
    """Merge models by removing parameters that conflict from models."""

    name = "ties"

    def __init__(self, *args, merge_lambda: float = 0.1, **kwargs):
        """Merge with TIES.

        Args:
          merge_lambda: Scales the TIES task vector.
        """
        super().__init__(*args, **kwargs)
        self.merge_lambda = merge_lambda

    def memory_efficient_ies(self, params):
        """Dis-mean averaging with a low memory footprint."""
        sum_params = utils.memory_efficient_interpolate(params)
        # Get the sign of the each parameter summed across models.
        resolved_sign = torch.sign(sum_params)
        # Replace any zeros with the majority sign across parameters.
        # Note: This diverges from the TIES implementation slightly, instead of
        #       being the majority sign *within this layer*, the original paper
        #       uses the majority sign across the whole model.
        resolved_sign.masked_fill_(
            resolved_sign == 0, torch.sign(torch.sum(resolved_sign))
        )

        # Manually do the memory efficient sum as building multiple lists uses too much.
        sum_selected = torch.zeros_like(params[0])
        num_selected = torch.zeros_like(params[0])
        for param in params:
            # Select the parameters whose values match the sign.
            # Note: When the selected sign was zero, picking the majority sign
            #       is still important as the sign selection is performed on *each*
            #       parameter, not on the sum of parameter across models (which was 0).
            selected = torch.where(resolved_sign > 0, param > 0, param < 0)
            selected_param = param * selected

            # Sum the selected parameters.
            torch.add(sum_selected, selected_param, out=sum_selected)
            # Track the number of parameters that contribute to the sum.
            torch.add(num_selected, selected, out=num_selected)

        # Divide to convert the sum into an average, clamp to avoid divides by zero.
        disjoint_merge = sum_selected / torch.clamp(num_selected, min=1)
        return disjoint_merge

    def merge(self, params, aux_data, ancestor, **kwargs):
        # Ties merging with the "dis-mean" method. Aux data should be task vectors
        # calculated w.r.t `ancestor` that has been "trimmed", i.e., low-magnitude
        # parameters are zero'd out.
        merged_tv = self.memory_efficient_ies(aux_data)
        return ancestor + self.merge_lambda * merged_tv


class TIESGPU(PyTorchGPUMixin, TIES):
    """Do it on the GPU!"""

    name = "ties-gpu"


class TIESLoRA(LoRAMixin, TIES):
    """Combine LoRA B @ A before merging."""

    name = "lora-ties"


class TIESLoRAGPU(LoRAMixin, TIESGPU):
    """Combine LoRA B @ A before merging on the GPU."""

    name = "lora-ties-gpu"
