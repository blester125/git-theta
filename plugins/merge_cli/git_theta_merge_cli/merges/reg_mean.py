"""Merging via RegMean. https://arxiv.org/abs/2212.09849"""

import git_theta_merge_cli.merges.utils as utils
import numpy as np
import torch
from git_theta_merge_cli.merges.base import (
    LoRAMixin,
    PyTorchGPUMixin,
    PyTorchMixin,
    RWVariadicMerge,
)


class RegMean(PyTorchMixin, RWVariadicMerge):
    """Merge parameters by minimizing the L2 distance between the individual layer outputs and the merged output."""

    name = "reg-mean"

    def __init__(self, *args, merge_lambda: float = 0.5, **kwargs):
        """Merge parameters using RegMean.

        Args:
          merge_lambda: Scales the non-diagonal elements of the gram matrix of
            the input activations.
        """
        super().__init__(*args, **kwargs)
        self.merge_lambda = merge_lambda

    def merge(self, params, aux_data, ancestor, **kwargs):
        # Average parameters that don't have aux data.
        if any(ad is None for ad in aux_data):
            # Pass None for the merge lambda to force averaging.
            self.logger.debug(
                f"No aux_data for {'/'.join(kwargs['param_name'])}, merging by averaging."
            )
            merging_lambdas = utils.get_merge_lambdas(None, len(params), uniform=True)
            return utils.interpolate(params, merging_lambdas)

        # Scale non-diagonal elements to improve invertability.
        scaled_gram_matrices = [
            scale_non_diag(gm, self.merge_lambda) for gm in aux_data
        ]
        params = [p.to(sgm.dtype) for p, sgm in zip(params, scaled_gram_matrices)]
        # Calculate the output activations.
        scaled_gram_times_weight = [
            torch.matmul(sgm, p.T) for sgm, p in zip(scaled_gram_matrices, params)
        ]
        # Sum of input activations across models.
        sum_of_grams = utils.interpolate(scaled_gram_matrices)
        inv_grams = matrix_inverse(sum_of_grams)
        # Sum of output activations across models.
        sum_of_scaled_gram_times_weight = utils.interpolate(scaled_gram_times_weight)
        # Closed form solution to least squares regression.
        merged_param = torch.matmul(inv_grams, sum_of_scaled_gram_times_weight)

        return merged_param.T


class RegMeanGPU(PyTorchGPUMixin, RegMean):
    """Do it on the GPU!"""

    name = "reg-mean-gpu"


class RegMeanLoRA(LoRAMixin, RegMean):
    """Combine LoRA B @ A before merging on the GPU."""

    name = "lora-reg-mean"


class RegMeanLoRAGPU(LoRAMixin, RegMeanGPU):
    """Combine LoRA B @ A before merging."""

    name = "lora-reg-mean-gpu"


def scale_non_diag(p, s) -> torch.Tensor:
    """Scale non-diagonal parameters."""
    return s * p + (1 - s) * torch.diag_embed(torch.diagonal(p))


def matrix_inverse(t):
    """Invert matrix `t` with some edge cases for singual matrices due to zeros."""
    matrixInverse_fn = torch.linalg.inv
    parameter = t
    try:
        original_device = parameter.device
        matrix_inverse = matrixInverse_fn(parameter.cuda()).to(original_device)
        return matrix_inverse
    except:
        # If matrix is not invertible because row/col is 0, then we remove the row/col with 0 and invert the submatrix
        # We then insert the row/col with 0 back afterwards
        nonZero_rowIdx = (torch.sum(parameter, dim=1) != 0).nonzero().squeeze()
        nonZero_colIdx = (torch.sum(parameter, dim=0) != 0).nonzero().squeeze()
        assert (nonZero_colIdx == nonZero_rowIdx).all()
        num_row = parameter.shape[0]
        nonZero_broadcastColIdx = nonZero_colIdx[None, :].repeat((num_row, 1))
        nonZero_broadcastRowIdx = nonZero_rowIdx[:, None].repeat(
            (1, nonZero_broadcastColIdx.shape[1])
        )

        # Get submatrix that is full rank
        fullRank_parameter = torch.gather(parameter, 1, nonZero_broadcastColIdx)
        fullRank_parameter = torch.gather(
            fullRank_parameter, 0, nonZero_broadcastRowIdx
        )

        # Invert submatrix that is full rank
        inverse_fullRankParameter = matrixInverse_fn(fullRank_parameter)
        inverse_parameter = copy.deepcopy(parameter)
        inverse_parameter[
            nonZero_rowIdx[:, None], nonZero_colIdx
        ] = inverse_fullRankParameter
        return inverse_parameter
