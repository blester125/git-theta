"""Merging via MaTS. https://arxiv.org/abs/2312.04339"""

import json
import logging

import numpy as np
import torch
from git_theta_merge_cli.merges import utils
from git_theta_merge_cli.merges.base import (
    LoRAMixin,
    PyTorchGPUMixin,
    PyTorchMixin,
    RWVariadicMerge,
)
from scipy.sparse.linalg import LinearOperator, cg

import git_theta


class MaTS(PyTorchMixin, RWVariadicMerge):
    """Merge models by solving a linear system to upweight important directions in parameter subspaces.

    Args:
      iterations: The number of iterations to run conjugate gradient optimization.
      merge_log: Where to write the log conjugate gradient outputs.
    """

    name = "mats"

    def __init__(
        self,
        *args,
        iterations: int = 10,
        merge_log: str = "/tmp/merge_log.jsonl",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.iterations = iterations
        self.merge_logger = logging.getLogger("MaTS-Merge-Log")
        log_level = getattr(
            logging, git_theta.utils.EnvVarConstants.LOG_LEVEL.upper(), logging.INFO
        )
        self.merge_logger.setLevel(log_level)
        handler = git_theta.async_utils.AsyncTaskFileHandler(filename=merge_log)
        handler.setLevel(log_level)
        self.merge_logger.addHandler(handler)

    def maybe_transpose(self, p):
        """We might transpose the parameter based on how auxiliary data was collected."""
        return p

    def conjugate_gradient(self, sum_fishers, sum_fisher_scaled_params, initialization):
        """
        Run CG to solve Ax = b where A is the sum of Fishers and b is the sum of Fishers times checkpoints
        """
        A_matrix = sum_fishers
        b_vector = sum_fisher_scaled_params.flatten().float().cpu().numpy()
        weight_shape = sum_fisher_scaled_params.shape

        def matrix_vector_product(vector):
            reshaped_vector = (
                torch.from_numpy(vector)
                .reshape(weight_shape)
                .float()
                .to(A_matrix.device)
            )
            matrixVector = torch.matmul(A_matrix, reshaped_vector)
            return matrixVector.flatten().cpu().numpy()

        A = LinearOperator(
            (weight_shape.numel(), weight_shape.numel()),
            matvec=matrix_vector_product,
        )

        initialization = initialization.to(A_matrix.dtype)
        x0 = initialization.detach().flatten().cpu().numpy()

        x_final, exit_code = cg(A, b_vector, x0=x0, maxiter=self.iterations)

        initial_error = np.linalg.norm(matrix_vector_product(x0) - b_vector)
        final_error = np.linalg.norm(matrix_vector_product(x_final) - b_vector)

        merged_param = torch.tensor(x_final).reshape(weight_shape)

        error_log = {
            "exit_code": exit_code,
            "initial_error": initial_error.astype(float),
            "final_error": final_error.astype(float),
        }

        return merged_param, error_log

    def merge(self, params, aux_data, ancestor, **kwargs):
        # Average parameters that don't have aux data.
        if any(ad is None for ad in aux_data):
            # Pass None for the merge lambda to force averaging.
            self.logger.debug(
                f"No aux_data for {'/'.join(kwargs['param_name'])}, merging by averaging."
            )
            merging_lambdas = utils.get_merge_lambdas(None, len(params), uniform=True)
            return utils.interpolate(params, merging_lambdas)

        fishers = aux_data
        # Maybe transpose the parameters based on how the aux data was collected.
        params = [self.maybe_transpose(p) for p in params]
        params = [p.to(f.dtype) for p, f in zip(params, fishers)]

        # Manually unroll the memory efficient sum as intermediate lists can be too big.
        sum_fishers = torch.zeros_like(fishers[0])
        sum_fisher_scaled_params = torch.zeros_like(params[0])
        with torch.no_grad():
            for f, p in zip(fishers, params):
                # Scale the parameters by the fisher.
                fisher_scaled = torch.matmul(f, p)
                # Sum the fishers and the scaled parameters.
                torch.add(
                    sum_fisher_scaled_params,
                    fisher_scaled,
                    out=sum_fisher_scaled_params,
                )
                torch.add(sum_fishers, f, out=sum_fishers)

        # If an ancestor is not provided for initialzation, use the average of
        # the parameters.
        if ancestor is None:
            # The parameters are already transposed so we don't need to do it again.
            initialization = utils.interpolate(params)
        else:
            initialization = self.maybe_transpose(ancestor)

        merged_model, merged_log = self.conjugate_gradient(
            sum_fishers,
            sum_fisher_scaled_params,
            initialization,
        )
        self.merge_logger.info(json.dumps({"/".join(kwargs["param_name"]): merged_log}))

        return self.maybe_transpose(merged_model)


class CovarianceMaTS(MaTS):
    """When the fishers are computed via covariance, we transpose the parameters."""

    name = "covariance-mats"

    def maybe_transpose(self, p):
        """We transpose the parameter as the auxiliary data is the covariance."""
        return p.T


class MaTSGPU(PyTorchGPUMixin, MaTS):
    """Do it on the GPU."""

    name = "mats-gpu"


class CovarianceMaTSGPU(PyTorchGPUMixin, CovarianceMaTS):
    """Do it on the GPU!"""

    name = "covariance-mats-gpu"


class CovarianceMaTSLoRA(LoRAMixin, CovarianceMaTS):
    """Combine LoRA B @ A before merging."""

    name = "lora-covariance-mats"


class CovarianceMaTSLoRAGPU(LoRAMixin, CovarianceMaTSGPU):
    """Combine LoRA B @ A before merging."""

    name = "lora-covariance-mats-gpu"


class MaTSLoRA(LoRAMixin, MaTS):
    """Combine LoRA B @ A before merging."""

    name = "lora-mats"


class MaTSLoRAGPU(LoRAMixin, MaTSGPU):
    """Combine LoRA B @ A before merging on the GPU."""

    name = "lora-mats-gpu"
