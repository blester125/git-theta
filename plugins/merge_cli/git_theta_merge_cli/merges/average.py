"""Merging via simple averaging, interpolation, or scale and sums."""

from typing import Optional

import numpy as np
from git_theta_merge_cli.merges import utils
from git_theta_merge_cli.merges.base import (
    LoRAMixin,
    PyTorchGPUMixin,
    PyTorchMixin,
    RWVariadicMerge,
)


class VariadicAverage(PyTorchMixin, RWVariadicMerge):
    """Merge parameters via averaging or interpolation.

    * Simple Average: set merge_lambda=None or 0.5 (for 2 models)
    * Interpolation: set merge_lambda=scalar for 2 models or set it to a list of
        values that sum to 1 for multiple models.
    * Scale and Sum: set merge_lambda to a list of scalars that needn't sum to one.
    """

    name = "average"

    def __init__(self, *args, merge_lambda: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_lambda = merge_lambda

    def merge(self, params, aux_data, ancestor, **kwargs):
        merge_lambdas = utils.get_merge_lambdas(
            self.merge_lambda, len(params), uniform=True
        )
        return utils.interpolate(params, merge_lambdas)


class VariadicAverageGPU(PyTorchGPUMixin, VariadicAverage):
    """Do it on the GPU!"""

    name = "average-gpu"


class VariadicAverageLoRA(LoRAMixin, VariadicAverage):
    """Combine LoRA B @ A before merging."""

    name = "lora-average"


class VariadicAverageLoRAGPU(LoRAMixin, VariadicAverageGPU):
    """Combine LoRA B @ A before merging on the GPU."""

    name = "lora-average-gpu"
