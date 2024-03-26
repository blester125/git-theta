"""Calculate Global Values chunk-by-chunk.

These tools allow use to calculate global values, like the norm of a model or
the majority sign, without loading the whole model into memory at once. They are
basically the same as merge objects, but they have a special new `reduction`
method that takes the output of the per-chunk calculates and combines them into
the global value.
"""


from typing import Sequence

import torch
from git_theta_merge_cli import utils
from git_theta_merge_cli.merges.base import (
    PyTorchGPUMixin,
    PyTorchMixin,
    RVariadicMerge,
    RWVariadicMerge,
)


class Reduction(RVariadicMerge):
    """Calculate a global model value piecewise."""

    def run_merge(self, models_md, aux_md, ancestor_md, **kwargs):
        results = super().run_merge(models_md, aux_md, ancestor_md, **kwargs)
        return self.reduction(list(results.values()))


class Dot(PyTorchMixin, Reduction):
    """Calculate the dot product piecewise."""

    name = "dot"

    def merge(self, params, aux_data=None, ancestor=None, **kwargs):
        if len(params) != 2:
            raise ValueError(
                f"dot product is only defined for 2 values, got {len(params)}"
            )
        return (params[0] * params[1]).sum()

    def reduction(self, results):
        return utils.interpolate(results)


class DotGPU(PyTorchGPUMixin, Dot):
    name = "dot-gpu"


class NormalizedDot(Dot):
    """Calculate a normalized dot product (each model is divided by its norm)."""

    name = "normalized-dot"

    def __init__(self, *args, norms=Sequence[float], **kwargs):
        self.norms = norms
        super().__init__(*args, **kwargs)

    def merge(self, params, aux_data=None, ancestor=None, **kwargs):
        params = [p / n for p, n in zip(params, self.norms)]
        return super().merge(params, aux_data, ancestor, **kwargs)


class NormalizedDotGPU(PyTorchGPUMixin, NormalizedDot):
    name = "normalized-dot-gpu"


class Norm(Dot):
    """Calculate the norm of a model."""

    name = "norm"

    def merge(self, params, aux_data=None, ancestor=None, **kwargs):
        if len(params) != 1:
            raise ValueError(
                f"Norm calculation can only be applied to 1 model at a time, got {len(params)}"
            )
        return super().merge([params[0], params[0]])

    def reduction(self, results):
        result = super().reduction(results)
        return torch.sqrt(result)


class NormGPU(PyTorchGPUMixin, Norm):
    name = "norm-gpu"


class Sign(PyTorchMixin, Reduction):
    """Calculate the most common sign across a whole model."""

    name = "sign"

    def merge(self, params, aux_data=None, ancestor=None, **kwargs):
        sum_params = utils.interpolate(params)
        signs = torch.sign(sum_params)
        sign_total = torch.sum(signs)
        return sign_total

    def reduction(self, results):
        majority_sign = torch.sign(torch.sum(torch.Tensor([results])))
        majority_sign = majority_sign.masked_fill(majority_sign == 0, 1)
        return majority_sign


class SignGPU(PyTorchGPUMixin, Sign):
    name = "sign-gpu"
