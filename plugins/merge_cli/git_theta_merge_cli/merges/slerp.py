"""SLERP/MLERP merging."""

import torch
from git_theta_merge_cli.merges import utils
from git_theta_merge_cli.merges.base import (
    PyTorchGPUMixin,
    PyTorchMixin,
    RWVariadicMerge,
)


class SLERP(PyTorchMixin, RWVariadicMerge):
    """Merge models ..."""

    name = "slerp"

    def __init__(self, *args, norms: Sequence[float], **kwargs):
        super().__init__(*args, **kwargs)
        self.norms = norms

    def merge(self, params, aux_data, ancestor, **kwargs):
        if len(params) != 2:
            raise ValueError(
                f"SLERP merging is only defined for 2 models, got {len(params)}, use MLERP for multiple models."
            )


class MLERP(PyTorchMixin, RWVariadicMerge):
    """Merge models ..."""

    name = "mlerp"

    def __init__(self, *args, norms: Sequence[float], avg_norm: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.norms = norms
        self.avg_norm = avg_norm

    def merge(self, params, aux_data, ancestor, **kwargs):
        ...
