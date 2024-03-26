"""SLERP/MLERP merging.

Citations:
SLERP:
@article{shoemake1985slerp,
    author = {Shoemake, Ken},
    title = {Animating rotation with quaternion curves},
    year = {1985},
    issue_date = {Jul. 1985},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {19},
    number = {3},
    issn = {0097-8930},
    url = {https://doi.org/10.1145/325165.325242},
    doi = {10.1145/325165.325242},
    journal = {SIGGRAPH Comput. Graph.},
    month = {jul},
    pages = {245–254},
    numpages = {10},
}
MLERP:
@inproceedings{kim2024token,
  title={Token fusion: Bridging the gap between token pruning and token merging},
  author={Kim, Minchul and Gao, Shangqian and Hsu, Yen-Chang and Shen, Yilin and Jin, Hongxia},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1383--1392},
  year={2024}
}
"""

from typing import Optional, Sequence

import git_theta_merge_cli.merges.reductions as r
import torch
from git_theta_merge_cli import utils
from git_theta_merge_cli.merges import average
from git_theta_merge_cli.merges.base import (
    PyTorchGPUMixin,
    PyTorchMixin,
    RWVariadicMerge,
)


class SLERP(PyTorchMixin, RWVariadicMerge):
    """Merge models ..."""

    name = "slerp"

    def __init__(
        self,
        *args,
        dot: Optional[float] = None,
        merge_lambda: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dot = dot
        self.merge_lambda = merge_lambda

    def merge(self, params, aux_data, ancestor, **kwargs):
        if len(params) != 2:
            raise ValueError(
                f"SLERP merging is only defined for 2 models, got {len(params)}, use MLERP for multiple models."
            )
        merge_lambdas = utils.get_merge_lambdas(
            self.merge_lambda, len(params), uniform=True
        )

        dot = torch.clamp(torch.Tensor([self.dot]), -1.0, 1.0)
        angle = torch.acos(dot)
        angle = torch.where(angle < 1e-6, 1e-6, angle)
        sin_angle = torch.sin(angle)

        scales = [torch.sin(m * angle) / sin_angle for m in merge_lambdas]
        return utils.interpolate(params, scales)

    def _get_norm(self):
        return r.Norm

    def _get_norm_dot(self):
        return r.NormalizedDot

    def run_merge(self, models_md, aux_md, ancestor_md, **kwargs):
        if self.dot is None:
            self.logger.warning("Dot Product not pre-computed, calculating now.")

            self.logger.info("Calculating Model Norms")
            norm = self._get_norm()()
            norms = [norm.run_merge([model], [], {}) for model in models_md]
            self.logger.info(f"Model Norms: {norms}")

            self.logger.info("Calculating the dot product of the normalized models.")
            dot = self._get_norm_dot()(norms=norms)
            self.dot = dot.run_merge(models_md, aux_md, ancestor_md).item()
            self.logger.info(f"Calculated dot product as {self.dot}")

        return super().run_merge(models_md, aux_md, ancestor_md, **kwargs)


class SLERPGPU(PyTorchGPUMixin, SLERP):
    name = "slerp-gpu"

    def _get_norm(self):
        return r.NormGPU

    def _get_norm_dot(self):
        return r.NormalizedDotGPU


class MLERP(PyTorchMixin, RWVariadicMerge):
    """Merge models ..."""

    name = "mlerp"

    def __init__(
        self,
        *args,
        norms: Optional[Sequence[float]] = None,
        avg_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.norms = norms
        self.avg_norm = avg_norm

    def merge(self, params, aux_data=None, ancestor=None, **kwargs):
        if len(params) == 1:
            self.logger.info(
                "Average model metadata was passed, reloading instead of recomputing."
            )
            avg_params = params[0]
        else:
            merge_lambdas = utils.get_merge_lambdas(None, len(params), uniform=True)
            avg_params = utils.interpolate(params, merge_lambdas)
        return avg_params / self.avg_norm * torch.max(torch.Tensor(self.norms))

    def _get_norm(self):
        return r.Norm

    def _get_average(self):
        return average.VariadicAverage

    def run_merge(self, models_md, aux_md, ancestor_md, **kwargs):
        if self.norms is None:
            self.logger.warning("Model Norms not pre-computed, computing now.")

            self.logger.info("Calculating Model Norms")
            norm = r.Norm()
            self.norms = [norm.run_merge([model], [], {}) for model in models_md]
            self.logger.info(f"Model Norms: {self.norms}")

        if self.avg_norm is None:
            self.logger.warning(
                "Average model norm is not pre-computed, computing now."
            )

            self.logger.info("Computing averaged model.")
            avg_model = self._get_average()().run_merge(
                models_md, aux_md, ancestor_md, **kwargs
            )

            self.logger.info("Computing averaged model norm.")
            self.avg_norm = norm.run_merge([avg_model], [], {})
            self.logger.info(f"Averaged Model Norm: {self.avg_norm}")
            # By passing the pre-averaged model metadata, we can reload the model,
            # which is saved in git, instead of reloading each of the individual
            # models and recomputing the averaged model.
            models_md = [avg_model]
            aux_md = []
            ancestor_md = {}

        return super().run_merge(models_md, aux_md, ancestor_md, **kwargs)


class MLERPGPU(PyTorchGPUMixin, MLERP):
    def _get_norm(self):
        return r.NormGPU

    def _get_average(self):
        return average.VariadicAverageGPU
