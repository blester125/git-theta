"""Merging via simple averaging, interpolation, or scale and sums.

Citations:
@inproceedings{mcmahan2017communication,
  title={{Communication-Efficient Learning of Deep Networks from Decentralized Data}},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Aguera},
  booktitle={Artificial intelligence and statistics},
  year={2017},
}
@inproceedings{wortsman2022robust,
  title={{Robust Fine-Tuning of Zero-Shot Models}},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Kim, Jong Wook and Li, Mike and Kornblith, Simon and Roelofs, Rebecca and Lopes, Raphael Gontijo and Hajishirzi, Hannaneh and Farhadi, Ali and Namkoong, Hongseok and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7959--7971},
  year={2022}
}
@article{choshen2022fusing,
  title={{Fusing Finetuned Models for Better Pretraining}},
  author={Choshen, Leshem and Venezian, Elad and Slonim, Noam and Katz, Yoav},
  journal={arXiv preprint arXiv:2204.03044},
  year={2022}
}
"""

from typing import Optional

import numpy as np
from git_theta_merge_cli import utils
from git_theta_merge_cli.merges.base import (
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
