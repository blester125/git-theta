"""Merging via Task Arithmetic. https://arxiv.org/abs/2212.04089

This also includes an implementation of DARE Task Arithmetic. https://arxiv.org/abs/2311.03099
"""

import numpy as np
import torch
import torch.nn.functional as F
from git_theta_merge_cli.merges import utils
from git_theta_merge_cli.merges.base import (
    LoRAMixin,
    PyTorchGPUMixin,
    PyTorchMixin,
    RWVariadicMerge,
)


class TaskArithmetic(PyTorchMixin, RWVariadicMerge):
    """Merge models by summing task vectors, the different between the fine-tuned and pre-trained model."""

    name = "task-arithmetic"

    def __init__(self, *args, merge_lambda: float = 0.1, **kwargs):
        """Merge with Task Arithmetic.

        Args:
          merge_lambda: Scales the task vectors.
        """
        super().__init__(*args, **kwargs)
        self.merge_lambda = merge_lambda

    def calculate_task_vectors(self, param, ancestor):
        """Calculate the task vector, the difference between the fine-tuned parameter and the pre-trained model.

        Note:
          We implement this on a per-parameter level and call it in a loop to allow
          for a future where fused task vectors are calculated in a single op.
        """
        return p - ancestor

    def merge(self, params, aux_data, ancestor, **kwargs):
        # Calculate scaling values.
        merge_lambdas = utils.get_merge_lambdas(self.merge_lambda, len(params))
        # Calculate the task vectors for each model.
        task_vectors = [self.calculate_task_vectors(p, ancestor) for p in params]
        # Scale and sum the task vectors.
        summed_vectors = utils.interpolate(task_vectors, merge_lambdas)
        # Apply the task vector to the pre-trained model.
        merged = summed_vectors + ancestor

        return merged


class TaskArithmeticGPU(PyTorchGPUMixin, TaskArithmetic):
    """Do it on the GPU!"""

    name = "task-arithmetic-gpu"


class TaskArithmeticLoRA(LoRAMixin, TaskArithmetic):
    """Combine LoRA B @ A before merging."""

    name = "lora-task-arithmetic"


class TaskArithmeticLoRAGPU(LoRAMixin, TaskArithmeticGPU):
    """Combine LoRA B @ A before merging on the GPU."""

    name = "lora-task-arithmetic-gpu"


class DARETaskArithmetic(TaskArithmetic):
    """Merge with TaskVectors, but apply dropout to the task vectors."""

    name = "dare-task-arithmetic"

    def __init__(
        self,
        *args,
        merge_lambda: float = 0.5,
        dropout_probability: float = 0.5,
        seed: int = None,
        **kwargs
    ):
        """Merge with DARE.

        Args:
          merge_lambda: Scales the task vectors.
          dropout_probability: The probability to dropout a task vector parameter.
          seed: Set a seed for dropout.
        """
        super().__init__(*args, merge_lambda=merge_lambda, **kwargs)
        self.dropout_probability = dropout_probability
        if seed:
            torch.manual_seed(seed)

    def calculate_task_vectors(self, param, ancestor):
        """Apply dropout to the task vectors."""
        task_vectors = super().calculate_task_vectors(param, ancestor)
        return F.dropout(tv, self.dropout_probability)


class DARETaskArithmeticGPU(PyTorchGPUMixin, DARETaskArithmetic):
    """Do it on the GPU!"""

    name = "dare-task-arithmetic-gpu"

    def __init__(self, *args, seed: int = None, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        # Configuration to get reproducable results on the GPU.
        if seed:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True


class DARETaskArithmeticLoRA(LoRAMixin, DARETaskArithmetic):
    """Combine LoRA B @ A before merging."""

    name = "lora-dare-task-arithmetic"


class DARETaskArithmeticLoRAGPU(LoRAMixin, DARETaskArithmeticGPU):
    """Combine LoRA B @ A before merging on the GPU."""

    name = "lora-date-task-arithmetic-gpu"
