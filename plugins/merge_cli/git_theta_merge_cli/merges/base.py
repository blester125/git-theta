"""Base classes for variadic merges."""

import asyncio
import logging
import re
import time
from abc import ABCMeta, abstractmethod

import numpy as np

from git_theta import git_utils, metadata, params, updates, utils


@utils.abstract_classattributes("name")
class VariadicMerge(metaclass=ABCMeta):
    """Merge parameters from a arbitrary number of models."""

    name: str = NotImplemented

    def __init__(self, *args, **kwargs):
        """A base __init__ to ensure that all merge sub classes can take any keyword args."""
        super().__init__()
        self.logger = logging.getLogger("git_theta")

    @abstractmethod
    def merge(self, params, aux_data, ancestor, **kwargs):
        """Code to do the actual merge, each argument is a list with values from each model."""

    async def __call__(self, *args, **kwargs):
        return self.merge(*args, **kwargs)

    def rewrite_checkpoint(self, ckpt):
        """Update checkpoint to handle things like combining LoRAs."""
        return ckpt


class RWVariadicMerge(VariadicMerge):
    """Merge parameters from an arbitray model where the parameters are loaded and written as needed."""

    async def __call__(self, param_name, params_md, aux_data_md, ancestor_md, **kwargs):
        """Load the parameters, aux data, and ancestor values as needed."""
        # Load the parameters, aux data, and stats based on metadata now that it time to merge.
        params = await asyncio.gather(
            *(self.read_param(p, param_name) for p in params_md)
        )
        aux_data = await asyncio.gather(
            *(self.read_param(p, param_name) for p in aux_data_md)
        )
        ancestor = (
            (await self.read_param(ancestor_md, param_name)) if ancestor_md else None
        )
        # There aren't any async calls inside the merge function so we can time it.
        tic = time.time()
        # Merge the real values
        merged = self.merge(params, aux_data, ancestor, param_name=param_name)
        toc = time.time()
        self.logger.info(f"Time to merge {'/'.join(param_name)}: {toc - tic} seconds.")
        # Write the merged value and get metadata back.
        merged_metadata = await self.write_param(merged, param_name)
        # Return the metadata
        return merged_metadata

    # TODO: Look at plumbing path and repo into here so we can do non-dense updates.
    async def read_param(self, param_md, param_name):
        """Read the parameter from git based on the metadata."""
        # Pass through None for cases when there isn't aux data or an ancestor.
        if param_md is None:
            return None
        # Read the parameter
        update_handler = updates.get_update_handler(
            param_md.theta_metadata.update_type
        )(params.get_update_serializer())
        return await update_handler.apply(param_md, param_name)

    async def write_param(self, param, param_name):
        """Write the parameter to git and return the metadata."""
        tensor_metadata = metadata.TensorMetadata.from_tensor(param)
        update_handler = updates.get_update_handler("dense")(
            params.get_update_serializer()
        )
        theta_metadata = metadata.ThetaMetadata("dense", None)
        # Dense only needs these two...
        lfs_metadata, _ = await update_handler.write(param, param_name)
        return metadata.ParamMetadata(
            lfs_metadata=lfs_metadata,
            tensor_metadata=tensor_metadata,
            theta_metadata=theta_metadata,
        )


class PyTorchMixin:
    """A mix in that converts all loaded parameters to pytorch."""

    async def read_param(self, param_md, param_name):
        # Import here for now because the import get cached (fast), otherwise
        # we would need to move it to a new file.
        import torch

        # Read parameter, most likely from git, and convert it to pytorch.
        p = await super().read_param(param_md, param_name)
        # Pass through None for cases when there isn't aux data or an ancestor.
        if p is None:
            return p
        return torch.Tensor(p)

    async def write_param(self, param, param_name):
        # Import here for now because the import get cached (fast), otherwise
        # we would need to move it to a new file.
        import torch

        # Convert the parameter back to numpy from pytorch and then write it.
        param = param.detach().numpy()
        return await super().write_param(param, param_name)


class PyTorchGPUMixin(PyTorchMixin):
    """A mix in that moves all loaded pytorch parameters to the GPU."""

    def __init__(self, *args, device: str = "cuda", **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    async def read_param(self, param_md, param_name):
        # Import here for now because the import get cached (fast), otherwise
        # we would need to move it to a new file.
        import torch

        # Read the parameter, it will be a pytorch tensor based on inheritance.
        p = await super().read_param(param_md, param_name)
        # Pass through None for cases when there isn't aux data or an ancestor.
        if p is None:
            return None
        # Stick it on the GPU.
        return p.to(torch.device(self.device))

    async def write_param(self, param, param_name):
        # Import here for now because the import get cached (fast), otherwise
        # we would need to move it to a new file.
        import torch

        # Bring the parameter back from the GPU to the host.
        param = param.to(torch.device("cpu"))
        # Write the now-on-cpu tensor.
        return await super().write_param(param, param_name)


class LoRAMixin:
    """Do a merge where LoRA parameters are combined before merging.

    As most statistics should probably be calculated on the combined LoRA parameters,
    thus it is probably eaiser to combine them first and then just standard merging
    implementations.
    """

    async def read_param(self, param_md, param_name):
        """Load parameters when LoRA A and B parameters are to be combined."""
        # Based on the re-writer, each param_md will be a tuple with of metadata
        # for A and then B, (A_md, B_md).
        a, b = await asyncio.gather(
            # super() doesn't work right in a comprehension.
            *(super(LoRAMixin, self).read_param(p, param_name) for p in param_md)
        )
        # Handle cases where aux_data or ancestor is None.
        if a is None or b is None:
            return None
        return b @ a

    def rewrite_checkpoint(self, ckpt):
        """Rewrite the checkpoint by grouping the A and B LoRA metadata together."""
        # TODO: make this more configurable.
        new_ckpt = {}
        ckpt = {"/".join(k): v for k, v in ckpt.items()}
        for name, value in ckpt.items():
            if "lora_A" in name:
                lora_name = re.sub(r"lora_A.default.weight$", "lora_layer.weight", name)
                b_value = ckpt[re.sub(r"lora_A", "lora_B", name)]
                new_ckpt[lora_name] = (value, b_value)
            elif "lora_B" in name:
                pass
            else:
                new_ckpt[name] = value
        return {tuple(k.split("/")): v for k, v in new_ckpt.items()}
