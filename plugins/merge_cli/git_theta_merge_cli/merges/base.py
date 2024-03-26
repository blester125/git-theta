"""Base classes for variadic merges."""

import asyncio
import logging
import re
import time
from abc import ABCMeta, abstractmethod

import numpy as np

from git_theta import async_utils, git_utils, metadata, params, updates, utils


# TODO: Maybe rename this so something like VariadicMap?
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

    async def _merge(self, param_name, to_merge):
        """Helper to make the merge call run in `async_utils.run_map`."""
        # These are all parameter metadata, not real values.
        params, aux_data, ancestor = to_merge
        self.logger.info(f"Merging {'/'.join(param_name)}.")
        # If the parameter metadata is the same for all models, just return the original metadata.
        return (param_name, await self(param_name, params, aux_data, ancestor))

    def run_merge(self, models_md, aux_md, ancestor_md, **kwargs):
        """Actually runs the merge."""

        # Assumes we have matching parameter names.
        # This is a mapping from parameter name to a tuple of lists representing
        #   (parameters_md, aux data_md, ancestor_md) for each model.
        # It groups together everything we need for the merge.
        # If ancestor wasn't provided, it is a empty dict so `.get` gives a None.
        # If aux_data isn't provided at all, it is an empty list so we don't call .get
        #   If aux_data isn't provided for a parameter in the model, `.get` gives a None.
        merged_model = {
            # TODO: Convert to namedtuple?
            p: (
                [m[p] for m in models_md],
                [aux.get(p) for aux in aux_md],
                ancestor_md.get(p),
            )
            for p in models_md[0]
        }
        if kwargs.get("test_run", False):
            # TODO: Remove this? Or make the extra param to merge configurable?
            # Only merge a few parameters, also include the lm_head as it is very large and
            # often requires special code. We want to make sure merging it works.
            merged_model = {
                k: v
                for i, (k, v) in enumerate(merged_model.items())
                if i < args.test_run or k == ("transformer.lm_head.weight",)
            }
            self.logger.info(f"Only merging {len(merged_model)} parameters")

        self.logger.info("Merging the model")
        # Merge the model while limiting how many parameters we have in memory at once.
        return async_utils.run(
            async_utils.run_map(
                merged_model,
                self._merge,
                max_concurrency=kwargs.get("limit_concurrency", -1),
            )
        )


class RVariadicMerge(VariadicMerge):
    """Do a merge where the parameters are read as needed before getting merged."""

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
        return merged

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


class RWVariadicMerge(RVariadicMerge):
    """Merge parameters from an arbitray model where the parameters are loaded and written as needed."""

    async def __call__(self, param_name, params_md, aux_data_md, ancestor_md, **kwargs):
        """Load the parameters, aux data, and ancestor values as needed."""
        merged = await super().__call__(
            param_name, params_md, aux_data_md, ancestor_md, **kwargs
        )
        # Write the merged value and get metadata back.
        merged_metadata = await self.write_param(merged, param_name)
        # Return the metadata
        return merged_metadata

    async def _merge(self, param_name, to_merge):
        """Helper to make the merge call run in `async_utils.run_map`."""
        # These are all parameter metadata, not real values.
        params, aux_data, ancestor = to_merge
        self.logger.info(f"Merging {'/'.join(param_name)}.")
        # If the parameter metadata is the same for all models, just return the original metadata.
        if all(params[0] == p for p in params[1:]):
            self.logger.debug(
                f"Skipping Merge of {'/'.join(param_name)} as it is the same across models."
            )
            return (param_name, params[0])
        return (param_name, await self(param_name, params, aux_data, ancestor))

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
