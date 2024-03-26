"""Utils for merging parameters."""

import collections
import logging
from typing import List, Optional, Sequence, Union

import numpy as np
import torch


def is_seq(x) -> bool:
    """Check if `x` is a sequence including an array/tensor."""
    if isinstance(x, str):
        return False
    return isinstance(
        x,
        (
            collections.abc.Sequence,
            collections.abc.MappingView,
            np.ndarray,
            torch.Tensor,
        ),
    )


def get_merge_lambdas(
    merge_lambda: Union[float, Sequence[float]], size: int, uniform: bool = False
) -> List[float]:
    """Turn easy for users lambda inputs into easy to use for interpolation.

    If you have a list of merge lambdas, it checks that there is enough. Having
      a list of merge lambdas overrides arguments like uniform
    Setting `uniform=True` returns lambdas that can be used to average
      parameters. merge_lambda can be None.
    If you are merging 2 parameters and merge_lambda is a scalar it give you
      [lambda, 1 - lambda] for interpolation.
    If merge lambda is None, it returns a list of 1s to result in no scaling.
    If merge lambda is a Scalar, it is replicated to be applied to each parameter.
    """
    logger = logging.getLogger("git_theta")
    if is_seq(merge_lambda):
        if len(merge_lambda) != size:
            raise ValueError(
                "A lambda is needed for each parameter, got {len(merge_lambda)} lambdas and {size} params."
            )
        if uniform:
            logger.warning(
                "Uniform lambdas requested, but per-model lambdas provided, ignoring uniform=True."
            )
        logger.debug(f"List of merging lambdas provided {merge_lambda}, using as is.")
        return merge_lambda
    # Easy Interpolate when there are two models.
    if size == 2 and merge_lambda is not None:
        merge_lambdas = [merge_lambda, 1 - merge_lambda]
        if uniform:
            logger.warning(
                "Uniform lambdas requested, but merge_lambda provided for 2 models, ignoring uniform=True."
            )
        logger.debug("Creating interpolating lambda for 2 parameters, {merges}")
        return merge_lambdas
    # For averaging
    if uniform:
        uniform = [1 / size for _ in range(size)]
        if merge_lambda and merge_lambda != (1 / size):
            logger.warning(
                "Averaging across multiple models requested, but merge lambda doesn't match. "
                + f"Got {merge_lambda}, overriding with {uniform}."
            )
        return uniform
    if merge_lambda is None:
        logger.warning("lambda not set and uniform not requested, setting all to 1")
        merge_lambda = 1.0
    merge_lambdas = [merge_lambda for _ in range(size)]
    logger.debug(f"Replicating {merge_lambda} for each parameter: {merge_lambdas}.")
    return merge_lambdas


def interpolate(
    params: List[torch.Tensor],
    scales: Optional[List[float]] = None,
    large: int = 10_000,
) -> torch.Tensor:
    """Interpolate a list of `params` together.

    Despite the name, this can do more than just interpolate values, it can be
    used to sum them by omitting `scales`, interpolate them by passing scales
    that sum to one, average them by passing `scales=1/len(params)`, and scale
    then sum them by settings `scales`.

    If scales is a scalar, it is replicated and applied to each parameter.
    Otherwise it should be the same length as params.

    When a list of vary large parameters (they have a dimension larger than
    `large`) are combined, a memory efficient implementation is used.
    """
    logger = logging.getLogger("git_theta")
    # Check if the parameter is huge.
    if any(d > large for d in params[0].shape):
        logger.info(
            f"Using Memory efficient interpolation as the params is: {params[0].shape}"
        )
        return memory_efficient_interpolate(params, scales)

    joined = torch.stack(params, dim=0)
    # Massage scales into the right shape
    if scales:
        # If you only have one value, repeat it for all parameters
        if not is_seq(scales):
            logger.info(
                f"Replicating {scales} to apply to each of the {joined.shape[0]} parameters."
            )
            scales = [scales for _ in range(joined.shape[0])]
        # Make sure that you have a scale for each parameter
        if len(scales) != len(params):
            raise ValueError(
                f"Need a scaling factor for all params, got {len(params)} params and {len(scales)} scales."
            )
        # Scale all the parameters.
        scales = torch.Tensor(scales).view([-1] + [1 for _ in joined.shape[1:]])
        joined = joined * scales.to(joined.device)
    # Sum the parameters
    return torch.sum(joined, dim=0)


def memory_efficient_interpolate(
    params: List[torch.Tensor], scales: Optional[List[float]] = None
) -> torch.Tensor:
    """A memory efficient implementation of interpolate.

    Instead of creating a new single tensor that is [len(params), *params[0].shape]
    a single accumulator tensor is created and all additions overwrite that.

    All the same scales are supported.
    """
    logger = logging.getLogger("git_theta")
    # Massage scales into the right shape
    if scales:
        # If you only have one value, repeat it for all parameters
        if not is_seq(scales):
            logger.info(
                f"Replicating {scales} to apply to each of the {joined.shape[0]} parameters."
            )
            scales = [scales for _ in len(params)]
        # Make sure that you have a scale for each parameter
        if len(scales) != len(params):
            raise ValueError(
                f"Need a scaling factor for all params, got {len(params)} params and {len(scales)} scales."
            )
        # Scale all the parameters.
        params = [p * s for p, s in zip(params, scales)]
    # Sum the parameters while only using an extra `param` amount of memory.
    out = params[0].clone()
    for p in params[1:]:
        torch.add(out, p, out=out)
    return out
