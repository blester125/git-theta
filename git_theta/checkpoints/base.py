"""Base class and utilities for different checkpoint format backends."""

from abc import ABCMeta, abstractmethod
import os
import sys
from typing import Optional
import numpy as np

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from git_theta import utils


@utils.abstract_classattributes("name")
class Checkpoint(dict, metaclass=ABCMeta):
    """Abstract base class for wrapping checkpoint formats."""

    name: str = NotImplemented  # The name of this checkpoint handler, can be used to lookup the plugin.

    def __init__(self, *args, extra_info=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_info = extra_info

    @classmethod
    def from_file(cls, checkpoint_path):
        """Create a new Checkpoint object.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to a checkpoint file
        """
        weights, extra = cls.load(checkpoint_path)
        return cls(weights, extra_info=extra)

    @classmethod
    @abstractmethod
    def load(cls, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to a checkpoint file

        Returns
        -------
        model_dict : dict
            Dictionary mapping parameter names to parameter values.  Parameters
            should be numpy arrays.
        extra_info : Any
            Any extra information that should be saved as part of the checkpoint
        """

    @abstractmethod
    def save(self, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to write out the checkpoint file to
        """

    def flatten(self):
        flat = utils.flatten(self, is_leaf=lambda v: isinstance(v, np.ndarray))
        flat.extra_info = self.extra_info
        return flat

    def unflatten(self):
        unflat = utils.unflatten(self)
        unflat.extra_info = self.extra_info
        return unflat

    @property
    def extra_info(self):
        return self._extra_info

    @extra_info.setter
    def extra_info(self, extra):
        self._extra_info = extra


def get_checkpoint_handler_name(checkpoint_type: Optional[str] = None) -> str:
    """Get the name of the checkpoint handler to use.

    Order of precedence is
    1. `checkpoint_type` argument
    2. `$GIT_THETA_CHECKPOINT_TYPE` environment variable
    3. default value (currently pytorch)

    Parameters
    ----------
    checkpoint_type
        Name of the checkpoint handler

    Returns
    -------
    str
        Name of the checkpoint handler
    """
    # TODO(bdlester): Find a better way to include checkpoint type information
    # in git clean filters that are run without `git theta add`.
    # TODO: Don't default to pytorch once other checkpoint formats are supported.
    return checkpoint_type or utils.EnvVarConstants.CHECKPOINT_TYPE


def get_checkpoint_handler(checkpoint_type: Optional[str] = None) -> Checkpoint:
    """Get the checkpoint handler either by name or from an environment variable.

    Gets the checkpoint handler either for the `checkpoint_type` argument or
    `$GIT_THETA_CHECKPOINT_TYPE` environment variable.

    Defaults to pytorch when neither are defined.

    Parameters
    ----------
    checkpoint_type
        Name of the checkpoint handler

    Returns
    -------
    Checkpoint
        The checkpoint handler (usually an instance of `git_theta.checkpoints.Checkpoint`).
        Returned handler may be defined in a user installed plugin.
    """
    checkpoint_type = get_checkpoint_handler_name(checkpoint_type)
    discovered_plugins = entry_points(group="git_theta.plugins.checkpoints")
    return discovered_plugins[checkpoint_type].load()
