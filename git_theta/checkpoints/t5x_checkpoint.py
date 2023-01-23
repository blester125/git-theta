"""Handle T5X checkpoints."""

from git_theta.checkpoints import Checkpoint
from t5x import checkpoints as t5x_checkpoints
import sys
from git_theta import utils, git_utils
import numpy as np
from file_or_name import file_or_name


class T5XCheckpoint(Checkpoint):
    @property
    def name(self):
        return "t5x"

    @classmethod
    def load(cls, checkpoint_path, true_file_name):
        return load_t5x_checkpoint(checkpoint_path, true_file_name)

    def flatten(self):
        return utils.flatten(self, is_leaf=lambda v: not isinstance(v, dict))

    def save(self, checkpoint_path):
        # TODO: How to output the checkpoint when things are partitioned?
        written_state_dict = _write_state_to_tensorstore(
            os.path.dirname(checkpoint_path), self
        )
        written_state_dict = jax.tree_util.tree_map(get_local_data, written_state_dict)
        msgpack_bytes = serialization.to_bytes(
            {"vesrion": VERSION, "optimizer": written_state_dict}
        )
        with open(checkpoint_path) as wf:
            wf.write(msgpack_bytes)

    @classmethod
    def track_action(cls, checkpoint_path):
        repo = git_utils.get_git_repo()
        gitignore = git_utils.get_gitignore_file(repo)
        git_utils.add_gitignore(
            git_ignore, [f"{os.path.dirname(checkpoint_path)}/target*"]
        )


from flax import serialization
from t5x.checkpoints import (
    _maybe_update_ts_from_gcs_to_file,
    _get_optimizer_state_dict,
    fake_param_info,
    LazyArray,
    LazyAwaitableArray,
    _ParameterInfo,
    _read_ts,
    _run_future_tree,
)
from flax import traverse_util
import jax
from t5x import state_utils
import tensorstore as ts
import functools
from typing import Optional, Any
import jax.numpy as jnp

PyTreeDef = Any


# Fork of t5x to handle the pre-open file version.
@file_or_name
def load_t5x_checkpoint(
    path: str,
    true_name: str,
    state_transformation_fns=(),
    remap: bool = True,
    restore_dtype: Optional[jnp.dtype] = None,
    lazy_parameters: bool = False,
) -> PyTreeDef:
    """Load a T5X checkpoint without pre-defining the optimizer."""
    # The msgpack file will have all the info we need about the parameter layout.
    ckpt_contents = serialization.msgpack_restore(path.read())
    ckpt_contents = _maybe_update_ts_from_gcs_to_file(ckpt_contents)

    # Remap that variable names to the most recent formatting.
    if remap:
        ckpt_optimizer_state = _get_optimizer_state_dict(
            ckpt_contents, {}, state_transformation_fns
        )
    # If we aren't remapping names we at least need to index into the checkpoint
    # file blob to make sure we are only dealing with the optimizer state.
    else:
        # Grab a subsection of the file depending on the version.
        version = ckpt_contents.get("version", 0)
        if version == 0:
            ckpt_optimizer_state = ckpt_contents
        else:
            ckpt_optimizer_state = ckpt_contents["optimizer"]

    # Replace all dicts of tensorstore specs with actual `ts.Spec`s.
    # When a checkpoint was trained using a MultiOptimizer, some of the parameter
    # states may be set to `None` (when a parameter was untouched by any
    # optimizer). We still needs references to these in our state so we keep
    # empty nodes.
    ckpt_optimizer_state_with_specs = state_utils.flatten_state_dict(
        ckpt_optimizer_state, keep_empty_nodes=True
    )
    ckpt_optimizer_state_with_specs = {
        k: ts.Spec(v) if isinstance(v, dict) else v
        for k, v in ckpt_optimizer_state_with_specs.items()
    }

    # Create fake parameter info that results in reading the whole variable.
    param_infos = {
        k: fake_param_info(v) for k, v in ckpt_optimizer_state_with_specs.items()
    }

    ckpt_optimizer_state_with_specs = traverse_util.unflatten_dict(
        ckpt_optimizer_state_with_specs, sep="/"
    )
    param_infos = traverse_util.unflatten_dict(param_infos, sep="/")

    def _create_lazy_awaitable_array(
        param_info: _ParameterInfo,
        maybe_ts_spec: Any,
        ckpt_path: str,
        restore_dtype: Optional[jnp.dtype],
    ) -> LazyAwaitableArray:
        get_fn = functools.partial(
            _read_ts,
            param_info,
            maybe_ts_spec,
            ckpt_path=ckpt_path,
            restore_dtype=restore_dtype,
        )
        return LazyAwaitableArray.from_tensor_store_spec_or_array(
            maybe_ts_spec, get_fn, dtype=restore_dtype
        )

    state_dict = jax.tree_util.tree_map(
        functools.partial(
            _create_lazy_awaitable_array,
            ckpt_path=true_name,
            restore_dtype=restore_dtype,
        ),
        param_infos,
        ckpt_optimizer_state_with_specs,
    )

    if not lazy_parameters:
        future_state_dict = jax.tree_util.tree_map(lambda x: x.get_async(), state_dict)
        state_dict = _run_future_tree(future_state_dict)

    if restore_dtype is not None:
        state_dict["target"] = _cast(state_dict["target"], restore_dtype)

    return state_dict
